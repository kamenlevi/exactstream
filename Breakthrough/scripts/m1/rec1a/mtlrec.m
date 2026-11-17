// mtlrec.m (Breakthrough, Mon 16 Nov 2026 21:0x; MODULE 1a, docs/MODULE1A.md): record an MLX region's Metal work once, then
// replay it natively. Loaded with ctypes BEFORE `import mlx.core` (as libmtlprof, scripts/m1/check55/mtlprof.m, whose
// swizzling this copies). While recording is on, every top-level call on the concrete compute / blit encoder and command
// buffer classes is appended to one op list, in order:
//   compute: setComputePipelineState, setBuffer / setBuffers / setBufferOffset, setBytes, setThreadgroupMemoryLength,
//            dispatchThreadgroups / dispatchThreads, memoryBarrierWithScope / WithResources, waitForFence / updateFence,
//            useResource(s), endEncoding;
//   blit:    copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:, fillBuffer:range:value:, endEncoding;
//   command buffer: computeCommandEncoder(WithDispatchType), blitCommandEncoder, commit (a boundary marker only).
// Every object an op names (pipelines, buffers, fences) is retained, and inline bytes are copied, so the record stays
// valid after MLX frees its arrays. Run MLX with mx.set_cache_limit(0) while a record is alive, so MLX's allocator never
// hands a recorded buffer to a new array. Encoder methods outside this list are not seen: a replay would miss their
// state, and the bitwise test is the judge of that.
// Replay: one command buffer on a private queue; each recorded encoder is re-opened (serial dispatch if forced, else the
// recorded type), its ops re-issued in order; commit + wait. Timed: CPU encode (ns), CPU commit + wait, and the GPU's own
// start / end. Build: scripts/m1/rec1a/build_mtlrec.sh.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import <objc/runtime.h>
#include <mach/mach_time.h>
#include <stdatomic.h>
#include <string.h>
#include <stdlib.h>

enum { OP_ENC = 1, OP_BLIT, OP_END, OP_COMMIT, OP_PSO, OP_BUF, OP_BUFOFF, OP_BYTES, OP_TGMEM, OP_DTG, OP_DT, OP_BAR,
       OP_BARR, OP_FWAIT, OP_FUPD, OP_USE, OP_BCOPY, OP_BFILL };

typedef struct {
  int op;
  long i0;            // index / dispatch type / scope / usage / fill value
  long n0, n1;        // offset / length / src offset, dst offset ...
  MTLSize a, b;       // grid, threadgroup
  void *o0, *o1;      // retained objects (pso, buffer, fence, src, dst) as CFBridgingRetain pointers
  void *bytes;        // malloc'd copy for setBytes; or a retained object array for BARR / USE
  long nb;            // bytes length; or object count
} rop;

static rop *R = 0;
static long nR = 0, capR = 0;
static _Atomic int recording = 0;
static __thread int depth = 0;
static long n_enc = 0, n_disp = 0;
static long *BIX = 0, nBIX = -1;   // record indices of the setBytes ops, in order (built on demand; -1 = stale)
// A patch rule (milestone 2), applied per check: value = (((vars[var] * a + b) div d) mod m) * s + c (floor division;
// no div when d <= 0, no mod when m <= 0). Targets: type 0 / 1 / 2 = word `off` of the k-th setBytes payload as int32 /
// float32 / int64; type 3 = the byte offset of buffer binding op k (k = record index of an OP_BUF op), in the plain
// replay's copy and in every ICB command that uses that binding (re-bound per check); type 4 / 5 = an int32 / int64 word
// at byte `off` of the CONTENTS of the buffer bound by BUF op k (a per-call input array the record retains, e.g. offs).
typedef struct { long k, off, a, b, d, m, s, c; int type, var; } prule;
static prule *RU = 0;
static long nRU = 0, capRU = 0;

// Bind-time snapshots (milestone 2): the contents of each small shared buffer at its FIRST binding in a record, i.e.
// before the GPU runs, so CPU-written per-call inputs are seen even if the graph later consumes them in place.
#define SNAPMAX 4096
static long *SNOP = 0, nSN = 0, capSN = 0;     // record index of the BUF op that first bound the buffer
static void **SNBUF = 0;                         // the buffer pointer (identity) per snapshot
static char **SNDAT = 0;
static long *SNLEN = 0;
static rop *push(int op);
static void snap_bind(id b, long opidx) {
  if (!b) return;
  id<MTLBuffer> mb = (id<MTLBuffer>)b;
  if (mb.storageMode == MTLStorageModePrivate || mb.length > SNAPMAX) return;
  void *key = (__bridge void *)b;
  for (long i = nSN - 1; i >= 0 && i >= nSN - 4096; i--) if (SNBUF[i] == key) return;   // first binding only (recent)
  if (nSN == capSN) {
    capSN = capSN ? capSN * 2 : 256;
    SNOP = realloc(SNOP, sizeof(long) * (size_t)capSN); SNBUF = realloc(SNBUF, sizeof(void *) * (size_t)capSN);
    SNDAT = realloc(SNDAT, sizeof(char *) * (size_t)capSN); SNLEN = realloc(SNLEN, sizeof(long) * (size_t)capSN);
  }
  SNOP[nSN] = opidx; SNBUF[nSN] = key; SNLEN[nSN] = (long)mb.length;
  SNDAT[nSN] = malloc(mb.length ? mb.length : 1);
  memcpy(SNDAT[nSN], mb.contents, mb.length);
  nSN++;
}
static rop *push(int op) {
  if (nR == capR) {
    capR = capR ? capR * 2 : 4096;
    R = realloc(R, capR * sizeof(rop));
  }
  rop *r = &R[nR++];
  memset(r, 0, sizeof(rop));
  r->op = op;
  return r;
}
static void *keep(id o) { return o ? (void *)CFBridgingRetain(o) : 0; }
static inline int rec(void) { return depth == 0 && atomic_load_explicit(&recording, memory_order_relaxed); }

// ---- originals
typedef void (*disp_t)(id, SEL, MTLSize, MTLSize);
typedef void (*v1_t)(id, SEL, NSUInteger);
typedef void (*vres_t)(id, SEL, const id *, NSUInteger);
typedef void (*vobj_t)(id, SEL, id);
typedef void (*vbuf_t)(id, SEL, id, NSUInteger, NSUInteger);
typedef void (*vbufs_t)(id, SEL, const id *, const NSUInteger *, NSRange);
typedef void (*vbufoff_t)(id, SEL, NSUInteger, NSUInteger);
typedef void (*vbytes_t)(id, SEL, const void *, NSUInteger, NSUInteger);
typedef void (*vtg_t)(id, SEL, NSUInteger, NSUInteger);
typedef void (*vuse_t)(id, SEL, id, NSUInteger);
typedef void (*vuses_t)(id, SEL, const id *, NSUInteger, NSUInteger);
typedef void (*v0_t)(id, SEL);
typedef id (*ret1_t)(id, SEL, NSUInteger);
typedef id (*ret0_t)(id, SEL);
typedef void (*bcopy_t)(id, SEL, id, NSUInteger, id, NSUInteger, NSUInteger);
typedef void (*bfill_t)(id, SEL, id, NSRange, uint8_t);
typedef id (*psod_t)(id, SEL, id, NSUInteger, id *, NSError **);
typedef id (*psof_t)(id, SEL, id, NSError **);
typedef id (*psofo_t)(id, SEL, id, NSUInteger, id *, NSError **);
static psod_t o_psod;
static psof_t o_psof;
static psofo_t o_psofo;
static NSMutableDictionary *PSOFN = nil;   // NSValue(pso pointer) -> @[function, maxTotalThreadsPerThreadgroup, tgMultiple]
static void note_pso(id pso, id fn, NSUInteger maxT, BOOL mult) {
  if (!pso || !fn) return;
  @synchronized(PSOFN) { PSOFN[[NSValue valueWithPointer:(__bridge void *)pso]] = @[fn, @(maxT), @(mult), pso]; }
}
static id h_psod(id s, SEL c, id desc, NSUInteger opt, id *refl, NSError **err) {
  id r = o_psod(s, c, desc, opt, refl, err);
  @try {
    MTLComputePipelineDescriptor *d = (MTLComputePipelineDescriptor *)desc;
    note_pso(r, d.computeFunction, d.maxTotalThreadsPerThreadgroup, d.threadGroupSizeIsMultipleOfThreadExecutionWidth);
  } @catch (id ex) {}
  return r;
}
static id h_psof(id s, SEL c, id fn, NSError **err) {
  id r = o_psof(s, c, fn, err);
  note_pso(r, fn, 0, NO);
  return r;
}
static id h_psofo(id s, SEL c, id fn, NSUInteger opt, id *refl, NSError **err) {
  id r = o_psofo(s, c, fn, opt, refl, err);
  note_pso(r, fn, 0, NO);
  return r;
}
static disp_t o_dtg, o_dt;
static v1_t o_bar;
static vres_t o_barr;
static vobj_t o_setpso, o_fwait, o_fupd;
static vbuf_t o_buf;
static vbufs_t o_bufs;
static vbufoff_t o_bufoff;
static vbytes_t o_bytes;
static vtg_t o_tg;
static vuse_t o_use;
static vuses_t o_uses;
static v0_t o_cend, o_bend, o_commit;
static ret1_t o_enc;
static ret0_t o_enc0, o_blit;
static bcopy_t o_bcopy;
static bfill_t o_bfill;

// ---- compute encoder hooks
static void h_dtg(id s, SEL c, MTLSize a, MTLSize b) {
  if (rec()) { rop *r = push(OP_DTG); r->a = a; r->b = b; n_disp++; }
  depth++; o_dtg(s, c, a, b); depth--;
}
static void h_dt(id s, SEL c, MTLSize a, MTLSize b) {
  if (rec()) { rop *r = push(OP_DT); r->a = a; r->b = b; n_disp++; }
  depth++; o_dt(s, c, a, b); depth--;
}
static void h_bar(id s, SEL c, NSUInteger scope) {
  if (rec()) push(OP_BAR)->i0 = (long)scope;
  depth++; o_bar(s, c, scope); depth--;
}
static void h_barr(id s, SEL c, const id *res, NSUInteger n) {
  if (rec()) {
    rop *r = push(OP_BARR);
    void **arr = malloc(sizeof(void *) * (n ? n : 1));
    for (NSUInteger i = 0; i < n; i++) arr[i] = keep(res[i]);
    r->bytes = arr; r->nb = (long)n;
  }
  depth++; o_barr(s, c, res, n); depth--;
}
static void h_setpso(id s, SEL c, id pso) {
  if (rec()) push(OP_PSO)->o0 = keep(pso);
  depth++; o_setpso(s, c, pso); depth--;
}
static void h_fwait(id s, SEL c, id f) {
  if (rec()) push(OP_FWAIT)->o0 = keep(f);
  depth++; o_fwait(s, c, f); depth--;
}
static void h_fupd(id s, SEL c, id f) {
  if (rec()) push(OP_FUPD)->o0 = keep(f);
  depth++; o_fupd(s, c, f); depth--;
}
static void h_buf(id s, SEL c, id b, NSUInteger off, NSUInteger idx) {
  if (rec()) { rop *r = push(OP_BUF); r->o0 = keep(b); r->n0 = (long)off; r->i0 = (long)idx; snap_bind(b, nR - 1); }
  depth++; o_buf(s, c, b, off, idx); depth--;
}
static void h_bufs(id s, SEL c, const id *bs, const NSUInteger *offs, NSRange rg) {
  if (rec())
    for (NSUInteger i = 0; i < rg.length; i++) {
      rop *r = push(OP_BUF); r->o0 = keep(bs[i]); r->n0 = (long)offs[i]; r->i0 = (long)(rg.location + i);
      snap_bind(bs[i], nR - 1);
    }
  depth++; o_bufs(s, c, bs, offs, rg); depth--;
}
static void h_bufoff(id s, SEL c, NSUInteger off, NSUInteger idx) {
  if (rec()) { rop *r = push(OP_BUFOFF); r->n0 = (long)off; r->i0 = (long)idx; }
  depth++; o_bufoff(s, c, off, idx); depth--;
}
static void h_bytes(id s, SEL c, const void *p, NSUInteger len, NSUInteger idx) {
  if (rec()) {
    rop *r = push(OP_BYTES);
    r->bytes = malloc(len ? len : 1); memcpy(r->bytes, p, len); r->nb = (long)len; r->i0 = (long)idx;
  }
  depth++; o_bytes(s, c, p, len, idx); depth--;
}
static void h_tg(id s, SEL c, NSUInteger len, NSUInteger idx) {
  if (rec()) { rop *r = push(OP_TGMEM); r->n0 = (long)len; r->i0 = (long)idx; }
  depth++; o_tg(s, c, len, idx); depth--;
}
static void h_use(id s, SEL c, id res, NSUInteger usage) {
  if (rec()) { rop *r = push(OP_USE); void **arr = malloc(sizeof(void *)); arr[0] = keep(res); r->bytes = arr; r->nb = 1; r->i0 = (long)usage; }
  depth++; o_use(s, c, res, usage); depth--;
}
static void h_uses(id s, SEL c, const id *res, NSUInteger n, NSUInteger usage) {
  if (rec()) {
    rop *r = push(OP_USE);
    void **arr = malloc(sizeof(void *) * (n ? n : 1));
    for (NSUInteger i = 0; i < n; i++) arr[i] = keep(res[i]);
    r->bytes = arr; r->nb = (long)n; r->i0 = (long)usage;
  }
  depth++; o_uses(s, c, res, n, usage); depth--;
}
static void h_cend(id s, SEL c) {
  if (rec()) push(OP_END);
  depth++; o_cend(s, c); depth--;
}
// ---- blit encoder hooks
static void h_bend(id s, SEL c) {
  if (rec()) push(OP_END);
  depth++; o_bend(s, c); depth--;
}
static void h_bcopy(id s, SEL c, id src, NSUInteger so, id dst, NSUInteger d0, NSUInteger size) {
  if (rec()) { rop *r = push(OP_BCOPY); r->o0 = keep(src); r->o1 = keep(dst); r->n0 = (long)so; r->n1 = (long)d0; r->nb = (long)size; }
  depth++; o_bcopy(s, c, src, so, dst, d0, size); depth--;
}
static void h_bfill(id s, SEL c, id b, NSRange rg, uint8_t v) {
  if (rec()) { rop *r = push(OP_BFILL); r->o0 = keep(b); r->n0 = (long)rg.location; r->n1 = (long)rg.length; r->i0 = v; }
  depth++; o_bfill(s, c, b, rg, v); depth--;
}
// ---- command buffer hooks
static id h_enc(id s, SEL c, NSUInteger t) {
  if (rec()) { push(OP_ENC)->i0 = (long)t; n_enc++; }
  depth++; id r = o_enc(s, c, t); depth--;
  return r;
}
static id h_enc0(id s, SEL c) {
  if (rec()) { push(OP_ENC)->i0 = (long)MTLDispatchTypeSerial; n_enc++; }
  depth++; id r = o_enc0(s, c); depth--;
  return r;
}
static id h_blit(id s, SEL c) {
  if (rec()) push(OP_BLIT);
  depth++; id r = o_blit(s, c); depth--;
  return r;
}
static void h_commit(id s, SEL c) {
  if (rec()) push(OP_COMMIT);
  depth++; o_commit(s, c); depth--;
}

static int hooked = 0;
static IMP hook(Class cls, SEL sel, IMP imp, const char *what) {
  Method m = class_getInstanceMethod(cls, sel);
  if (!m) {
    fprintf(stderr, "mtlrec: %s: %s not found on %s\n", what, sel_getName(sel), class_getName(cls));
    return 0;
  }
  hooked++;
  return method_setImplementation(m, imp);
}

static id<MTLDevice> DEV = 0;
static id<MTLCommandQueue> RQ = 0;

__attribute__((constructor)) static void mtlrec_init(void) {
  @autoreleasepool {
    DEV = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [DEV newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    Class ce = object_getClass(enc), cbc = object_getClass(cb);
    o_dtg = (disp_t)hook(ce, @selector(dispatchThreadgroups:threadsPerThreadgroup:), (IMP)h_dtg, "dtg");
    o_dt = (disp_t)hook(ce, @selector(dispatchThreads:threadsPerThreadgroup:), (IMP)h_dt, "dt");
    o_bar = (v1_t)hook(ce, @selector(memoryBarrierWithScope:), (IMP)h_bar, "bar");
    o_barr = (vres_t)hook(ce, @selector(memoryBarrierWithResources:count:), (IMP)h_barr, "barr");
    o_setpso = (vobj_t)hook(ce, @selector(setComputePipelineState:), (IMP)h_setpso, "pso");
    o_fwait = (vobj_t)hook(ce, @selector(waitForFence:), (IMP)h_fwait, "fwait");
    o_fupd = (vobj_t)hook(ce, @selector(updateFence:), (IMP)h_fupd, "fupd");
    o_buf = (vbuf_t)hook(ce, @selector(setBuffer:offset:atIndex:), (IMP)h_buf, "buf");
    o_bufs = (vbufs_t)hook(ce, @selector(setBuffers:offsets:withRange:), (IMP)h_bufs, "bufs");
    o_bufoff = (vbufoff_t)hook(ce, @selector(setBufferOffset:atIndex:), (IMP)h_bufoff, "bufoff");
    o_bytes = (vbytes_t)hook(ce, @selector(setBytes:length:atIndex:), (IMP)h_bytes, "bytes");
    o_tg = (vtg_t)hook(ce, @selector(setThreadgroupMemoryLength:atIndex:), (IMP)h_tg, "tg");
    o_use = (vuse_t)hook(ce, @selector(useResource:usage:), (IMP)h_use, "use");
    o_uses = (vuses_t)hook(ce, @selector(useResources:count:usage:), (IMP)h_uses, "uses");
    o_cend = (v0_t)hook(ce, @selector(endEncoding), (IMP)h_cend, "cend");
    [enc endEncoding];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    Class bc = object_getClass(be);
    o_bend = (v0_t)hook(bc, @selector(endEncoding), (IMP)h_bend, "bend");
    o_bcopy = (bcopy_t)hook(bc, @selector(copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:), (IMP)h_bcopy, "bcopy");
    o_bfill = (bfill_t)hook(bc, @selector(fillBuffer:range:value:), (IMP)h_bfill, "bfill");
    [be endEncoding];
    o_enc = (ret1_t)hook(cbc, @selector(computeCommandEncoderWithDispatchType:), (IMP)h_enc, "enc");
    o_enc0 = (ret0_t)hook(cbc, @selector(computeCommandEncoder), (IMP)h_enc0, "enc0");
    o_blit = (ret0_t)hook(cbc, @selector(blitCommandEncoder), (IMP)h_blit, "blit");
    o_commit = (v0_t)hook(cbc, @selector(commit), (IMP)h_commit, "commit");
    PSOFN = [NSMutableDictionary new];
    Class dc = object_getClass(DEV);
    o_psod = (psod_t)hook(dc, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), (IMP)h_psod, "psod");
    o_psof = (psof_t)hook(dc, @selector(newComputePipelineStateWithFunction:error:), (IMP)h_psof, "psof");
    o_psofo = (psofo_t)hook(dc, @selector(newComputePipelineStateWithFunction:options:reflection:error:), (IMP)h_psofo, "psofo");
    RQ = [DEV newCommandQueue];
    fprintf(stderr, "mtlrec: %d hooks on %s / %s / %s\n", hooked, class_getName(ce), class_getName(bc), class_getName(cbc));
  }
}

// ---- Python API (ctypes)
static void free_ops(void) {
  for (long i = 0; i < nR; i++) {
    rop *r = &R[i];
    if (r->o0) CFRelease(r->o0);
    if (r->o1) CFRelease(r->o1);
    if (r->op == OP_BARR || r->op == OP_USE) {
      void **arr = (void **)r->bytes;
      for (long k = 0; k < r->nb; k++) if (arr[k]) CFRelease(arr[k]);
    }
    free(r->bytes);
  }
  nR = 0; n_enc = 0; n_disp = 0;
  nBIX = -1; nRU = 0;
  for (long i = 0; i < nSN; i++) free(SNDAT[i]);
  nSN = 0;
}
void rec_start(void) { free_ops(); atomic_store(&recording, 1); }
void rec_stop(void) { atomic_store(&recording, 0); }
void rec_clear(void) { atomic_store(&recording, 0); free_ops(); }
// per op type: o[k] = number of ops with op == k, k = 0..19
void rec_opcounts(long *o) {
  for (int k = 0; k < 20; k++) o[k] = 0;
  for (long i = 0; i < nR; i++) if (R[i].op >= 0 && R[i].op < 20) o[R[i].op]++;
}
// counts: ops, encoders, dispatches, blits, commits, setBytes bytes total
void rec_counts(long *o) {
  long nb = 0, nbl = 0, nc = 0, by = 0;
  for (long i = 0; i < nR; i++) {
    if (R[i].op == OP_BLIT) nbl++;
    if (R[i].op == OP_COMMIT) nc++;
    if (R[i].op == OP_BYTES) { nb++; by += R[i].nb; }
  }
  o[0] = nR; o[1] = n_enc; o[2] = n_disp; o[3] = nbl; o[4] = nc; o[5] = by;
}

static inline double ns_now(void) {
  static mach_timebase_info_data_t tb;
  if (!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom;
}

// Replay the record. serial: 0 = each recorded encoder as recorded; 1 = each encoder serial (barriers kept);
// 2 = MERGED: one concurrent compute encoder for the whole record, the recorded barriers kept, a full buffer barrier
// at every original encoder boundary, fences dropped (blits, if any, still split it).
// out (doubles): [0] CPU ns to encode, [1] CPU ns commit -> completed, [2] GPU ns (GPUEndTime - GPUStartTime),
// [3] status (MTLCommandBufferStatus), [4] ops issued. Returns 0 on success, -1 on an error.
int rec_replay(int serial, double *out) {
  @autoreleasepool {
    double t0 = ns_now();
    id<MTLCommandBuffer> cb = [RQ commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> ce = nil;
    id<MTLBlitCommandEncoder> be = nil;
    long issued = 0;
    depth++;   // our own encoder calls go straight through the hooks
    for (long i = 0; i < nR; i++) {
      rop *r = &R[i];
      switch (r->op) {
        case OP_ENC:
          if (serial == 2) {
            if (!ce) ce = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            else [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
          } else {
            ce = [cb computeCommandEncoderWithDispatchType:(serial ? MTLDispatchTypeSerial : (MTLDispatchType)r->i0)];
          }
          break;
        case OP_BLIT:
          if (ce) { [ce endEncoding]; ce = nil; }
          be = [cb blitCommandEncoder];
          break;
        case OP_END:
          if (be) { [be endEncoding]; be = nil; }
          else if (ce && serial != 2) { [ce endEncoding]; ce = nil; }
          break;
        case OP_COMMIT: break;   // one command buffer for the whole replay
        case OP_PSO: [ce setComputePipelineState:(__bridge id<MTLComputePipelineState>)r->o0]; break;
        case OP_BUF: [ce setBuffer:(__bridge id<MTLBuffer>)r->o0 offset:(NSUInteger)r->n0 atIndex:(NSUInteger)r->i0]; break;
        case OP_BUFOFF: [ce setBufferOffset:(NSUInteger)r->n0 atIndex:(NSUInteger)r->i0]; break;
        case OP_BYTES: [ce setBytes:r->bytes length:(NSUInteger)r->nb atIndex:(NSUInteger)r->i0]; break;
        case OP_TGMEM: [ce setThreadgroupMemoryLength:(NSUInteger)r->n0 atIndex:(NSUInteger)r->i0]; break;
        case OP_DTG: [ce dispatchThreadgroups:r->a threadsPerThreadgroup:r->b]; issued++; break;
        case OP_DT: [ce dispatchThreads:r->a threadsPerThreadgroup:r->b]; issued++; break;
        case OP_BAR: [ce memoryBarrierWithScope:(MTLBarrierScope)r->i0]; break;
        case OP_BARR: [ce memoryBarrierWithResources:(__unsafe_unretained id<MTLResource> *)r->bytes count:(NSUInteger)r->nb]; break;
        case OP_FWAIT: if (ce && serial != 2) [ce waitForFence:(__bridge id<MTLFence>)r->o0]; break;
        case OP_FUPD: if (ce && serial != 2) [ce updateFence:(__bridge id<MTLFence>)r->o0]; break;
        case OP_USE:
          if (r->nb == 1) [ce useResource:(__bridge id<MTLResource>)((void **)r->bytes)[0] usage:(MTLResourceUsage)r->i0];
          else [ce useResources:(__unsafe_unretained id<MTLResource> *)r->bytes count:(NSUInteger)r->nb usage:(MTLResourceUsage)r->i0];
          break;
        case OP_BCOPY:
          [be copyFromBuffer:(__bridge id<MTLBuffer>)r->o0 sourceOffset:(NSUInteger)r->n0 toBuffer:(__bridge id<MTLBuffer>)r->o1
           destinationOffset:(NSUInteger)r->n1 size:(NSUInteger)r->nb];
          break;
        case OP_BFILL: [be fillBuffer:(__bridge id<MTLBuffer>)r->o0 range:NSMakeRange((NSUInteger)r->n0, (NSUInteger)r->n1) value:(uint8_t)r->i0]; break;
      }
    }
    if (ce) [ce endEncoding];
    if (be) [be endEncoding];
    double t1 = ns_now();
    [cb commit];
    [cb waitUntilCompleted];
    depth--;
    double t2 = ns_now();
    out[0] = t1 - t0; out[1] = t2 - t1; out[2] = (cb.GPUEndTime - cb.GPUStartTime) * 1e9;
    out[3] = (double)cb.status; out[4] = (double)issued;
    return cb.status == MTLCommandBufferStatusCompleted ? 0 : -1;
  }
}

// ---- ICB form (mode 2's semantics in one indirect command buffer)
static id<MTLIndirectCommandBuffer> ICB = nil;
static id<MTLBuffer> CONSTB = nil;
static NSMutableArray *RES = nil;            // every buffer the ICB's commands reference (for useResources)
static NSMutableDictionary *ICBPSO = nil;    // NSValue(recorded pso) -> ICB-capable pso
static long ICBN = 0;
static long *SEGE = NULL, nSEGE = 0;         // ICB command index where each recorded command buffer ended (segments)
static __unsafe_unretained id<MTLResource> *RESARR = NULL;   // RES keeps them alive
static NSUInteger RESN = 0;
static char icb_err[512];
const char *rec_icb_error(void) { return icb_err; }

static id<MTLComputePipelineState> icb_pso(void *recorded) {
  NSValue *k = [NSValue valueWithPointer:recorded];
  id<MTLComputePipelineState> p = ICBPSO[k];
  if (p) return p;
  NSArray *e = nil;
  @synchronized(PSOFN) { e = PSOFN[k]; }
  if (!e) return nil;
  MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
  d.computeFunction = e[0];
  if ([e[1] unsignedIntegerValue]) d.maxTotalThreadsPerThreadgroup = [e[1] unsignedIntegerValue];
  d.threadGroupSizeIsMultipleOfThreadExecutionWidth = [e[2] boolValue];
  d.supportIndirectCommandBuffers = YES;
  NSError *err = nil;
  p = [DEV newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&err];
  if (!p) { snprintf(icb_err, sizeof icb_err, "pso: %s", err ? [[err description] UTF8String] : "nil"); return nil; }
  ICBPSO[k] = p;
  return p;
}

// Build the ICB from the record. Returns the command count, or < 0 (rec_icb_error says why).
long rec_build_icb(void) {
  @autoreleasepool {
    icb_err[0] = 0;
    if (!ICBPSO) ICBPSO = [NSMutableDictionary new];
    long nd = 0, cb = 0;
    for (long i = 0; i < nR; i++) {
      if (R[i].op == OP_DTG || R[i].op == OP_DT) nd++;
      if (R[i].op == OP_BYTES) cb += (R[i].nb + 255) & ~255L;
      if (R[i].op == OP_BLIT || R[i].op == OP_BCOPY || R[i].op == OP_BFILL) { snprintf(icb_err, sizeof icb_err, "blit ops cannot go in an ICB"); return -2; }
    }
    if (!nd) { snprintf(icb_err, sizeof icb_err, "no dispatches"); return -3; }
    CONSTB = [DEV newBufferWithLength:(cb ? cb : 256) options:MTLResourceStorageModeShared];
    MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
    d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
    d.inheritPipelineState = NO;
    d.inheritBuffers = NO;
    d.maxKernelBufferBindCount = 31;
    d.maxKernelThreadgroupMemoryBindCount = 31;
    ICB = [DEV newIndirectCommandBufferWithDescriptor:d maxCommandCount:(NSUInteger)nd options:MTLResourceStorageModeShared];
    if (!ICB) { snprintf(icb_err, sizeof icb_err, "newIndirectCommandBuffer failed"); return -4; }
    RES = [NSMutableArray new];
    NSMutableSet *seen = [NSMutableSet new];
    [RES addObject:CONSTB];
    id bbuf[31]; NSUInteger boff[31], tgl[31]; int bset[31], tset[31];
    long bsrc[31];                                 // record index of the BUF op behind each binding (-1: constants / none)
    memset(bset, 0, sizeof bset); memset(tset, 0, sizeof tset);
    for (int k = 0; k < 31; k++) { bbuf[k] = nil; bsrc[k] = -1; }
    for (long i = 0; i < nR; i++)                  // reset the BUF ops' ICB use lists (a rebuild starts over)
      if (R[i].op == OP_BUF) { free(R[i].bytes); R[i].bytes = 0; R[i].nb = 0; }
    id<MTLComputePipelineState> pso = nil;
    long ci = 0, coff = 0;
    int bar = 0, any = 0;
    free(SEGE); SEGE = malloc(sizeof(long) * (size_t)(nR + 1)); nSEGE = 0;
    char *cbase = (char *)CONSTB.contents;
    for (long i = 0; i < nR; i++) {
      rop *r = &R[i];
      switch (r->op) {
        case OP_ENC:
          if (any) bar = 1;                        // mode 2: a full barrier at every original encoder boundary
          memset(bset, 0, sizeof bset); memset(tset, 0, sizeof tset);
          for (int k = 0; k < 31; k++) { bbuf[k] = nil; bsrc[k] = -1; }
          pso = nil;
          break;
        case OP_PSO:
          pso = icb_pso(r->o0);
          if (!pso) { if (!icb_err[0]) snprintf(icb_err, sizeof icb_err, "no function recorded for a pipeline (created before the hook?)"); return -5; }
          break;
        case OP_BUF:
          if (r->i0 < 31) { bbuf[r->i0] = (__bridge id)r->o0; boff[r->i0] = (NSUInteger)r->n0; bset[r->i0] = 1; bsrc[r->i0] = i; }
          if (r->o0 && ![seen containsObject:[NSValue valueWithPointer:r->o0]]) {
            [seen addObject:[NSValue valueWithPointer:r->o0]]; [RES addObject:(__bridge id)r->o0];
          }
          break;
        case OP_BUFOFF: if (r->i0 < 31) boff[r->i0] = (NSUInteger)r->n0; break;
        case OP_BYTES:
          memcpy(cbase + coff, r->bytes, (size_t)r->nb);
          if (r->i0 < 31) { bbuf[r->i0] = CONSTB; boff[r->i0] = (NSUInteger)coff; bset[r->i0] = 1; bsrc[r->i0] = -1; }
          r->n1 = coff + 1;                        // its slot in the constants buffer (+1; 0 = none), for patching
          coff += (r->nb + 255) & ~255L;
          break;
        case OP_TGMEM: if (r->i0 < 31) { tgl[r->i0] = (NSUInteger)r->n0; tset[r->i0] = 1; } break;
        case OP_BAR: case OP_BARR: bar = 1; break;
        case OP_DTG: case OP_DT: {
          id<MTLIndirectComputeCommand> c = [ICB indirectComputeCommandAtIndex:(NSUInteger)ci];
          [c setComputePipelineState:pso];
          for (int k = 0; k < 31; k++) if (bset[k]) [c setKernelBuffer:bbuf[k] offset:boff[k] atIndex:(NSUInteger)k];
          for (int k = 0; k < 31; k++)
            if (bset[k] && bsrc[k] >= 0) {         // note this command in the BUF op's use list (for offset rules)
              rop *u = &R[bsrc[k]];
              u->bytes = realloc(u->bytes, sizeof(int32_t) * 2 * (size_t)(u->nb + 1));
              ((int32_t *)u->bytes)[2 * u->nb] = (int32_t)ci; ((int32_t *)u->bytes)[2 * u->nb + 1] = k;
              u->nb++;
            }
          for (int k = 0; k < 31; k++) if (tset[k]) [c setThreadgroupMemoryLength:tgl[k] atIndex:(NSUInteger)k];
          if (bar) { [c setBarrier]; bar = 0; }
          if (r->op == OP_DTG) [c concurrentDispatchThreadgroups:r->a threadsPerThreadgroup:r->b];
          else [c concurrentDispatchThreads:r->a threadsPerThreadgroup:r->b];
          ci++; any = 1;
          break;
        }
        case OP_COMMIT:                            // a segment boundary for the segmented replay (empty ones skipped)
          if (ci > (nSEGE ? SEGE[nSEGE - 1] : 0)) SEGE[nSEGE++] = ci;
          break;
        default: break;                            // END / fences / USE: mode 2 drops them
      }
    }
    ICBN = ci;
    if (ci > (nSEGE ? SEGE[nSEGE - 1] : 0)) SEGE[nSEGE++] = ci;   // the tail after the last recorded commit
    free(RESARR);
    RESN = [RES count];
    RESARR = (__unsafe_unretained id<MTLResource> *)malloc(sizeof(id) * (RESN ? RESN : 1));
    for (NSUInteger k = 0; k < RESN; k++) RESARR[k] = RES[k];
    return ci;
  }
}

// Replay the ICB: one compute encoder, useResources (array prebuilt at build), one execute; commit + wait. out as
// rec_replay (CPU ns to encode, CPU ns commit -> completed, GPU ns, status, commands). Returns 0 on success.
int rec_replay_icb(double *out) {
  @autoreleasepool {
    if (!ICB || !ICBN) return -9;
    double t0 = ns_now();
    id<MTLCommandBuffer> cb = [RQ commandBufferWithUnretainedReferences];
    depth++;
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [ce useResources:RESARR count:RESN usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
    [ce executeCommandsInBuffer:ICB withRange:NSMakeRange(0, (NSUInteger)ICBN)];
    [ce endEncoding];
    double t1 = ns_now();
    [cb commit];
    [cb waitUntilCompleted];
    depth--;
    double t2 = ns_now();
    out[0] = t1 - t0; out[1] = t2 - t1; out[2] = (cb.GPUEndTime - cb.GPUStartTime) * 1e9;
    out[3] = (double)cb.status; out[4] = (double)ICBN;
    return cb.status == MTLCommandBufferStatusCompleted ? 0 : -1;
  }
}

// ---- residency set (macOS 15+): built once from the ICB's resources, attached to the replay queue; then a replay needs
// no useResources. rec_build_rset returns the allocation count, or < 0.
static id<MTLResidencySet> RSET = nil;
long rec_build_rset(void) {
  @autoreleasepool {
    if (!RES || !RESN) return -1;
    MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
    rd.initialCapacity = RESN;
    NSError *e = nil;
    RSET = [DEV newResidencySetWithDescriptor:rd error:&e];
    if (!RSET) { snprintf(icb_err, sizeof icb_err, "residency set: %s", e ? [[e description] UTF8String] : "nil"); return -2; }
    for (NSUInteger k = 0; k < RESN; k++) [RSET addAllocation:(id<MTLAllocation>)RES[k]];
    [RSET addAllocation:(id<MTLAllocation>)ICB];
    [RSET commit];
    [RSET requestResidency];
    [RQ addResidencySet:RSET];
    return (long)RSET.allocationCount;
  }
}
// ICB replay with the per-replay CPU split. use_rs: 1 = rely on the queue's residency set (no useResources).
// out (doubles): [0] encode total ns, [1] commit -> completed ns, [2] GPU ns, [3] status, [4] commands,
// [5] command buffer ns, [6] encoder ns, [7] useResources ns, [8] execute + endEncoding ns.
int rec_replay_icb2(int use_rs, double *out) {
  @autoreleasepool {
    if (!ICB || !ICBN) return -9;
    double t0 = ns_now();
    id<MTLCommandBuffer> cb = [RQ commandBufferWithUnretainedReferences];
    double t1 = ns_now();
    depth++;
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    double t2 = ns_now();
    if (!use_rs) [ce useResources:RESARR count:RESN usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
    double t3 = ns_now();
    [ce executeCommandsInBuffer:ICB withRange:NSMakeRange(0, (NSUInteger)ICBN)];
    [ce endEncoding];
    double t4 = ns_now();
    [cb commit];
    [cb waitUntilCompleted];
    depth--;
    double t5 = ns_now();
    out[0] = t4 - t0; out[1] = t5 - t4; out[2] = (cb.GPUEndTime - cb.GPUStartTime) * 1e9; out[3] = (double)cb.status;
    out[4] = (double)ICBN; out[5] = t1 - t0; out[6] = t2 - t1; out[7] = t3 - t2; out[8] = t4 - t3;
    return cb.status == MTLCommandBufferStatusCompleted ? 0 : -1;
  }
}
long rec_icb_nres(void) { return (long)RESN; }

// ---- setBytes inspection and per-check patching (MODULE 1a milestone 2). The engine's per-pass values (the pass's
// seq, the table and region offsets derived from it) ride in setBytes payloads; a replay must rewrite them per check.
// Find them by diffing two records made at different seqs (rec_bytes_meta + the payload dump), then register one rule
// per word; rec_rules_apply(seq) rewrites every rule in ONE call, in the recorded copies (plain replays) and in the ICB's
// constants buffer (if built). Patch only while no replay is in flight.
static void bix_build(void) {
  if (nBIX >= 0) return;
  long n = 0;
  for (long i = 0; i < nR; i++) if (R[i].op == OP_BYTES) n++;
  BIX = realloc(BIX, sizeof(long) * (size_t)(n ? n : 1));
  n = 0;
  for (long i = 0; i < nR; i++) if (R[i].op == OP_BYTES) BIX[n++] = i;
  nBIX = n;
}
long rec_bytes_n(void) { bix_build(); return nBIX; }
// meta: 4 longs per setBytes op [payload length, bind index, ordinal of the first dispatch after it (-1 if none),
// offset of its payload in dst]; dst: every payload, concatenated in record order. Returns the op count, or -1 when
// nmax or cap is too small.
long rec_bytes_meta(long *meta, long nmax, char *dst, long cap) {
  long k = 0, off = 0, disp = 0, pend = 0;
  for (long i = 0; i < nR; i++) {
    rop *r = &R[i];
    if (r->op == OP_BYTES) {
      if (k >= nmax || off + r->nb > cap) return -1;
      meta[4 * k] = r->nb; meta[4 * k + 1] = r->i0; meta[4 * k + 2] = -1; meta[4 * k + 3] = off;
      memcpy(dst + off, r->bytes, (size_t)r->nb);
      off += r->nb; k++;
    } else if (r->op == OP_DTG || r->op == OP_DT) {
      for (long j = pend; j < k; j++) meta[4 * j + 2] = disp;
      pend = k; disp++;
    }
  }
  return k;
}
// The kernel (function) name of every dispatch, in order, one per line, into dst. Returns bytes written, or -1.
long rec_disp_names(char *dst, long cap) {
  long off = 0;
  void *pso = 0;
  for (long i = 0; i < nR; i++) {
    rop *r = &R[i];
    if (r->op == OP_PSO) pso = r->o0;
    if (r->op != OP_DTG && r->op != OP_DT) continue;
    const char *nm = "?";
    NSArray *e = nil;
    if (pso) { @synchronized(PSOFN) { e = PSOFN[[NSValue valueWithPointer:pso]]; } }
    if (e) nm = [[(id<MTLFunction>)e[0] name] UTF8String];
    long l = (long)strlen(nm);
    if (off + l + 1 > cap) return -1;
    memcpy(dst + off, nm, (size_t)l); off += l; dst[off++] = '\n';
  }
  return off;
}
// Write n bytes at byte offset off of the k-th setBytes payload. Returns 0, or -1 when out of range.
int rec_bytes_patch(long k, long off, const void *src, long n) {
  bix_build();
  if (k < 0 || k >= nBIX) return -1;
  rop *r = &R[BIX[k]];
  if (off < 0 || n < 0 || off + n > r->nb) return -1;
  memcpy((char *)r->bytes + off, src, (size_t)n);
  if (CONSTB && r->n1 > 0) memcpy((char *)CONSTB.contents + (r->n1 - 1) + off, src, (size_t)n);
  return 0;
}
// Register a rule (see prule). Returns the rule count, or -1 when the target is out of range.
long rec_rule_add2(int type, long k, long off, int var, long a, long b, long d, long m, long s, long c) {
  bix_build();
  if (type >= 0 && type <= 2) {
    long w = type == 2 ? 8 : 4;
    if (k < 0 || k >= nBIX || off < 0 || off + w > R[BIX[k]].nb) return -1;
  } else if (type == 3) {
    if (k < 0 || k >= nR || R[k].op != OP_BUF) return -1;
  } else if (type == 4 || type == 5) {
    if (k < 0 || k >= nR || R[k].op != OP_BUF || !R[k].o0) return -1;
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)R[k].o0;
    if (off < 0 || off + (type == 5 ? 8 : 4) > (long)b.length || b.storageMode == MTLStorageModePrivate) return -1;
  } else return -1;
  if (var < 0 || var > 15) return -1;
  if (nRU == capRU) { capRU = capRU ? capRU * 2 : 64; RU = realloc(RU, sizeof(prule) * (size_t)capRU); }
  RU[nRU++] = (prule){k, off, a, b, d, m, s, c, type, var};
  return nRU;
}
long rec_rule_add(long k, long off, int type, long a, long b, long m, long s, long c) {
  return rec_rule_add2(type, k, off, 0, a, b, 0, m, s, c);
}
void rec_rules_clear(void) { nRU = 0; }
static inline long fdiv(long x, long d) { long q = x / d; return (x % d != 0 && ((x < 0) != (d < 0))) ? q - 1 : q; }
// Apply every rule with these per-check values (vars[0..nv-1]). Returns the rule count, or -1 on a bad var.
long rec_rules_apply_v(const long *vars, long nv) {
  char *cb = CONSTB ? (char *)CONSTB.contents : 0;
  for (long j = 0; j < nRU; j++) {
    prule *u = &RU[j];
    if (u->var >= nv) return -1;
    long v = vars[u->var] * u->a + u->b;
    if (u->d > 0) v = fdiv(v, u->d);
    if (u->m > 0) { v %= u->m; if (v < 0) v += u->m; }
    v = v * u->s + u->c;
    if (u->type == 4 || u->type == 5) {
      char *p = (char *)((__bridge id<MTLBuffer>)R[u->k].o0).contents + u->off;
      if (u->type == 4) { int32_t x = (int32_t)v; memcpy(p, &x, 4); } else { int64_t x = (int64_t)v; memcpy(p, &x, 8); }
      continue;
    }
    if (u->type == 3) {
      rop *r = &R[u->k];
      r->n0 = v;
      if (ICB) {
        @autoreleasepool {
          for (long q = 0; q < r->nb; q++) {
            int32_t ci = ((int32_t *)r->bytes)[2 * q], kk = ((int32_t *)r->bytes)[2 * q + 1];
            [[ICB indirectComputeCommandAtIndex:(NSUInteger)ci] setKernelBuffer:(__bridge id<MTLBuffer>)r->o0
                                                                         offset:(NSUInteger)v atIndex:(NSUInteger)kk];
          }
        }
      }
      continue;
    }
    rop *r = &R[BIX[u->k]];
    char *p = (char *)r->bytes + u->off;
    char *q = (cb && r->n1 > 0) ? cb + (r->n1 - 1) + u->off : 0;
    if (u->type == 0) { int32_t x = (int32_t)v; memcpy(p, &x, 4); if (q) memcpy(q, &x, 4); }
    else if (u->type == 1) { float x = (float)v; memcpy(p, &x, 4); if (q) memcpy(q, &x, 4); }
    else { int64_t x = (int64_t)v; memcpy(p, &x, 8); if (q) memcpy(q, &x, 8); }
  }
  return nRU;
}
long rec_rules_apply(long seq) { return rec_rules_apply_v(&seq, 1); }
// One row of 13 longs per op: [op, i0, n0, n1, grid w h d, threadgroup w h d, o0, o1, nb] (o0 / o1 = object pointers;
// for BUF ops nb is the ICB use count once the ICB is built, for BYTES n1 is its constants slot + 1). Returns the op
// count, or -1 when nmax is too small.
long rec_ops_dump(long *dst, long nmax) {
  if (nmax < nR) return -1;
  for (long i = 0; i < nR; i++) {
    rop *r = &R[i];
    long *o = dst + 13 * i;
    o[0] = r->op; o[1] = r->i0; o[2] = r->n0; o[3] = r->n1;
    o[4] = (long)r->a.width; o[5] = (long)r->a.height; o[6] = (long)r->a.depth;
    o[7] = (long)r->b.width; o[8] = (long)r->b.height; o[9] = (long)r->b.depth;
    o[10] = (long)r->o0; o[11] = (long)r->o1; o[12] = r->nb;
  }
  return nR;
}

// ---- segmented ICB replay (milestone 2): one command buffer per recorded command buffer, so GPU writes reach the CPU at
// the same points as in the recorded run (e.g. the engine's routing row after each post, for its loader thread). All
// segments are committed back to back; the CPU waits on the last. use_rs: 1 = rely on the queue's residency set.
// out (doubles): [0] CPU ns to encode + commit all, [1] ns from the last commit to completion, [2] GPU ns (first segment's
// start -> last segment's end), [3] status of the last, [4] segments, [5] commands. Returns 0 on success.
long rec_icb_nseg(void) { return nSEGE; }
int rec_replay_icb_seg(int use_rs, double *out) {
  @autoreleasepool {
    if (!ICB || !ICBN || !nSEGE) return -9;
    double t0 = ns_now();
    id<MTLCommandBuffer> first = nil, last = nil;
    long a = 0;
    depth++;
    for (long g = 0; g < nSEGE; g++) {
      id<MTLCommandBuffer> cb = [RQ commandBufferWithUnretainedReferences];
      id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      if (!use_rs) [ce useResources:RESARR count:RESN usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
      [ce executeCommandsInBuffer:ICB withRange:NSMakeRange((NSUInteger)a, (NSUInteger)(SEGE[g] - a))];
      [ce endEncoding];
      [cb commit];
      if (!first) first = cb;
      last = cb;
      a = SEGE[g];
    }
    double t1 = ns_now();
    [last waitUntilCompleted];
    depth--;
    double t2 = ns_now();
    out[0] = t1 - t0; out[1] = t2 - t1; out[2] = (last.GPUEndTime - first.GPUStartTime) * 1e9;
    out[3] = (double)last.status; out[4] = (double)nSEGE; out[5] = (double)ICBN;
    return last.status == MTLCommandBufferStatusCompleted ? 0 : -1;
  }
}

// ---- a toy loader thread (test only, milestone 2): it polls a routing row the GPU writes ([seq, value]) and, on a new
// seq, writes the release [seq, value * mul + add] (the slot first, then the tag with release order), like the engine's
// loader answering post_ids. Plain C on its own pthread, so the GIL never blocks it.
#include <pthread.h>
static pthread_t TL;
static _Atomic int tl_stop = 0, tl_on = 0;
static volatile int *TL_ROW = 0, *TL_REL = 0;
static long TL_MUL = 3, TL_ADD = 1;
static _Atomic long tl_n = 0;
static void *tl_main(void *arg) {
  (void)arg;
  int last = 0;
  while (!atomic_load_explicit(&tl_stop, memory_order_relaxed)) {
    int s = __atomic_load_n(&TL_ROW[0], __ATOMIC_ACQUIRE);
    if (s != last && s > 0) {
      int v = __atomic_load_n(&TL_ROW[1], __ATOMIC_ACQUIRE);
      __atomic_store_n(&TL_REL[1], (int)(v * TL_MUL + TL_ADD), __ATOMIC_RELAXED);
      __atomic_store_n(&TL_REL[0], s, __ATOMIC_RELEASE);
      last = s;
      atomic_fetch_add_explicit(&tl_n, 1, memory_order_relaxed);
    }
  }
  return 0;
}
int rec_toyloader_start(void *row, void *rel, long mul, long add) {
  if (atomic_load(&tl_on)) return -1;
  TL_ROW = (volatile int *)row; TL_REL = (volatile int *)rel; TL_MUL = mul; TL_ADD = add;
  atomic_store(&tl_stop, 0); atomic_store(&tl_n, 0);
  if (pthread_create(&TL, NULL, tl_main, NULL) != 0) return -2;
  atomic_store(&tl_on, 1);
  return 0;
}
long rec_toyloader_stop(void) {
  if (!atomic_load(&tl_on)) return -1;
  atomic_store(&tl_stop, 1);
  pthread_join(TL, NULL);
  atomic_store(&tl_on, 0);
  return atomic_load(&tl_n);
}
long rec_toyloader_count(void) { return atomic_load(&tl_n); }

// ---- buffer contents of the record's bindings (to find per-call input arrays by a two-record diff)
long rec_buf_len(long op) {
  if (op < 0 || op >= nR || R[op].op != OP_BUF || !R[op].o0) return -1;
  id<MTLBuffer> b = (__bridge id<MTLBuffer>)R[op].o0;
  return b.storageMode == MTLStorageModePrivate ? -2 : (long)b.length;
}
long rec_buf_read(long op, long off, void *dst, long n) {
  long L = rec_buf_len(op);
  if (L < 0 || off < 0 || n < 0 || off + n > L) return -1;
  memcpy(dst, (char *)((__bridge id<MTLBuffer>)R[op].o0).contents + off, (size_t)n);
  return n;
}
long rec_buf_write(long op, long off, const void *src, long n) {
  long L = rec_buf_len(op);
  if (L < 0 || off < 0 || n < 0 || off + n > L) return -1;
  memcpy((char *)((__bridge id<MTLBuffer>)R[op].o0).contents + off, src, (size_t)n);
  return n;
}

// The bind-time snapshot of the buffer bound by BUF op `op` (the snapshot is keyed by that buffer's FIRST binding in the
// record; any op binding the same buffer finds it). Returns its length (copying up to cap bytes into dst), or -1.
long rec_buf_snap(long op, void *dst, long cap) {
  if (op < 0 || op >= nR || R[op].op != OP_BUF) return -1;
  void *key = R[op].o0;
  for (long i = 0; i < nSN; i++)
    if (SNBUF[i] == key) {
      long n = SNLEN[i] < cap ? SNLEN[i] : cap;
      memcpy(dst, SNDAT[i], (size_t)n);
      return SNLEN[i];
    }
  return -1;
}
long rec_snap_count(void) { return nSN; }
