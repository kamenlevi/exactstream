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
  if (rec()) { rop *r = push(OP_BUF); r->o0 = keep(b); r->n0 = (long)off; r->i0 = (long)idx; }
  depth++; o_buf(s, c, b, off, idx); depth--;
}
static void h_bufs(id s, SEL c, const id *bs, const NSUInteger *offs, NSRange rg) {
  if (rec())
    for (NSUInteger i = 0; i < rg.length; i++) {
      rop *r = push(OP_BUF); r->o0 = keep(bs[i]); r->n0 = (long)offs[i]; r->i0 = (long)(rg.location + i);
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
}
void rec_start(void) { free_ops(); atomic_store(&recording, 1); }
void rec_stop(void) { atomic_store(&recording, 0); }
void rec_clear(void) { atomic_store(&recording, 0); free_ops(); }
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

// Replay the record. serial: 1 = every compute encoder serial (barriers kept), 0 = the recorded dispatch type.
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
          ce = [cb computeCommandEncoderWithDispatchType:(serial ? MTLDispatchTypeSerial : (MTLDispatchType)r->i0)];
          break;
        case OP_BLIT: be = [cb blitCommandEncoder]; break;
        case OP_END:
          if (ce) { [ce endEncoding]; ce = nil; } else if (be) { [be endEncoding]; be = nil; }
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
        case OP_FWAIT: if (ce) [ce waitForFence:(__bridge id<MTLFence>)r->o0]; break;
        case OP_FUPD: if (ce) [ce updateFence:(__bridge id<MTLFence>)r->o0]; break;
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
