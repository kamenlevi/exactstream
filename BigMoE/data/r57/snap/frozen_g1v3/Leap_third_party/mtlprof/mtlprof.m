// mtlprof.m (Breakthrough, Sun 15 Nov 2026; CHECK55 step 0): in-process Metal counters for an MLX process, no xctrace.
// Loaded with ctypes BEFORE `import mlx.core`. It swizzles the concrete Metal classes MLX uses (classic Metal: one compute
// encoder per command buffer segment, MLX 0.32.2 device.h) and records, at ~tens of ns per call:
//   - every compute dispatch (dispatchThreadgroups / dispatchThreads), per pipeline (its function name, from the
//     pipeline-creation hook) and per tag (an int Python sets around code regions);
//   - memory barriers, fence waits / updates, compute encoders, blit encoders;
//   - every committed command buffer: host time at commit, the tag at commit, its dispatch count, and the GPU's own
//     GPUStartTime / GPUEndTime (same time base as CACurrentMediaTime) from a completed handler.
// Python reads it back through the prof_* functions (ctypes). Build: see build_mtlprof.sh.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import <objc/runtime.h>
#include <stdatomic.h>
#include <string.h>

#define MAXCB (1 << 20)
#define MAXTAG 64
#define HSIZE 8192

static double cb_commit[MAXCB], cb_start[MAXCB], cb_end[MAXCB], cb_done[MAXCB];
static int cb_tag[MAXCB], cb_ndisp[MAXCB];
static _Atomic int cb_ready[MAXCB];
static _Atomic long n_cb = 0;                         // command buffers committed (slots handed out)
static _Atomic long c_disp = 0, c_bar = 0, c_fwait = 0, c_fupd = 0, c_enc = 0, c_blit = 0, c_commit = 0;
static _Atomic long c_disp_tag[MAXTAG];
static _Atomic long c_tag_pso[MAXTAG][HSIZE];          // dispatches per (tag, pipeline slot)
static _Atomic int cur_tag = 0;
static _Atomic int on = 1;
static long disp_at_last_commit = 0;
static __thread int depth = 0;                         // guards nested (super) calls through hooked methods
static __thread void *cur_pso = 0;

typedef struct { void *key; _Atomic long n; char name[120]; } hent;
static hent H[HSIZE];
static long hslot(hent *e) { return e ? (long)(e - H) : -1; }

static hent *hget(void *k, int create) {
  unsigned long h = ((unsigned long)k >> 4) * 11400714819323198485ull;
  for (int i = 0; i < HSIZE; i++) {
    hent *e = &H[(h + i) & (HSIZE - 1)];
    if (e->key == k) return e;
    if (e->key == 0) {
      if (!create) return 0;
      e->key = k;
      return e;
    }
  }
  return 0;
}

// ---- originals
typedef void (*disp_t)(id, SEL, MTLSize, MTLSize);
typedef void (*v1_t)(id, SEL, NSUInteger);
typedef void (*vres_t)(id, SEL, const id *, NSUInteger);
typedef void (*vobj_t)(id, SEL, id);
typedef id (*ret1_t)(id, SEL, NSUInteger);
typedef id (*ret0_t)(id, SEL);
typedef void (*commit_t)(id, SEL);
typedef id (*psod_t)(id, SEL, id, NSUInteger, id *, NSError **);
typedef id (*psof_t)(id, SEL, id, NSError **);
static disp_t o_dtg = 0, o_dt = 0;
static v1_t o_bar = 0;
static vres_t o_barr = 0;
static vobj_t o_setpso = 0, o_fwait = 0, o_fupd = 0;
static ret1_t o_enc = 0;
static ret0_t o_enc0 = 0, o_blit = 0;
static commit_t o_commit = 0;
static psod_t o_psod = 0;
static psof_t o_psof = 0;

static inline void count_disp(void) {
  if (!atomic_load_explicit(&on, memory_order_relaxed)) return;
  atomic_fetch_add_explicit(&c_disp, 1, memory_order_relaxed);
  int t = atomic_load_explicit(&cur_tag, memory_order_relaxed);
  if (t >= 0 && t < MAXTAG) atomic_fetch_add_explicit(&c_disp_tag[t], 1, memory_order_relaxed);
  if (cur_pso) {
    hent *e = hget(cur_pso, 1);
    if (e) {
      atomic_fetch_add_explicit(&e->n, 1, memory_order_relaxed);
      if (t >= 0 && t < MAXTAG) atomic_fetch_add_explicit(&c_tag_pso[t][hslot(e)], 1, memory_order_relaxed);
    }
  }
}

static void h_dtg(id s, SEL c, MTLSize a, MTLSize b) {
  if (depth++ == 0) count_disp();
  o_dtg(s, c, a, b);
  depth--;
}
static void h_dt(id s, SEL c, MTLSize a, MTLSize b) {
  if (depth++ == 0) count_disp();
  o_dt(s, c, a, b);
  depth--;
}
static void h_bar(id s, SEL c, NSUInteger scope) {
  if (depth++ == 0 && on) atomic_fetch_add_explicit(&c_bar, 1, memory_order_relaxed);
  o_bar(s, c, scope);
  depth--;
}
static void h_barr(id s, SEL c, const id *r, NSUInteger n) {
  if (depth++ == 0 && on) atomic_fetch_add_explicit(&c_bar, 1, memory_order_relaxed);
  o_barr(s, c, r, n);
  depth--;
}
static void h_setpso(id s, SEL c, id pso) {
  if (depth++ == 0) cur_pso = (__bridge void *)pso;
  o_setpso(s, c, pso);
  depth--;
}
static void h_fwait(id s, SEL c, id f) {
  if (depth++ == 0 && on) atomic_fetch_add_explicit(&c_fwait, 1, memory_order_relaxed);
  o_fwait(s, c, f);
  depth--;
}
static void h_fupd(id s, SEL c, id f) {
  if (depth++ == 0 && on) atomic_fetch_add_explicit(&c_fupd, 1, memory_order_relaxed);
  o_fupd(s, c, f);
  depth--;
}
static id h_enc(id s, SEL c, NSUInteger t) {
  if (depth++ == 0 && on) atomic_fetch_add_explicit(&c_enc, 1, memory_order_relaxed);
  id r = o_enc(s, c, t);
  depth--;
  return r;
}
static id h_enc0(id s, SEL c) {
  if (depth++ == 0 && on) atomic_fetch_add_explicit(&c_enc, 1, memory_order_relaxed);
  id r = o_enc0(s, c);
  depth--;
  return r;
}
static id h_blit(id s, SEL c) {
  if (depth++ == 0 && on) atomic_fetch_add_explicit(&c_blit, 1, memory_order_relaxed);
  id r = o_blit(s, c);
  depth--;
  return r;
}
static void h_commit(id s, SEL c) {
  if (depth++ == 0 && on) {
    long d = atomic_load(&c_disp);
    long i = atomic_fetch_add(&n_cb, 1);
    atomic_fetch_add(&c_commit, 1);
    if (i < MAXCB) {
      cb_commit[i] = CACurrentMediaTime();
      cb_tag[i] = atomic_load(&cur_tag);
      cb_ndisp[i] = (int)(d - disp_at_last_commit);
      atomic_store(&cb_ready[i], 0);
      [(id<MTLCommandBuffer>)s addCompletedHandler:^(id<MTLCommandBuffer> b) {
        cb_start[i] = b.GPUStartTime;
        cb_end[i] = b.GPUEndTime;
        cb_done[i] = CACurrentMediaTime();
        atomic_store(&cb_ready[i], 1);
      }];
    }
    disp_at_last_commit = d;
  }
  o_commit(s, c);
  depth--;
}
static void name_pso(id pso, NSString *nm) {
  if (!pso || !nm) return;
  hent *e = hget((__bridge void *)pso, 1);
  if (e) strlcpy(e->name, [nm UTF8String], sizeof(e->name));
}
static id h_psod(id s, SEL c, id desc, NSUInteger opt, id *refl, NSError **err) {
  id r = o_psod(s, c, desc, opt, refl, err);
  @try {
    id<MTLFunction> f = [(MTLComputePipelineDescriptor *)desc computeFunction];
    name_pso(r, f ? f.name : nil);
  } @catch (id ex) {}
  return r;
}
static id h_psof(id s, SEL c, id fn, NSError **err) {
  id r = o_psof(s, c, fn, err);
  @try { name_pso(r, [(id<MTLFunction>)fn name]); } @catch (id ex) {}
  return r;
}

static int hooked = 0;
static IMP hook(Class cls, SEL sel, IMP imp, const char *what) {
  Method m = class_getInstanceMethod(cls, sel);
  if (!m) {
    fprintf(stderr, "mtlprof: %s: %s not found on %s\n", what, sel_getName(sel), class_getName(cls));
    return 0;
  }
  hooked++;
  return method_setImplementation(m, imp);
}

__attribute__((constructor)) static void mtlprof_init(void) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    Class ce = object_getClass(enc), cbc = object_getClass(cb), dc = object_getClass(dev);
    o_dtg = (disp_t)hook(ce, @selector(dispatchThreadgroups:threadsPerThreadgroup:), (IMP)h_dtg, "dtg");
    o_dt = (disp_t)hook(ce, @selector(dispatchThreads:threadsPerThreadgroup:), (IMP)h_dt, "dt");
    o_bar = (v1_t)hook(ce, @selector(memoryBarrierWithScope:), (IMP)h_bar, "bar");
    o_barr = (vres_t)hook(ce, @selector(memoryBarrierWithResources:count:), (IMP)h_barr, "barr");
    o_setpso = (vobj_t)hook(ce, @selector(setComputePipelineState:), (IMP)h_setpso, "pso");
    o_fwait = (vobj_t)hook(ce, @selector(waitForFence:), (IMP)h_fwait, "fwait");
    o_fupd = (vobj_t)hook(ce, @selector(updateFence:), (IMP)h_fupd, "fupd");
    [enc endEncoding];
    o_enc = (ret1_t)hook(cbc, @selector(computeCommandEncoderWithDispatchType:), (IMP)h_enc, "enc");
    o_enc0 = (ret0_t)hook(cbc, @selector(computeCommandEncoder), (IMP)h_enc0, "enc0");
    o_blit = (ret0_t)hook(cbc, @selector(blitCommandEncoder), (IMP)h_blit, "blit");
    o_commit = (commit_t)hook(cbc, @selector(commit), (IMP)h_commit, "commit");
    o_psod = (psod_t)hook(dc, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), (IMP)h_psod, "psod");
    o_psof = (psof_t)hook(dc, @selector(newComputePipelineStateWithFunction:error:), (IMP)h_psof, "psof");
    fprintf(stderr, "mtlprof: %d hooks on %s / %s / %s\n", hooked, class_getName(ce), class_getName(cbc), class_getName(dc));
  }
}

// ---- Python API (ctypes)
double prof_now(void) { return CACurrentMediaTime(); }
void prof_on(int v) { atomic_store(&on, v); }
void prof_tag(int t) { atomic_store(&cur_tag, t); }
int prof_get_tag(void) { return atomic_load(&cur_tag); }
// counters: disp, bar, fwait, fupd, enc, blit, commit, n_cb
void prof_counts(long *o) {
  o[0] = c_disp; o[1] = c_bar; o[2] = c_fwait; o[3] = c_fupd; o[4] = c_enc; o[5] = c_blit; o[6] = c_commit; o[7] = n_cb;
}
void prof_tag_counts(long *o) {
  for (int i = 0; i < MAXTAG; i++) o[i] = c_disp_tag[i];
}
// dispatches per pipeline slot for tag t (HSIZE longs)
void prof_tag_pso(int t, long *o) {
  for (int i = 0; i < HSIZE; i++) o[i] = (t >= 0 && t < MAXTAG) ? c_tag_pso[t][i] : 0;
}
long prof_hsize(void) { return HSIZE; }
// command buffers [i0, i1): commit, start, end, done, tag, ndisp, ready. Returns how many were written.
long prof_cbs(long i0, long i1, double *commit, double *start, double *end, double *done, int *tag, int *nd, int *ready) {
  long n = atomic_load(&n_cb);
  if (i1 > n) i1 = n;
  if (i1 > MAXCB) i1 = MAXCB;
  long k = 0;
  for (long i = i0; i < i1; i++, k++) {
    ready[k] = atomic_load(&cb_ready[i]);
    commit[k] = cb_commit[i]; start[k] = cb_start[i]; end[k] = cb_end[i]; done[k] = cb_done[i];
    tag[k] = cb_tag[i]; nd[k] = cb_ndisp[i];
  }
  return k;
}
// pipelines: every used slot (up to max): its slot index, dispatch count and name (max x 120 chars); returns the count.
long prof_psos(long max, long *slots, long *counts, char *names) {
  long k = 0;
  for (int i = 0; i < HSIZE && k < max; i++) {
    if (!H[i].key) continue;
    slots[k] = i;
    counts[k] = H[i].n;
    strlcpy(names + 120 * k, H[i].name[0] ? H[i].name : "?", 120);
    k++;
  }
  return k;
}
