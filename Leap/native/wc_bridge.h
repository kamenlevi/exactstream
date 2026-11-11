// wc_bridge.h - a volatile load for Swift's spin on a GPU-written flag (WHOLE-CHIP-2, wc_handoff.swift)
#include <stdint.h>
static inline uint32_t wc_vload(const void* p) { return *(const volatile uint32_t*)p; }
static inline void wc_vstore(void* p, uint32_t v) { __atomic_store_n((uint32_t*)p, v, __ATOMIC_RELEASE); }
