/*
 * Copyright 2026 The libmali authors
 * SPDX-License-Identifier: MIT
 */

/*
 * User-space CPU cache maintenance for GPU memory that is CPU-cached but
 * not system-coherent. COHERENT_LOCAL only makes GPU accesses snoop the
 * CPU's point of coherency; a dirty line in a CPU cache still has to be
 * cleaned to it. Linux lets EL0 run DC CVAC and DC CIVAC (SCTLR_EL1.UCI),
 * which is what the blob uses. There is no user-space invalidate-only
 * instruction, so "invalidate" is clean-and-invalidate, as in the blob.
 */

#include "kbase_priv.h"

#if defined(__aarch64__)

static size_t
dcache_line_size(void)
{
   uint64_t ctr;
   __asm__ volatile("mrs %0, ctr_el0" : "=r"(ctr));
   /* DminLine, bits 19:16: log2 of the smallest data cache line in words. */
   return (size_t)4 << ((ctr >> 16) & 0xf);
}

#define DC_RANGE(insn, addr, size)                                       \
   do {                                                                  \
      size_t line_ = dcache_line_size();                                 \
      uintptr_t p_ = (uintptr_t)(addr) & ~(uintptr_t)(line_ - 1);        \
      uintptr_t end_ = (uintptr_t)(addr) + (size);                       \
      for (; p_ < end_; p_ += line_)                                     \
         __asm__ volatile("dc " insn ", %0" ::"r"(p_) : "memory");       \
      __asm__ volatile("dsb sy" ::: "memory");                           \
   } while (0)

void
mali_kbase_cpu_clean(const void *addr, size_t size)
{
   if (size)
      DC_RANGE("cvac", addr, size);
}

void
mali_kbase_cpu_clean_invalidate(const void *addr, size_t size)
{
   if (size)
      DC_RANGE("civac", addr, size);
}

#else

/* Only aarch64 has a kbase GPU we drive; keep other hosts building. */
void
mali_kbase_cpu_clean(const void *addr, size_t size)
{
   (void)addr;
   (void)size;
   __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

void
mali_kbase_cpu_clean_invalidate(const void *addr, size_t size)
{
   (void)addr;
   (void)size;
   __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

#endif
