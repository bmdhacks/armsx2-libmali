/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * A pool of driver-internal GPU memory: kbase allocations ("slabs") with one
 * set of flags, sub-allocated with an address-range allocator. The device
 * has two:
 *
 *  - the executable pool: shader code, MALI_KBASE_FLAGS_PROGRAM, class
 *    PROGRAM. kbase puts it in the 4 GiB executable zone; one slab is at
 *    most 16 MiB, the largest GPU_EX allocation the G615 kernel accepts;
 *  - the pipeline descriptor pool: the 32-byte Shader Program Descriptors
 *    and other small read-only descriptors pipelines own, CPU-uncached,
 *    SAME_VA, class INTERNAL.
 *
 * Both mappings are CPU-uncached, so a CPU write needs no cache
 * maintenance before the GPU reads it. Thread-safe.
 */

#ifndef MALI_BO_POOL_H
#define MALI_BO_POOL_H

#include <stdint.h>

#include "util/list.h"
#include "util/simple_mtx.h"
#include "util/vma.h"

#include "kbase/kbase.h"

struct mali_bo_pool_slab {
   struct list_head link;
   struct mali_kbase_bo bo;
   struct util_vma_heap heap;   /* over [bo.gpu_va, bo.gpu_va + bo.size) */
   uint64_t used;               /* bytes handed out */
};

struct mali_bo_pool {
   struct mali_kbase *kb;
   const char *name;
   uint64_t flags;              /* kbase flags, SAME_VA already decided */
   enum mali_kbase_mem_class mem_class;
   uint64_t slab_size;          /* size of an ordinary slab */
   uint64_t max_alloc;          /* largest single sub-allocation */
   /* Every sub-allocation must stay inside one 4 GiB-aligned window (shader
    * code: PC-relative constants). */
   bool within_4g;

   simple_mtx_t lock;
   struct list_head slabs;
   unsigned slab_count;
};

/* One sub-allocation. */
struct mali_bo_ref {
   struct mali_bo_pool_slab *slab;
   uint64_t gpu_va;
   void *cpu;
   uint64_t size;
};

void mali_bo_pool_init(struct mali_bo_pool *pool, struct mali_kbase *kb,
                       const char *name, uint64_t flags,
                       enum mali_kbase_mem_class mem_class, uint64_t slab_size,
                       uint64_t max_alloc, bool within_4g);

/* The executable pool and the pipeline descriptor pool described above. */
void mali_bo_pool_init_exec(struct mali_bo_pool *pool, struct mali_kbase *kb);
void mali_bo_pool_init_desc(struct mali_bo_pool *pool, struct mali_kbase *kb);

/* Frees every slab. Every sub-allocation must have been freed. */
void mali_bo_pool_finish(struct mali_bo_pool *pool);

/*
 * size bytes at a multiple of align (a power of two). The memory is CPU
 * mapped and uncached. MALI_KBASE_ERROR_INVALID_ARGUMENT if size exceeds
 * max_alloc; otherwise the kbase allocation's result.
 */
enum mali_kbase_result mali_bo_pool_alloc(struct mali_bo_pool *pool,
                                          uint64_t size, uint64_t align,
                                          struct mali_bo_ref *out);

/* Returns the range; an empty slab other than the last one is freed.
 * Clears *ref. A zeroed ref is ignored. */
void mali_bo_pool_free(struct mali_bo_pool *pool, struct mali_bo_ref *ref);

#endif
