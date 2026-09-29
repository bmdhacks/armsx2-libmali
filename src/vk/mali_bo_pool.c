/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

#include "mali_bo_pool.h"

#include <stdlib.h>
#include <string.h>

#include "util/log.h"
#include "util/macros.h"
#include "util/u_math.h"

/* 16 MiB: kbase refuses a GPU_EX allocation larger than the program
 * counter range, 2^24 bytes on the G615. */
#define MALI_EXEC_MAX_ALLOC (16ull << 20)
/* Slab sizes. ARMSX2 builds a few hundred pipelines of a few KiB each. */
#define MALI_EXEC_SLAB_SIZE (1ull << 20)
#define MALI_DESC_SLAB_SIZE (256ull << 10)

void
mali_bo_pool_init(struct mali_bo_pool *pool, struct mali_kbase *kb,
                  const char *name, uint64_t flags,
                  enum mali_kbase_mem_class mem_class, uint64_t slab_size,
                  uint64_t max_alloc, bool within_4g)
{
   memset(pool, 0, sizeof(*pool));
   pool->kb = kb;
   pool->name = name;
   pool->flags = flags;
   pool->mem_class = mem_class;
   pool->slab_size = slab_size;
   pool->max_alloc = max_alloc;
   pool->within_4g = within_4g;
   simple_mtx_init(&pool->lock, mtx_plain);
   list_inithead(&pool->slabs);
}

void
mali_bo_pool_init_exec(struct mali_bo_pool *pool, struct mali_kbase *kb)
{
   /* No SAME_VA: GPU_EX memory must land in the executable zone. */
   mali_bo_pool_init(pool, kb, "exec", MALI_KBASE_FLAGS_PROGRAM,
                     MALI_KBASE_MEM_CLASS_PROGRAM, MALI_EXEC_SLAB_SIZE,
                     MALI_EXEC_MAX_ALLOC, true);
}

void
mali_bo_pool_init_desc(struct mali_bo_pool *pool, struct mali_kbase *kb)
{
   /* The blob's pipeline pool: GPU read/write, CPU-uncached, SAME_VA,
    * class 45. */
   mali_bo_pool_init(pool, kb, "pipeline descriptors",
                     mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
                     MALI_KBASE_MEM_CLASS_INTERNAL, MALI_DESC_SLAB_SIZE,
                     MALI_DESC_SLAB_SIZE, false);
}

static void
slab_destroy(struct mali_bo_pool *pool, struct mali_bo_pool_slab *slab)
{
   list_del(&slab->link);
   util_vma_heap_finish(&slab->heap);
   mali_kbase_free(pool->kb, &slab->bo);
   pool->slab_count--;
   free(slab);
}

void
mali_bo_pool_finish(struct mali_bo_pool *pool)
{
   list_for_each_entry_safe(struct mali_bo_pool_slab, slab, &pool->slabs, link) {
      if (slab->used)
         mesa_loge("%s pool: %llu bytes still allocated at destruction",
                   pool->name, (unsigned long long)slab->used);
      slab_destroy(pool, slab);
   }
   simple_mtx_destroy(&pool->lock);
}

static enum mali_kbase_result
slab_create(struct mali_bo_pool *pool, uint64_t min_size,
            struct mali_bo_pool_slab **out)
{
   struct mali_bo_pool_slab *slab = calloc(1, sizeof(*slab));
   if (!slab)
      return MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY;

   const struct mali_kbase_alloc_info info = {
      .size = MAX2(pool->slab_size, align64(min_size, 4096)),
      .flags = pool->flags,
      .mem_class = pool->mem_class,
      .cpu_map = true,
   };
   enum mali_kbase_result r = mali_kbase_alloc(pool->kb, &info, &slab->bo);
   if (r != MALI_KBASE_SUCCESS) {
      free(slab);
      return r;
   }

   if (pool->within_4g &&
       (slab->bo.gpu_va >> 32) != ((slab->bo.gpu_va + slab->bo.size - 1) >> 32)) {
      /* The executable zone is one aligned 4 GiB window, so the kernel
       * never does this; refuse rather than hand out code that cannot
       * reach its constants. */
      mesa_loge("%s pool: slab at 0x%llx crosses a 4 GiB boundary", pool->name,
                (unsigned long long)slab->bo.gpu_va);
      mali_kbase_free(pool->kb, &slab->bo);
      free(slab);
      return MALI_KBASE_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   util_vma_heap_init(&slab->heap, slab->bo.gpu_va, slab->bo.size);
   list_addtail(&slab->link, &pool->slabs);
   pool->slab_count++;
   *out = slab;
   return MALI_KBASE_SUCCESS;
}

enum mali_kbase_result
mali_bo_pool_alloc(struct mali_bo_pool *pool, uint64_t size, uint64_t align,
                   struct mali_bo_ref *out)
{
   memset(out, 0, sizeof(*out));
   if (size == 0 || size > pool->max_alloc || !util_is_power_of_two_nonzero64(align))
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;

   simple_mtx_lock(&pool->lock);

   uint64_t va = 0;
   struct mali_bo_pool_slab *found = NULL;
   list_for_each_entry(struct mali_bo_pool_slab, slab, &pool->slabs, link) {
      va = util_vma_heap_alloc(&slab->heap, size, align);
      if (va) {
         found = slab;
         break;
      }
   }

   if (!found) {
      enum mali_kbase_result r = slab_create(pool, size, &found);
      if (r != MALI_KBASE_SUCCESS) {
         simple_mtx_unlock(&pool->lock);
         return r;
      }
      va = util_vma_heap_alloc(&found->heap, size, align);
      assert(va);
   }

   found->used += size;
   simple_mtx_unlock(&pool->lock);

   out->slab = found;
   out->gpu_va = va;
   out->cpu = (uint8_t *)found->bo.cpu + (va - found->bo.gpu_va);
   out->size = size;
   return MALI_KBASE_SUCCESS;
}

void
mali_bo_pool_free(struct mali_bo_pool *pool, struct mali_bo_ref *ref)
{
   struct mali_bo_pool_slab *slab = ref->slab;
   if (!slab)
      return;

   simple_mtx_lock(&pool->lock);
   util_vma_heap_free(&slab->heap, ref->gpu_va, ref->size);
   slab->used -= ref->size;
   /* Keep the last slab around so a pipeline created and destroyed in a
    * loop does not allocate a kernel region each time. */
   if (slab->used == 0 && pool->slab_count > 1)
      slab_destroy(pool, slab);
   simple_mtx_unlock(&pool->lock);

   memset(ref, 0, sizeof(*ref));
}
