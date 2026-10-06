/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The v9 command buffer, as far as submit and the builders shared with
 * v11 need it: the list of closed batches and their fragment segments
 * (mali_jm.h), command-buffer memory, and the binds. Recording job chains
 * (barriers, batch closing, the draw template, render passes) is not
 * written yet; until it is, a render pass records
 * VK_ERROR_FEATURE_NOT_PRESENT.
 */

#include "mali_jm.h"

#include <stdlib.h>
#include <string.h>

#include "util/log.h"
#include "vk_alloc.h"
#include "vk_command_pool.h"
#include "vk_log.h"

/* ---------------------------------------------------------------------- */
/* Memory                                                                  */

/* A slab of command-buffer memory, CPU and GPU read/write, CPU-uncached,
 * as on v11 (the GPU writes job headers back). Not recycled through the
 * device yet: each command buffer frees its own at reset. */
static struct mali_cmd_slab *
slab_new(struct mali_device *dev, uint64_t size)
{
   struct mali_cmd_slab *s = calloc(1, sizeof(*s));
   if (!s)
      return NULL;
   const struct mali_kbase_alloc_info ai = {
      .size = size,
      .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
      .mem_class = MALI_KBASE_MEM_CLASS_COMMAND_ALLOCATOR,
   };
   if (mali_kbase_alloc(dev->kbase, &ai, &s->bo) != MALI_KBASE_SUCCESS) {
      free(s);
      return NULL;
   }
   return s;
}

static void
slabs_free(struct mali_cmd_buffer *cmd)
{
   list_for_each_entry_safe(struct mali_cmd_slab, s, &cmd->slabs, link) {
      list_del(&s->link);
      mali_kbase_free(cmd->dev->kbase, &s->bo);
      free(s);
   }
   cmd->cur = NULL;
   cmd->cur_offset = 0;
   cmd->cur_cpu = NULL;
   cmd->cur_gpu = 0;
}

struct mali_ptr
MALI_PER_ARCH(cmd_alloc_slow)(struct mali_cmd_buffer *cmd, uint64_t size, uint64_t align)
{
   assert(util_is_power_of_two_nonzero64(align) && align <= 4096);
   size = MAX2(size, 1);

   if (size > MALI_CMD_SLAB_SIZE / 2) {
      struct mali_cmd_slab *s = slab_new(cmd->dev, size);
      if (!s)
         goto oom;
      list_addtail(&s->link, &cmd->slabs);
      return (struct mali_ptr){s->bo.cpu, s->bo.gpu_va};
   }

   uint64_t off = ALIGN_POT(cmd->cur_offset, align);
   if (!cmd->cur || off + size > MALI_CMD_SLAB_SIZE) {
      struct mali_cmd_slab *s = slab_new(cmd->dev, MALI_CMD_SLAB_SIZE);
      if (!s)
         goto oom;
      s->recycle = true;
      list_addtail(&s->link, &cmd->slabs);
      cmd->cur = s;
      cmd->cur_cpu = s->bo.cpu;
      cmd->cur_gpu = s->bo.gpu_va;
      off = 0;
   }
   cmd->cur_offset = off + size;
   return (struct mali_ptr){cmd->cur_cpu + off, cmd->cur_gpu + off};

oom:
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return (struct mali_ptr){0};
}

/* ---------------------------------------------------------------------- */
/* Binds, and the render-pass hooks job-chain recording will fill in      */

void
MALI_PER_ARCH(cmd_bind_pipeline)(struct mali_cmd_buffer *cmd, struct vk_pipeline *pipeline)
{
   if (pipeline->bind_point == VK_PIPELINE_BIND_POINT_COMPUTE)
      cmd->compute.shader = mali_compute_pipeline(pipeline)->cs;
   else if (pipeline->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS)
      MALI_PER_ARCH(cmd_bind_graphics)(cmd, mali_graphics_pipeline(pipeline));
}

static void
not_recorded(struct mali_cmd_buffer *cmd, const char *what)
{
   mesa_loge("libmali: %s is not implemented on the job manager (v9) yet", what);
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
}

void
MALI_PER_ARCH(cmd_render_begin)(struct mali_cmd_buffer *cmd, const struct mali_render_desc *desc)
{
   not_recorded(cmd, "a render pass");
}

void
MALI_PER_ARCH(cmd_render_end)(struct mali_cmd_buffer *cmd)
{
}

bool
MALI_PER_ARCH(cmd_render_tiler)(struct mali_cmd_buffer *cmd)
{
   not_recorded(cmd, "tiling");
   return false;
}

/* ---------------------------------------------------------------------- */
/* Command buffer objects                                                  */

static void
cmd_clear(struct mali_cmd_buffer *cmd)
{
   util_dynarray_clear(&cmd->jm.batches);
   util_dynarray_clear(&cmd->jm.frags);
   slabs_free(cmd);
   memset(&cmd->compute, 0, sizeof(cmd->compute));
   memset(&cmd->gfx, 0, sizeof(cmd->gfx));
   memset(&cmd->tls, 0, sizeof(cmd->tls));
   cmd->dispatches = cmd->draws = cmd->passes = 0;
}

static VkResult
cmd_create(struct vk_command_pool *pool, VkCommandBufferLevel level,
           struct vk_command_buffer **out)
{
   struct mali_device *dev = container_of(pool->base.device, struct mali_device, vk);
   struct mali_cmd_buffer *cmd =
      vk_zalloc(&pool->alloc, sizeof(*cmd), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_command_buffer_init(pool, &cmd->vk, &MALI_PER_ARCH(cmd_buffer_ops), level);
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return result;
   }
   cmd->dev = dev;
   util_dynarray_init(&cmd->jm.batches, NULL);
   util_dynarray_init(&cmd->jm.frags, NULL);
   list_inithead(&cmd->slabs);
   cmd->vk.dynamic_graphics_state.vi = &cmd->dyn_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations = &cmd->dyn_sl;
   *out = &cmd->vk;
   return VK_SUCCESS;
}

static void
cmd_reset(struct vk_command_buffer *vk_cmd, VkCommandBufferResetFlags flags)
{
   struct mali_cmd_buffer *cmd = container_of(vk_cmd, struct mali_cmd_buffer, vk);
   vk_command_buffer_reset(&cmd->vk);
   cmd_clear(cmd);
}

static void
cmd_destroy(struct vk_command_buffer *vk_cmd)
{
   struct mali_cmd_buffer *cmd = container_of(vk_cmd, struct mali_cmd_buffer, vk);
   slabs_free(cmd);
   util_dynarray_fini(&cmd->jm.batches);
   util_dynarray_fini(&cmd->jm.frags);
   vk_command_buffer_finish(&cmd->vk);
   vk_free(&cmd->vk.pool->alloc, cmd);
}

const struct vk_command_buffer_ops MALI_PER_ARCH(cmd_buffer_ops) = {
   .create = cmd_create,
   .reset = cmd_reset,
   .destroy = cmd_destroy,
};

bool
mali_jm_cmd_add_batch(struct mali_cmd_buffer *cmd, const struct mali_jm_batch *batch,
                      const struct mali_jm_frag_seg *frags, uint32_t frag_count)
{
   struct mali_jm_batch b = *batch;
   b.frag_first = util_dynarray_num_elements(&cmd->jm.frags, struct mali_jm_frag_seg);
   b.frag_count = frag_count;
   if (frag_count) {
      struct mali_jm_frag_seg *f =
         util_dynarray_grow(&cmd->jm.frags, struct mali_jm_frag_seg, frag_count);
      if (!f)
         return false;
      memcpy(f, frags, frag_count * sizeof(*f));
   }
   struct mali_jm_batch *slot = util_dynarray_grow(&cmd->jm.batches, struct mali_jm_batch, 1);
   if (!slot)
      return false;
   *slot = b;
   return true;
}
