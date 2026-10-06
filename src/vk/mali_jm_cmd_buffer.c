/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The v9 command buffer, as far as submit needs it: the list of closed
 * batches and their fragment segments (mali_jm.h). Recording (chains,
 * barriers, batch closing, the draw template) is the command-buffer unit's
 * (G8), which grows this file.
 */

#include "mali_jm.h"

#include <string.h>

#include "vk_alloc.h"
#include "vk_command_pool.h"
#include "vk_log.h"

static void
cmd_clear(struct mali_cmd_buffer *cmd)
{
   util_dynarray_clear(&cmd->jm.batches);
   util_dynarray_clear(&cmd->jm.frags);
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
