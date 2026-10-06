/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The command-buffer state both back halves keep the same way, out of
 * line (the per-draw builders are inline, in mali_cmd_state.h): graphics
 * pipeline binds and their dirty bits, vertex and index buffer binds, the
 * FAU block of a graphics-stage shader, and the command buffer's TLS
 * buffer.
 */

#include "mali_cmd_state.h"

#include "vk_buffer.h"

#include "mali_arch.h"

/* ---------------------------------------------------------------------- */
/* Binding                                                                 */

/*
 * The static states other than vertex input that the draws read, copied
 * into the command buffer's dynamic state the way the runtime's
 * vk_dynamic_graphics_state_copy does it (value, then the dirty and set
 * bits when it changed), for the few states that need it.
 */
static void
bind_static_state(struct vk_dynamic_graphics_state *dyn, const struct mali_graphics_pipeline *p,
                  uint32_t m)
{
   const struct vk_dynamic_graphics_state *src = &p->state;
#define MARK(STATE)                                                            \
   do {                                                                        \
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_##STATE);                         \
      BITSET_SET(dyn->set, MESA_VK_DYNAMIC_##STATE);                           \
   } while (0)

   if (m & MALI_GFX_STATIC_VIEWPORTS) {
      const uint32_t n = src->vp.viewport_count;
      if (dyn->vp.viewport_count != n ||
          memcmp(dyn->vp.viewports, src->vp.viewports, n * sizeof(src->vp.viewports[0]))) {
         dyn->vp.viewport_count = n;
         memcpy(dyn->vp.viewports, src->vp.viewports, n * sizeof(src->vp.viewports[0]));
         MARK(VP_VIEWPORT_COUNT);
         MARK(VP_VIEWPORTS);
      }
   }
   if (m & MALI_GFX_STATIC_SCISSORS) {
      const uint32_t n = src->vp.scissor_count;
      if (dyn->vp.scissor_count != n ||
          memcmp(dyn->vp.scissors, src->vp.scissors, n * sizeof(src->vp.scissors[0]))) {
         dyn->vp.scissor_count = n;
         memcpy(dyn->vp.scissors, src->vp.scissors, n * sizeof(src->vp.scissors[0]));
         MARK(VP_SCISSOR_COUNT);
         MARK(VP_SCISSORS);
      }
   }
   if ((m & MALI_GFX_STATIC_BLEND_CONSTANTS) &&
       memcmp(dyn->cb.blend_constants, src->cb.blend_constants, sizeof(src->cb.blend_constants))) {
      memcpy(dyn->cb.blend_constants, src->cb.blend_constants, sizeof(src->cb.blend_constants));
      MARK(CB_BLEND_CONSTANTS);
   }
   /* These two from the bind block's copies: every ARMSX2 pipeline sets
    * them, and the full state is on cache lines a bind otherwise does not
    * touch. */
   if ((m & MALI_GFX_STATIC_LINE_WIDTH) && dyn->rs.line.width != p->bind.line_width) {
      dyn->rs.line.width = p->bind.line_width;
      MARK(RS_LINE_WIDTH);
   }
   if ((m & MALI_GFX_STATIC_IAL) && memcmp(&dyn->ial, &p->bind.ial, sizeof(p->bind.ial))) {
      dyn->ial = p->bind.ial;
      MARK(INPUT_ATTACHMENT_MAP);
   }
#undef MARK
}

/*
 * vkCmdBindPipeline for a graphics pipeline. A pipeline whose baked words
 * use no dynamic state (every ARMSX2 pipeline) does not go through the
 * runtime's copy of its whole static state: the draws read only vertex
 * input (pointed to), viewport, scissor, blend constants, line width and
 * the input attachment map (copied when the pipeline sets them). The
 * runtime's state is left stale for the rest, which nothing reads: a later
 * pipeline that repacks from dynamic state gets the full copy, and state
 * it leaves dynamic must be set again after its bind (Vulkan's rule for
 * state a bound pipeline had static).
 */
void
MALI_PER_ARCH(cmd_bind_graphics)(struct mali_cmd_buffer *cmd, struct mali_graphics_pipeline *p)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct mali_gfx_bind *bd = &p->bind;

   if (unlikely(bd->dynamic)) {
      vk_cmd_set_dynamic_graphics_state(&cmd->vk, &p->state);
      d->vi = dyn->vi;
      d->vi_strides = dyn->vi_binding_strides;
   } else {
      const uint32_t m = bd->static_mask;
      d->vi = (m & MALI_GFX_STATIC_VI) ? &p->vi : dyn->vi;
      d->vi_strides = (m & MALI_GFX_STATIC_STRIDES) ? p->state.vi_binding_strides
                                                     : dyn->vi_binding_strides;
      if (m & ~(MALI_GFX_STATIC_VI | MALI_GFX_STATIC_STRIDES))
         bind_static_state(dyn, p, m);
   }

   if (d->pipeline == p)
      return;
   d->pipeline = p;
   d->dirty |= MALI_GFX_DIRTY_PIPELINE | MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_FS_FAU |
               MALI_GFX_DIRTY_BLEND | MALI_GFX_DIRTY_ZSD;
   /* Tables whose inputs from the pipeline side did not change stay. */
   if (!bd->vs_srt_key || bd->vs_srt_key != d->vs_srt_key)
      d->dirty |= MALI_GFX_DIRTY_VS_SRT;
   if (!bd->fs_srt_key || bd->fs_srt_key != d->fs_srt_key)
      d->dirty |= MALI_GFX_DIRTY_FS_SRT;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdBindVertexBuffers2)(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                                     uint32_t bindingCount, const VkBuffer *pBuffers,
                                     const VkDeviceSize *pOffsets, const VkDeviceSize *pSizes,
                                     const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;

   assert(firstBinding + bindingCount <= MALI_MAX_VBS);
   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(vk_buffer, buf, pBuffers[i]);
      const uint32_t b = firstBinding + i;
      if (!buf) {
         d->vb[b].addr = 0;
         d->vb[b].size = 0;
         continue;
      }
      d->vb[b].addr = vk_buffer_address(buf, pOffsets[i]);
      d->vb[b].size = vk_buffer_range(buf, pOffsets[i], pSizes ? pSizes[i] : VK_WHOLE_SIZE);
   }
   if (pStrides)
      vk_cmd_set_vertex_binding_strides(&cmd->vk, firstBinding, bindingCount, pStrides);
   d->dirty |= MALI_GFX_DIRTY_VS_SRT;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdBindIndexBuffer2KHR)(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                      VkDeviceSize offset, VkDeviceSize size, VkIndexType indexType)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, buf, buffer);
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;

   d->ib.index_size = vk_index_type_to_bytes(indexType);
   if (buf) {
      d->ib.addr = vk_buffer_address(buf, offset);
      d->ib.size = (uint32_t)MIN2(vk_buffer_range(buf, offset, size), UINT32_MAX);
   } else {
      d->ib.addr = 0;
      d->ib.size = 0;
   }
   d->dirty |= MALI_GFX_DIRTY_INDEX;
}

/* ---------------------------------------------------------------------- */
/* FAU                                                                     */

uint64_t
MALI_PER_ARCH(cmd_gfx_fau)(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                           const void *sysvals, const void *push, uint32_t push_size)
{
   if (!s || !s->fau.total_count)
      return 0;

   const uint32_t total = s->fau.total_count;
   struct mali_ptr p = mali_cmd_alloc(cmd, total * 8, 16);
   if (!p.cpu)
      return 0;

   /* The sysval words the shader reads, with the block's own address in
    * common.push_uniforms, then its push-constant words (zero past the
    * end of push). Read in place: this runs twice a draw. */
   const unsigned push_uniforms_word =
      offsetof(struct mali_graphics_sysvals, common.push_uniforms) / MALI_FAU_WORD_SIZE;
   uint64_t tmp[MALI_FAU_WORD_COUNT];
   unsigned n = 0, w;
   BITSET_FOREACH_SET(w, s->fau.used_sysvals, MALI_MAX_SYSVAL_FAUS) {
      tmp[n++] = w == push_uniforms_word ?
                    p.gpu :
                    mali_sysval_word(sysvals, sizeof(struct mali_graphics_sysvals), w);
   }
   const uint32_t push_len = MIN2(push_size, MALI_MAX_PUSH_CONST_FAUS * MALI_FAU_WORD_SIZE);
   BITSET_FOREACH_SET(w, s->fau.used_push_consts, MALI_MAX_PUSH_CONST_FAUS) {
      uint64_t v = 0;
      const uint32_t off = w * MALI_FAU_WORD_SIZE;
      if (off < push_len)
         memcpy(&v, (const uint8_t *)push + off, MIN2(push_len - off, sizeof(v)));
      tmp[n++] = v;
   }
   n += mali_shader_fau_consts(&s->fau, &tmp[n]);
   /* Uniform-buffer words (v11): zero here, copied in by the GPU
    * (mali_cmd_draw.c emit_ubo_push). */
   assert(!s->fau.ubo_push_count || n == s->fau.ubo_push_start);
   for (unsigned i = n; i < total; i++)
      tmp[i] = 0;

   pan_fau_foreach_imm(&s->info.fau, i) {
      bool hi = i & 1;
      unsigned idx = i / 2;
      assert(idx < total);
      tmp[idx] = (tmp[idx] & ((uint64_t)UINT32_MAX << (32 * !hi))) |
                 ((uint64_t)s->info.fau.words[i].constant << (32 * hi));
   }

   memcpy(p.cpu, tmp, total * sizeof(tmp[0]));
   return p.gpu | ((uint64_t)total << 56);
}

/* ---------------------------------------------------------------------- */
/* Thread-local storage                                                    */

uint64_t
MALI_PER_ARCH(cmd_tls_buffer)(struct mali_cmd_buffer *cmd, uint32_t tls_size)
{
   struct mali_device *dev = cmd->dev;
   const struct mali_thread_props tp = mali_thread_props(dev);

   /* Per thread: the next power of two of the size, at least 16 bytes; for
    * every thread slot of every core (pan_get_total_stack_size). */
   unsigned per_thread = util_next_power_of_two(ALIGN_POT(tls_size, 16));
   if (per_thread > cmd->tls.size) {
      uint64_t total = (uint64_t)per_thread * tp.max_threads_per_core * tp.core_id_range;
      /* TLS memory: GPU group 9. */
      struct mali_cmd_slab *s = calloc(1, sizeof(*s));
      const struct mali_kbase_alloc_info ai = {
         .size = total,
         .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_TLS),
         .mem_class = MALI_KBASE_MEM_CLASS_INTERNAL_TLS,
      };
      if (!s || mali_kbase_alloc(dev->kbase, &ai, &s->bo) != MALI_KBASE_SUCCESS) {
         free(s);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return 0;
      }
      list_addtail(&s->link, &cmd->slabs);
      cmd->tls.size = per_thread;
      cmd->tls.gpu = s->bo.gpu_va;
   }
   return cmd->tls.gpu;
}
