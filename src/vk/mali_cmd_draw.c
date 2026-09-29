/*
 * Copyright 2026 The libmali authors
 * SPDX-License-Identifier: MIT
 */

/*
 * Draws: graphics pipeline binds, vertex and index buffers, vkCmdDraw /
 * vkCmdDrawIndexed as RUN_IDVS on the vertex/tiler subqueue, full-screen
 * draws (RUN_FULLSCREEN) for vkCmdClearAttachments and in-pass barriers.
 *
 * The encoding follows panvk v11 (csf/panvk_vX_cmd_draw.c: prepare_draw,
 * launch_draw, cmd_run_fullscreen, cmd_fb_barrier), "which source wins"
 * for command-stream encoding; the per-pipeline words are the ones our
 * pipeline compile baked in (mali_pipeline_state.c). Differences:
 *
 *  - RUN_IDVS sets bit 55, which v11.xml does not define, as every blob
 *    RUN_IDVS does (read as "Tiler Flags2 Enable", the RUN_FULLSCREEN
 *    field at the same bit). TILER_FLAGS2 is written as 0 before every
 *    draw either way.
 *  - The FAU blocks carry our sysval layout (mali_shader.h), built only
 *    when their inputs changed.
 *
 * Register use on the vertex/tiler subqueue is the v11 IDVS staging set;
 * state registers persist between draws and are rewritten from dirty
 * bits (MALI_GFX_DIRTY_*).
 */

#include "mali_blend.h"
#include "mali_cmd_buffer.h"
#include "mali_measure.h"

#include <string.h>

#include "util/format/u_format.h"
#include "util/u_math.h"
#include "vk_buffer.h"
#include "vk_format.h"
#include "vk_log.h"

#include "pan_format.h"

#include "mali_descriptor_set.h"
#include "mali_image.h"
#include "mali_pipeline.h"
#include "mali_queue.h"
#include "mali_vk.h"

#if PAN_ARCH != 11
#error "mali_cmd_draw.c is written for arch v11"
#endif

#define IDVS(reg) MALI_IDVS_SR_##reg

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
mali_cmd_bind_graphics(struct mali_cmd_buffer *cmd, struct mali_graphics_pipeline *p)
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
mali_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t firstBinding,
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
mali_CmdBindIndexBuffer2KHR(VkCommandBuffer commandBuffer, VkBuffer buffer,
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

/*
 * The fragment shader's pushed uniform-buffer words (mali_shader_fau::
 * ubo_push): the vertex/tiler stream copies them from the buffer into the
 * FAU block at fau_va (LOAD_MULTIPLE into scratch registers,
 * STORE_MULTIPLE into the block), ahead of the draw's RUN_IDVS. The GPU
 * reads the buffer when it runs the command buffer, as for any uniform
 * buffer read; the CPU never reads it (ARMSX2's rings are write-combined,
 * and CPU reads of them cost the GS thread up to 25 % on draw-heavy
 * dumps). Nothing waits for the stores here: the
 * fragment shader runs in the pass's fragment job, after flush_tiling has
 * waited for them. Runs of adjacent words share one load and one store;
 * words past the binding's range are not copied and stay zero (the CPU
 * wrote the block).
 */
static void
emit_ubo_push_general(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                      uint64_t fau_va)
{
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);
   const struct mali_desc_state *desc = &cmd->gfx.desc;
   /* Scratch 0-1 the source address, 2-3 the FAU block, 4-15 data. */
   enum { SRC = 0, DST = 2, DATA = 4, DATA_REGS = 12 };
   struct {
      unsigned reg, count;
      int dst_off;
   } pend[DATA_REGS];
   unsigned npend = 0, used = 0;
   uint64_t cur_src = 0;

   /* The data registers may still be the source of the previous draw's
    * stores. */
   cs_flush_stores(b);

#define FLUSH_PENDING()                                                        \
   do {                                                                        \
      if (npend) {                                                             \
         cs_flush_loads(b);                                                    \
         cs_move64_to(b, mali_cs_scratch_reg64(b, DST), fau_va);               \
         for (unsigned k = 0; k < npend; k++)                                  \
            cs_store(b, mali_cs_scratch_reg_tuple(b, pend[k].reg, pend[k].count), \
                     mali_cs_scratch_reg64(b, DST), BITFIELD_MASK(pend[k].count), \
                     pend[k].dst_off);                                         \
      }                                                                        \
      npend = used = 0;                                                        \
   } while (0)

   const uint32_t n = s->fau.ubo_push_count;
   for (uint32_t i = 0; i < n;) {
      const struct mali_ubo_push_word w0 = s->fau.ubo_push[i];
      const uint32_t h = s->desc.dyn_bufs.map[w0.dyn];
      const uint32_t set = MALI_COPY_DESC_HANDLE_SET(h);
      const uint32_t idx = MALI_COPY_DESC_HANDLE_INDEX(h);
      const struct mali_descriptor_set *ds = desc->sets[set];
      const uint64_t range = ds ? ds->dyn_bufs[idx].range : 0;
      const uint64_t addr = ds ? ds->dyn_bufs[idx].addr + desc->dyn_offsets[set][idx] : 0;

      /* A run: consecutive words of one buffer, whole words inside the
       * range, the last may be half a word. */
      unsigned regs = 0, k = 0;
      while (i + k < n && regs < DATA_REGS) {
         const struct mali_ubo_push_word w = s->fau.ubo_push[i + k];
         if (w.dyn != w0.dyn || w.offset != w0.offset + 8 * k)
            break;
         const unsigned valid = w.offset >= range ? 0 :
                                range - w.offset >= 8 ? 2 :
                                range - w.offset >= 4 ? 1 : 0;
         if (!valid || regs + valid > DATA_REGS)
            break;
         regs += valid;
         k++;
         if (valid < 2)
            break;
      }
      if (!k || !addr) {
         i++;   /* nothing of this word is inside the binding */
         continue;
      }

      if (used + regs > DATA_REGS)
         FLUSH_PENDING();
      if (addr != cur_src) {
         cs_move64_to(b, mali_cs_scratch_reg64(b, SRC), addr);
         cur_src = addr;
      }
      cs_load_to(b, mali_cs_scratch_reg_tuple(b, DATA + used, regs),
                 mali_cs_scratch_reg64(b, SRC), BITFIELD_MASK(regs), w0.offset);
      pend[npend].reg = DATA + used;
      pend[npend].count = regs;
      pend[npend].dst_off = (s->fau.ubo_push_start + i) * MALI_FAU_WORD_SIZE;
      npend++;
      used += regs;
      i += k;
   }
   FLUSH_PENDING();
#undef FLUSH_PENDING
}

uint64_t
mali_cmd_gfx_fau(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
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
   /* Uniform-buffer words: zero here, copied by the GPU (emit_ubo_push). */
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

/* The sysvals every graphics stage of the current draw sees. */
static void
fill_sysvals(struct mali_cmd_buffer *cmd, struct mali_graphics_sysvals *sv)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_graphics_pipeline *p = d->pipeline;

   memset(sv, 0, sizeof(*sv));
   for (unsigned i = 0; i < 4; i++)
      sv->blend.constants[i] = dyn->cb.blend_constants[i];

   /* Viewport transform (the vertex shader applies it): scale (w/2, h/2,
    * zmax - zmin), offset (x + w/2, y + h/2, zmin). */
   const VkViewport *vp = &dyn->vp.viewports[0];
   sv->viewport[0].scale = 0.5f * vp->width;
   sv->viewport[1].scale = 0.5f * vp->height;
   sv->viewport[2].scale = vp->maxDepth - vp->minDepth;
   sv->viewport[0].offset = 0.5f * vp->width + vp->x;
   sv->viewport[1].offset = 0.5f * vp->height + vp->y;
   sv->viewport[2].offset = vp->minDepth;

   sv->vs.first_vertex = d->first_vertex;
   sv->vs.base_instance = d->base_instance;
   sv->vs.noperspective_varyings = p->bind.noperspective;
   memcpy(sv->fs.blend_descs, d->blend_descs, sizeof(sv->fs.blend_descs));

   /* Input attachments: colour targets through the tile buffer with their
    * conversion, depth/stencil as the ZS target (panvk
    * prepare_iam_sysvals). */
   memset(sv->iam, 0xff, sizeof(sv->iam));
   const struct vk_input_attachment_location_state *ial = &dyn->ial;
   const struct mali_render_state *r = &cmd->gfx.render;
   const uint32_t catt = ial->color_attachment_count == MESA_VK_COLOR_ATTACHMENT_COUNT_UNKNOWN ?
                            MALI_MAX_RTS : ial->color_attachment_count;
   for (uint32_t i = 0; i < catt && i < MALI_MAX_RTS; i++) {
      if (ial->color_map[i] == MESA_VK_ATTACHMENT_UNUSED || i >= r->desc.rt_count ||
          !r->desc.rt[i].image)
         continue;
      const uint32_t idx = ial->color_map[i] + 1;
      if (idx >= MALI_INPUT_ATTACHMENT_MAP_SIZE)
         continue;
      struct mali_internal_conversion_packed conv;
      pan_pack(&conv, INTERNAL_CONVERSION, cfg) {
         cfg.memory_format =
            GENX(pan_dithered_format_from_pipe_format)(r->desc.rt[i].format, false);
      }
      sv->iam[idx].target = MALI_COLOR_ATTACHMENT(i);
      sv->iam[idx].conversion = conv.opaque[0];
   }
   if (ial->depth_att != MESA_VK_ATTACHMENT_UNUSED) {
      uint32_t idx = ial->depth_att == MESA_VK_ATTACHMENT_NO_INDEX ? 0 : ial->depth_att + 1;
      if (idx < MALI_INPUT_ATTACHMENT_MAP_SIZE) {
         sv->iam[idx].target = MALI_ZS_ATTACHMENT;
         sv->iam[idx].conversion = 0;
      }
   }
   if (ial->stencil_att != MESA_VK_ATTACHMENT_UNUSED) {
      uint32_t idx = ial->stencil_att == MESA_VK_ATTACHMENT_NO_INDEX ? 0 : ial->stencil_att + 1;
      if (idx < MALI_INPUT_ATTACHMENT_MAP_SIZE) {
         sv->iam[idx].target = MALI_ZS_ATTACHMENT;
         sv->iam[idx].conversion = 0;
      }
   }
}

/* ---------------------------------------------------------------------- */
/* Resource tables                                                         */

static void
pack_dyn_bufs(const struct mali_shader_desc_info *di, const struct mali_desc_state *desc,
              uint8_t *out)
{
   for (uint32_t i = 0; i < di->dyn_bufs.count; i++) {
      uint32_t h = di->dyn_bufs.map[i];
      uint32_t set = MALI_COPY_DESC_HANDLE_SET(h);
      uint32_t idx = MALI_COPY_DESC_HANDLE_INDEX(h);
      const struct mali_descriptor_set *s = desc->sets[set];
      uint8_t *o = out + i * MALI_DESCRIPTOR_SIZE;
      if (s)
         mali_descriptor_set_pack_dyn_buf(s, idx, desc->dyn_offsets[set][idx], o);
      else
         memset(o, 0, MALI_DESCRIPTOR_SIZE);
   }
}

/* Resource table: entry 0 the driver set (already built), entry N + 1
 * descriptor set N. */
static uint64_t
build_res_table(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                const struct mali_desc_state *desc, uint64_t drv, uint32_t drv_count)
{
   const struct mali_shader_desc_info *di = &s->desc;
   uint32_t last = util_last_bit(di->used_set_mask);
   uint32_t count = ALIGN_POT(1 + last, MALI_RESOURCE_TABLE_SIZE_ALIGNMENT);
   struct mali_ptr t = mali_cmd_alloc(cmd, count * MALI_RESOURCE_SIZE, 64);
   if (!t.cpu)
      return 0;

   pan_cast_and_pack(t.cpu, RESOURCE, cfg) {
      cfg.address = drv;
      cfg.size = drv_count * MALI_DESCRIPTOR_SIZE;
      cfg.contains_descriptors = true;
   }
   for (uint32_t i = 1; i < count; i++) {
      uint32_t set = i - 1;
      const struct mali_descriptor_set *ds =
         (set < MALI_MAX_SETS && (di->used_set_mask & BITFIELD_BIT(set))) ? desc->sets[set]
                                                                          : NULL;
      mali_descriptor_set_pack_resource(ds, (uint8_t *)t.cpu + i * MALI_RESOURCE_SIZE);
   }
   return t.gpu | count;
}

/* One vertex attribute (panvk emit_vs_attrib). */
static void
pack_attrib(const struct vk_vertex_input_state *vi, const uint16_t *strides, uint32_t i,
            uint32_t vb_offset, uint32_t base_instance, void *out)
{
   const struct vk_vertex_attribute_state *a = &vi->attributes[i];
   const struct vk_vertex_binding_state *bs = &vi->bindings[a->binding];
   const uint32_t stride = strides[a->binding];
   const bool per_instance = bs->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   const enum pipe_format f = vk_format_to_pipe_format(a->format);

   pan_cast_and_pack(out, ATTRIBUTE, cfg) {
      cfg.offset = a->offset + (per_instance ? base_instance * stride : 0);
      cfg.format = GENX(pan_format_from_pipe_format)(f)->hw;
      cfg.table = 0;
      cfg.buffer_index = vb_offset + a->binding;
      cfg.stride = stride;
      if (!per_instance) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_VERTEX;
         cfg.offset_enable = true;
      } else if (bs->divisor == 1) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
      } else if (bs->divisor == 0) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.stride = 0;
      } else if (util_is_power_of_two_or_zero(bs->divisor)) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D_POT_DIVISOR;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.divisor_r = __builtin_ctz(bs->divisor);
      } else {
         /* Not reachable for ARMSX2 (no divisors); the NPOT magic numbers
          * are not implemented. */
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
      }
   }
}

/* Vertex stage table 0: 16 attributes, the dummy sampler, the dynamic
 * buffers, then one Buffer per vertex binding (panvk
 * prepare_vs_driver_set). */
static uint64_t
build_vs_srt(struct mali_cmd_buffer *cmd)
{
   const struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_shader *vs = d->pipeline->vs;
   const struct vk_vertex_input_state *vi = d->vi;

   uint32_t vb_count = 0;
   u_foreach_bit(i, vi->attributes_valid)
      vb_count = MAX2(vi->attributes[i].binding + 1, vb_count);
   const uint32_t vb_offset = MALI_MAX_VS_ATTRIBS + 1 + vs->desc.dyn_bufs.count;
   const uint32_t count = vb_offset + vb_count;

   struct mali_ptr p = mali_cmd_alloc(cmd, count * MALI_DESCRIPTOR_SIZE, MALI_DESCRIPTOR_SIZE);
   if (!p.cpu)
      return 0;
   uint8_t *descs = p.cpu;

   for (uint32_t i = 0; i < MALI_MAX_VS_ATTRIBS; i++) {
      void *o = descs + i * MALI_DESCRIPTOR_SIZE;
      if (vi->attributes_valid & BITFIELD_BIT(i)) {
         pack_attrib(vi, d->vi_strides, i, vb_offset, d->base_instance, o);
      } else {
         /* An invalid table makes reads out of bounds (panvk). */
         pan_cast_and_pack(o, ATTRIBUTE, cfg) {
            cfg.table = 17;
            cfg.format = (MALI_R16F << 12) | MALI_RGB_COMPONENT_ORDER_RGBA;
         }
      }
   }
   mali_pack_dummy_sampler(descs + MALI_MAX_VS_ATTRIBS * MALI_DESCRIPTOR_SIZE);
   pack_dyn_bufs(&vs->desc, &cmd->gfx.desc,
                 descs + (MALI_MAX_VS_ATTRIBS + 1) * MALI_DESCRIPTOR_SIZE);
   for (uint32_t i = 0; i < vb_count; i++) {
      void *o = descs + (vb_offset + i) * MALI_DESCRIPTOR_SIZE;
      if ((vi->bindings_valid & BITFIELD_BIT(i)) && d->vb[i].addr) {
         pan_cast_and_pack(o, BUFFER, cfg) {
            cfg.address = d->vb[i].addr;
            cfg.size = (uint32_t)MIN2(d->vb[i].size, UINT32_MAX);
         }
      } else {
         pan_cast_and_pack(o, NULL_DESCRIPTOR, cfg);
      }
   }
   return build_res_table(cmd, vs, &cmd->gfx.desc, p.gpu, count);
}

/* Fragment varyings read with LD_VAR (panvk emit_varying_descs). */
static void
pack_varying_descs(const struct mali_shader *vs, const struct mali_shader *fs, uint8_t *out)
{
   const struct pan_varying_layout *vl = &vs->info.varyings.formats;
   const struct pan_varying_layout *fl = &fs->info.varyings.formats;

   for (uint32_t i = 0; i < fl->count && i < fs->desc.driver_table_prefix; i++) {
      const struct pan_varying_slot *fslot = pan_varying_layout_slot_at(fl, i);
      if (!fslot || fslot->section != PAN_VARYING_SECTION_GENERIC)
         continue;
      unsigned offset = 0;
      enum pipe_format format = PIPE_FORMAT_NONE;
      const struct pan_varying_slot *vslot = pan_varying_layout_find_slot(vl, fslot->location);
      if (vslot) {
         nir_alu_type base = nir_alu_type_get_base_type(fslot->alu_type);
         nir_alu_type bits = nir_alu_type_get_type_size(vslot->alu_type);
         offset = vslot->offset;
         format = pan_varying_format(base | bits, vslot->ncomps);
      }
      pan_cast_and_pack(out + i * MALI_DESCRIPTOR_SIZE, ATTRIBUTE, cfg) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_VERTEX_PACKET;
         cfg.offset_enable = false;
         cfg.format = GENX(pan_format_from_pipe_format)(format)->hw;
         cfg.table = 61;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_VERTEX;
         cfg.offset = 1024 + offset;
         cfg.buffer_index = 0;
         cfg.attribute_stride = vl->generic_size_B;
         cfg.packet_stride = vl->generic_size_B + 16;
      }
   }
}

static uint64_t
build_fs_srt(struct mali_cmd_buffer *cmd)
{
   const struct mali_graphics_pipeline *p = cmd->gfx.draw.pipeline;
   const struct mali_shader *fs = p->fs;
   const uint32_t prefix = fs->desc.driver_table_prefix;
   const uint32_t count = prefix + 1 + fs->desc.dyn_bufs.count;

   struct mali_ptr t = mali_cmd_alloc(cmd, count * MALI_DESCRIPTOR_SIZE, MALI_DESCRIPTOR_SIZE);
   if (!t.cpu)
      return 0;
   uint8_t *descs = t.cpu;
   memset(descs, 0, prefix * MALI_DESCRIPTOR_SIZE);
   if (fs->desc.needs_varying_descs)
      pack_varying_descs(p->vs, fs, descs);
   mali_pack_dummy_sampler(descs + prefix * MALI_DESCRIPTOR_SIZE);
   pack_dyn_bufs(&fs->desc, &cmd->gfx.desc, descs + (prefix + 1) * MALI_DESCRIPTOR_SIZE);
   return build_res_table(cmd, fs, &cmd->gfx.desc, t.gpu, count);
}

/* ---------------------------------------------------------------------- */
/* Blend                                                                   */

/* The blend constant for fixed-function blending: UNORM of the target's
 * channel size, in the top bits of 16 (pan_pack_blend_constant). Blend
 * factors of UNORM targets are clamped to [0, 1] (Vulkan, "Blend
 * Factors"); ARMSX2 passes AFIX / 128, up to 1.99. */
static uint16_t
pack_blend_constant(enum pipe_format format, float c)
{
   const struct util_format_description *desc = util_format_description(format);
   unsigned size = 0;
   for (unsigned i = 0; i < desc->nr_channels; i++)
      size = MAX2(desc->channel[0].size, size);
   if (!size || size > 16)
      return 0;
   float factor = (float)(((1u << size) - 1) << (16 - size));
   return (uint16_t)(CLAMP(c, 0.0f, 1.0f) * factor);
}

/* The blend constant channels `mask` reads are all the same value
 * (fixed-function blending has one constant). */
static bool
homogeneous_constant(unsigned mask, const float *consts)
{
   const float c = consts[ffs(mask) - 1];
   u_foreach_bit(i, mask) {
      if (consts[i] != c)
         return false;
   }
   return true;
}

/* The address of the blend shader for render target `i`, 0 if there is
 * none (compile failure, logged). */
static uint64_t
blend_shader_for(struct mali_cmd_buffer *cmd, const struct mali_graphics_pipeline *p, unsigned i)
{
   const struct mali_render_state *r = &cmd->gfx.render;
   struct mali_blend_shader_key key;
   mali_blend_shader_key_init(&key, &p->baked, i, r->desc.rt[i].format,
                              r->desc.rt[i].image->vk.samples, &p->fs->info);

   uint64_t addr = 0;
   if (mali_blend_shader_get(cmd->dev, &key, &addr) != VK_SUCCESS) {
      static bool warned;
      if (!warned) {
         warned = true;
         mesa_logw("libmali: a blend shader failed to compile; the colour is stored "
                   "unblended");
      }
      return 0;
   }
   /* The descriptor holds 32 bits of the address; the fragment shader's
    * jump supplies the rest. */
   assert(addr >> 32 == p->fs->code.gpu_va >> 32);
   assert(!(addr & 15));
   return addr;
}

/*
 * The Blend descriptors of the current pipeline for the current render
 * pass (panvk blend_emit_descs with the equations our pipeline compile
 * baked in), and the internal words the fragment shader's BLEND
 * instructions read (by output location).
 *
 * A target uses a blend shader when the pipeline says so, or when it uses
 * the blend constant in fixed function and either its channels disagree or
 * the value differs from the one an earlier target already uses: the
 * hardware has a single constant (panvk blend_needs_shader).
 */
static uint64_t
build_blend(struct mali_cmd_buffer *cmd, uint32_t *count_out)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_graphics_pipeline *p = d->pipeline;
   const struct mali_gfx_baked *bk = &p->baked;
   const struct mali_render_state *r = &cmd->gfx.render;
   const float *consts = cmd->vk.dynamic_graphics_state.cb.blend_constants;
   const uint32_t count = r->rt_count;

   struct mali_ptr m = mali_cmd_alloc(cmd, count * pan_size(BLEND), 64);
   if (!m.cpu)
      return 0;

   /* Which targets need a blend shader, and the fixed-function constant
    * (all ones: none yet). */
   uint64_t shader[MALI_MAX_RTS] = {0};
   bool opaque_fallback[MALI_MAX_RTS] = {false};
   uint32_t ff_constant = ~0u;
   for (uint32_t i = 0; i < count && i < bk->rt_count; i++) {
      const struct mali_blend_rt_baked *rt = &bk->blend[i];
      const bool have_rt = i < r->desc.rt_count && r->desc.rt[i].image;
      if (!p->fs || rt->mode == MALI_BLEND_RT_OFF || !have_rt)
         continue;

      bool use_shader = rt->mode == MALI_BLEND_RT_SHADER;
      if (rt->mode == MALI_BLEND_RT_FIXED_FUNCTION && rt->constant_mask) {
         const uint32_t c = pack_blend_constant(r->desc.rt[i].format,
                                                consts[ffs(rt->constant_mask) - 1]);
         if (!homogeneous_constant(rt->constant_mask, consts) ||
             (ff_constant != ~0u && c != ff_constant))
            use_shader = true;
         else
            ff_constant = c;
      }
      if (use_shader) {
         shader[i] = blend_shader_for(cmd, p, i);
         opaque_fallback[i] = !shader[i];
      }
   }
   if (ff_constant == ~0u)
      ff_constant = 0;

   d->blend_shader_mask = 0;
   for (uint32_t i = 0; i < count; i++)
      d->blend_shader_mask |= shader[i] ? BITFIELD_BIT(i) : 0;

   memset(d->blend_descs, 0, sizeof(d->blend_descs));
   struct mali_blend_packed off;
   pan_pack(&off, BLEND, cfg) {
      cfg.enable = false;
      cfg.internal.mode = MALI_BLEND_MODE_OFF;
   }
   for (unsigned loc = 0; loc < MALI_MAX_RTS; loc++)
      d->blend_descs[loc] = off.opaque[2] | (uint64_t)off.opaque[3] << 32;

   /* Packed here and copied out at the end: m is uncached, and reading
    * back from it (the internal words below) stalls for a memory round
    * trip. */
   struct mali_blend_packed descs[MALI_MAX_RTS];
   assert(count <= MALI_MAX_RTS);
   for (uint32_t i = 0; i < count; i++) {
      struct mali_blend_packed *out = &descs[i];
      const struct mali_blend_rt_baked *rt = i < bk->rt_count ? &bk->blend[i] : NULL;
      const bool have_rt = i < r->desc.rt_count && r->desc.rt[i].image;

      if (!p->fs || !rt || rt->mode == MALI_BLEND_RT_OFF || !have_rt) {
         *out = off;
         continue;
      }

      const enum pipe_format fmt = r->desc.rt[i].format;
      if (shader[i]) {
         pan_pack(out, BLEND, cfg) {
            cfg.srgb = util_format_is_srgb(fmt);
            cfg.load_destination = rt->load_destination;
            cfg.round_to_fb_precision = true;
            cfg.blend_constant = ff_constant;
            cfg.internal.mode = MALI_BLEND_MODE_SHADER;
            cfg.internal.shader.pc = (uint32_t)shader[i];
         }
      } else {
         enum mali_blend_rt_mode mode = opaque_fallback[i] ? MALI_BLEND_RT_OPAQUE : rt->mode;
         pan_pack(out, BLEND, cfg) {
            cfg.srgb = util_format_is_srgb(fmt);
            cfg.load_destination = rt->load_destination && mode == MALI_BLEND_RT_FIXED_FUNCTION;
            cfg.round_to_fb_precision = true;
            cfg.blend_constant = ff_constant;
            cfg.internal.mode = mode == MALI_BLEND_RT_OPAQUE ? MALI_BLEND_MODE_OPAQUE
                                                             : MALI_BLEND_MODE_FIXED_FUNCTION;
            cfg.internal.fixed_function.num_comps = 4;
            cfg.internal.fixed_function.conversion.memory_format =
               GENX(pan_dithered_format_from_pipe_format)(fmt, false);
            if (cfg.internal.mode == MALI_BLEND_MODE_FIXED_FUNCTION &&
                (cfg.internal.fixed_function.conversion.memory_format & 0xff) ==
                   MALI_RGB_COMPONENT_ORDER_RGB1) {
               /* Fixed-function blending does not take RGB1 (panvk). */
               cfg.internal.fixed_function.conversion.memory_format &= ~0xff;
               cfg.internal.fixed_function.conversion.memory_format |=
                  MALI_RGB_COMPONENT_ORDER_RGBA;
            }
            cfg.internal.fixed_function.rt = i;
         }
         /* The baked equation word (descriptor word 1); a failed blend
          * shader stores the source, colour mask applied. */
         if (rt->mode == MALI_BLEND_RT_SHADER) {
            struct mali_blend_equation_packed replace;
            pan_pack(&replace, BLEND_EQUATION, cfg) {
               cfg.rgb.a = MALI_BLEND_OPERAND_A_SRC;
               cfg.rgb.b = MALI_BLEND_OPERAND_B_SRC;
               cfg.rgb.c = MALI_BLEND_OPERAND_C_ZERO;
               cfg.alpha.a = MALI_BLEND_OPERAND_A_SRC;
               cfg.alpha.b = MALI_BLEND_OPERAND_B_SRC;
               cfg.alpha.c = MALI_BLEND_OPERAND_C_ZERO;
               cfg.color_mask = rt->eq.color_mask;
            }
            out->opaque[1] = replace.opaque[0];
         } else {
            out->opaque[1] = rt->equation;
         }
      }

      if (rt->shader_location < MALI_MAX_RTS)
         d->blend_descs[rt->shader_location] =
            out->opaque[2] | (uint64_t)out->opaque[3] << 32;
   }

   memcpy(m.cpu, descs, count * sizeof(descs[0]));
   *count_out = count;
   return m.gpu;
}

/* ---------------------------------------------------------------------- */
/* Viewport and scissor                                                    */

static uint64_t
scissor_box(const struct mali_cmd_buffer *cmd)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const VkViewport *vp = &dyn->vp.viewports[0];
   const VkRect2D *sc = &dyn->vp.scissors[0];

   /* The viewport rectangle clipped by the scissor (panvk v11
    * prepare_vp); the hardware clips to the box. */
   int minx = (int)vp->x;
   int maxx = (int)(vp->x + vp->width);
   int miny = (int)MIN2(vp->y, vp->y + vp->height);
   int maxy = (int)MAX2(vp->y, vp->y + vp->height);
   minx = MAX2(sc->offset.x, minx);
   miny = MAX2(sc->offset.y, miny);
   maxx = MIN2(sc->offset.x + (int)sc->extent.width, maxx);
   maxy = MIN2(sc->offset.y + (int)sc->extent.height, maxy);
   maxx = maxx > minx ? maxx - 1 : maxx;
   maxy = maxy > miny ? maxy - 1 : maxy;

   struct mali_scissor_packed s;
   pan_pack(&s, SCISSOR, cfg) {
      cfg.scissor_minimum_x = CLAMP(minx, 0, UINT16_MAX);
      cfg.scissor_minimum_y = CLAMP(miny, 0, UINT16_MAX);
      cfg.scissor_maximum_x = CLAMP(maxx, 0, UINT16_MAX);
      cfg.scissor_maximum_y = CLAMP(maxy, 0, UINT16_MAX);
   }
   return s.opaque[0] | (uint64_t)s.opaque[1] << 32;
}

/* ---------------------------------------------------------------------- */
/* Draw                                                                    */

/* Dynamic states this driver reads at draw time, per register group. */
static void
collect_dynamic_dirty(struct mali_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;

   if (BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VP_VIEWPORTS) ||
       BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VP_SCISSORS)) {
      d->dirty |= MALI_GFX_DIRTY_VIEWPORT | MALI_GFX_DIRTY_VS_FAU;
   }
   if (BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_CONSTANTS))
      d->dirty |= MALI_GFX_DIRTY_BLEND | MALI_GFX_DIRTY_FS_FAU;
   if (BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_RS_LINE_WIDTH))
      d->dirty |= MALI_GFX_DIRTY_PIPELINE;
   if (BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI) ||
       BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI_BINDING_STRIDES) ||
       BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI_BINDINGS_VALID))
      d->dirty |= MALI_GFX_DIRTY_VS_SRT;
   if (BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_INPUT_ATTACHMENT_MAP))
      d->dirty |= MALI_GFX_DIRTY_FS_FAU;
   if (d->pipeline && d->pipeline->bind.dynamic) {
      /* Words baked from state the pipeline left dynamic: repack. */
      d->dirty |= MALI_GFX_DIRTY_PIPELINE | MALI_GFX_DIRTY_ZSD | MALI_GFX_DIRTY_BLEND |
                  MALI_GFX_DIRTY_FS_FAU;
   }
   vk_dynamic_graphics_state_clear_dirty(dyn);
}

static enum mali_index_type
index_type(uint32_t size)
{
   switch (size) {
   case 1: return MALI_INDEX_TYPE_UINT8;
   case 2: return MALI_INDEX_TYPE_UINT16;
   case 4: return MALI_INDEX_TYPE_UINT32;
   default: return MALI_INDEX_TYPE_NONE;
   }
}

/* Bit 55 of RUN_IDVS; the on-device smoke test clears it once to check
 * what it does. */
bool mali_idvs_bit55 = true;

/* RUN_IDVS with bit 55 set. Selects as panvk v11: varying stage from
 * group 0/1 (SPD_1), fragment from group 2, both with TSD_0. */
static uint64_t
run_idvs_ins(uint32_t flags_override)
{
   struct mali_cs_run_idvs_packed ins;
   pan_pack(&ins, CS_RUN_IDVS, I) {
      I.flags_override = flags_override;
      I.malloc_enable = true;
      I.draw_id_register_enable = false;
      I.varying_srt_select = false;
      I.varying_fau_select = false;
      I.varying_tsd_select = false;
      I.fragment_srt_select = true;
      I.fragment_tsd_select = false;
   }
   if (mali_idvs_bit55)
      ins.opaque[1] |= 1u << (55 - 32);
   uint64_t v;
   memcpy(&v, &ins, sizeof(v));
   return v;
}

static void
emit_run_idvs(struct cs_builder *b, uint32_t flags_override)
{
   cs_flush_loads(b);
   b->req_resource_mask |= CS_IDVS_RES;
   const uint64_t ins = run_idvs_ins(flags_override);
   memcpy(cs_alloc_ins(b), &ins, sizeof(ins));
}

/*
 * The instructions of one draw, collected here and copied into the stream
 * in one piece (one space check instead of one per instruction). Only
 * register moves to the IDVS staging registers and RUN_IDVS go through it,
 * and a move is left out when the register already holds the value
 * (mali_gfx_draw_state.regs).
 */
struct idvs_batch {
   struct mali_gfx_draw_state *d;
   unsigned n;
   uint64_t ins[128];
};

static inline void
batch_move32(struct idvs_batch *e, unsigned reg, uint32_t v)
{
   struct mali_gfx_draw_state *d = e->d;
   if ((d->regs_valid & BITFIELD64_BIT(reg)) && d->regs[reg] == v)
      return;
   d->regs_valid |= BITFIELD64_BIT(reg);
   d->regs[reg] = v;
   struct mali_cs_move32_packed m;
   pan_pack(&m, CS_MOVE32, I) {
      I.destination = reg;
      I.immediate = v;
   }
   assert(e->n < ARRAY_SIZE(e->ins));
   memcpy(&e->ins[e->n++], &m, sizeof(m));
}

static inline void
batch_move64(struct idvs_batch *e, unsigned reg, uint64_t v)
{
   struct mali_gfx_draw_state *d = e->d;
   const uint32_t lo = (uint32_t)v, hi = (uint32_t)(v >> 32);
   const bool lo_ok = (d->regs_valid & BITFIELD64_BIT(reg)) && d->regs[reg] == lo;
   const bool hi_ok = (d->regs_valid & BITFIELD64_BIT(reg + 1)) && d->regs[reg + 1] == hi;
   if (lo_ok && hi_ok)
      return;
   if (!lo_ok && !hi_ok && v < (1ull << 48)) {
      /* MOVE48 zero-extends into the high register. */
      d->regs_valid |= BITFIELD64_BIT(reg) | BITFIELD64_BIT(reg + 1);
      d->regs[reg] = lo;
      d->regs[reg + 1] = hi;
      struct mali_cs_move48_packed m;
      pan_pack(&m, CS_MOVE48, I) {
         I.destination = reg;
         I.immediate = v;
      }
      assert(e->n < ARRAY_SIZE(e->ins));
      memcpy(&e->ins[e->n++], &m, sizeof(m));
      return;
   }
   batch_move32(e, reg, lo);
   batch_move32(e, reg + 1, hi);
}

static void
batch_flush(struct cs_builder *b, struct idvs_batch *e)
{
   if (!e->n)
      return;
   /* A move must not overtake a pending load into its register; waiting
    * for all loads is what the RUN_IDVS after it does anyway. */
   cs_flush_loads(b);
   cs_flush_pending_if(b);
   uint64_t *dst = cs_alloc_ins_block(b, e->n);
   if (dst)
      memcpy(dst, e->ins, e->n * sizeof(e->ins[0]));
   e->n = 0;
}

/* The uniform-buffer copy for the fragment FAU block at fau_va. ARMSX2's
 * case, one buffer whose binding covers every pushed word, appends the
 * shader's prebuilt words (mali_shader::ubo_copy) to the draw's batch
 * after the two address moves; anything else goes through the stream
 * builder. */
static void
emit_ubo_push(struct mali_cmd_buffer *cmd, struct idvs_batch *e, const struct mali_shader *s,
              uint64_t fau_va)
{
   const struct mali_desc_state *desc = &cmd->gfx.desc;
   const uint32_t h = s->desc.dyn_bufs.map[s->fau.ubo_push[0].dyn];
   const uint32_t set = MALI_COPY_DESC_HANDLE_SET(h);
   const uint32_t idx = MALI_COPY_DESC_HANDLE_INDEX(h);
   const struct mali_descriptor_set *ds = desc->sets[set];

   if (!s->ubo_copy.count || !ds || ds->dyn_bufs[idx].range < s->ubo_copy.end ||
       !ds->dyn_bufs[idx].addr ||
       e->n + 2 + s->ubo_copy.count + 32 > ARRAY_SIZE(e->ins)) {
      emit_ubo_push_general(cmd, s, fau_va);
      return;
   }

   const uint64_t addrs[2] = {ds->dyn_bufs[idx].addr + desc->dyn_offsets[set][idx], fau_va};
   const unsigned regs[2] = {MALI_UBO_COPY_SRC, MALI_UBO_COPY_DST};
   for (unsigned i = 0; i < 2; i++) {
      struct mali_cs_move48_packed m;
      pan_pack(&m, CS_MOVE48, I) {
         I.destination = regs[i];
         I.immediate = addrs[i];
      }
      memcpy(&e->ins[e->n++], &m, sizeof(m));
   }
   memcpy(&e->ins[e->n], s->ubo_copy.ins, s->ubo_copy.count * sizeof(uint64_t));
   e->n += s->ubo_copy.count;
   cmd->gfx.render.ubo_stores = true;
}

struct draw_info {
   uint32_t count;
   uint32_t instance_count;
   uint32_t first_index;       /* indexed only */
   int32_t vertex_offset;      /* firstVertex or vertexOffset */
   uint32_t first_instance;
   bool indexed;
};

static bool
prepare_draw(struct mali_cmd_buffer *cmd, const struct draw_info *di, struct idvs_batch *e)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_graphics_pipeline *p = d->pipeline;
   const struct mali_gfx_bind *bd = &p->bind;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   collect_dynamic_dirty(cmd);

   if (!mali_cmd_render_tiler(cmd))
      return false;

   const struct mali_gfx_baked *dbk = NULL;
   if (unlikely(bd->dynamic)) {
      if (d->dirty & MALI_GFX_DIRTY_PIPELINE) {
         const struct mali_gfx_pack_input in = {
            .dyn = dyn,
            .rp = &p->rp,
            .vs = p->vs,
            .fs = p->fs,
         };
         d->dyn_baked.uses_dynamic_state = true;
         mali_gfx_pack_state(&in, &d->dyn_baked);
      }
      dbk = &d->dyn_baked;
   }

   if (d->first_vertex != di->vertex_offset && bd->vs_first_vertex)
      d->dirty |= MALI_GFX_DIRTY_VS_FAU;
   if (d->base_instance != di->first_instance) {
      d->dirty |= MALI_GFX_DIRTY_VS_SRT;
      if (bd->vs_base_instance)
         d->dirty |= MALI_GFX_DIRTY_VS_FAU;
   }
   d->first_vertex = di->vertex_offset;
   d->base_instance = di->first_instance;

   /* Descriptor sets bound since a stage's table was built. */
   if (d->vs_sets_dirty & bd->vs_sets)
      d->dirty |= MALI_GFX_DIRTY_VS_SRT;
   if (d->fs_sets_dirty & bd->fs_sets)
      d->dirty |= MALI_GFX_DIRTY_FS_SRT;
   /* A fragment shader with uniform-buffer words in FAU needs them again
    * whenever its sets or dynamic offsets change. */
   if ((d->dirty & MALI_GFX_DIRTY_FS_SRT) && p->fs && p->fs->fau.ubo_push_count)
      d->dirty |= MALI_GFX_DIRTY_FS_FAU;

   r->tls_size = MAX2(r->tls_size, bd->tls_size);

   /* Blend first: the fragment FAU carries its internal words. */
   if (d->dirty & MALI_GFX_DIRTY_BLEND) {
      uint32_t n = 0;
      uint64_t bl = build_blend(cmd, &n);
      if (!bl)
         return false;
      batch_move64(e, IDVS(BLEND_DESC), bl | n);
      d->dirty |= MALI_GFX_DIRTY_FS_FAU;
   }

   struct mali_graphics_sysvals sv;
   if (d->dirty & (MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_FS_FAU))
      fill_sysvals(cmd, &sv);

   if (d->dirty & MALI_GFX_DIRTY_VS_FAU) {
      uint64_t fau = mali_cmd_gfx_fau(cmd, p->vs, &sv, cmd->push_constants,
                                      sizeof(cmd->push_constants));
      if (p->vs->fau.total_count && !fau)
         return false;
      batch_move64(e, IDVS(FAU_0), fau);
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_FAU) {
      uint64_t fau = p->fs ? mali_cmd_gfx_fau(cmd, p->fs, &sv, cmd->push_constants,
                                              sizeof(cmd->push_constants)) : 0;
      if (p->fs && p->fs->fau.total_count && !fau)
         return false;
      /* Straight into the stream, ahead of the batched moves and RUN_IDVS. */
      if (p->fs && p->fs->fau.ubo_push_count)
         emit_ubo_push(cmd, e, p->fs, fau & BITFIELD64_MASK(56));
      batch_move64(e, IDVS(FAU_2), fau);
   }
   if (d->dirty & MALI_GFX_DIRTY_VS_SRT) {
      uint64_t srt = build_vs_srt(cmd);
      if (!srt)
         return false;
      batch_move64(e, IDVS(SRT_0), srt);
      d->vs_srt_key = bd->vs_srt_key;
      d->vs_sets_dirty = 0;
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_SRT) {
      uint64_t srt = p->fs ? build_fs_srt(cmd) : 0;
      if (p->fs && !srt)
         return false;
      batch_move64(e, IDVS(SRT_2), srt);
      d->fs_srt_key = bd->fs_srt_key;
      d->fs_sets_dirty = 0;
   }
   if (d->dirty & MALI_GFX_DIRTY_ZSD) {
      uint64_t zsd = bd->zsd;
      if (!zsd) {
         struct mali_ptr z = mali_cmd_alloc(cmd, sizeof(dbk->zsd), 32);
         if (!z.cpu)
            return false;
         memcpy(z.cpu, dbk->zsd, sizeof(dbk->zsd));
         zsd = z.gpu;
      }
      batch_move64(e, IDVS(ZSD), zsd);
   }
   if (d->dirty & MALI_GFX_DIRTY_PIPELINE) {
      bool lines;
      if (unlikely(dbk)) {
         batch_move64(e, IDVS(SPD_0), dbk->vs_pos_spd);
         batch_move64(e, IDVS(SPD_1), dbk->vs_var_spd);
         batch_move64(e, IDVS(SPD_2), dbk->fs_spd);
         batch_move32(e, IDVS(TILER_FLAGS), dbk->tiler_flags);
         batch_move32(e, IDVS(DCD0), dbk->dcd0[0]);
         batch_move32(e, IDVS(DCD1), dbk->dcd1);
         batch_move32(e, IDVS(DCD2), dbk->dcd2);
         lines = u_reduced_prim(dbk->prim) == MESA_PRIM_LINES;
      } else {
         batch_move64(e, IDVS(SPD_0), bd->spd[0]);
         batch_move64(e, IDVS(SPD_1), bd->spd[1]);
         batch_move64(e, IDVS(SPD_2), bd->spd[2]);
         batch_move32(e, IDVS(TILER_FLAGS), bd->tiler_flags);
         batch_move32(e, IDVS(DCD0), bd->dcd0);
         batch_move32(e, IDVS(DCD1), bd->dcd1);
         batch_move32(e, IDVS(DCD2), bd->dcd2);
         lines = bd->lines;
      }
      batch_move32(e, IDVS(TILER_FLAGS2), 0);
      batch_move32(e, IDVS(VARY_SIZE), bd->vary_size);
      /* Point size 1.0 unless the shader writes it; line width. */
      batch_move32(e, IDVS(PRIMITIVE_SIZE), fui(lines ? dyn->rs.line.width : 1.0f));
      batch_move32(e, IDVS(GLOBAL_ATTRIBUTE_OFFSET), 0);
      batch_move64(e, IDVS(OQ), 0);
   }
   if (d->dirty & MALI_GFX_DIRTY_VIEWPORT) {
      const VkViewport *vp = &dyn->vp.viewports[0];
      batch_move64(e, IDVS(SCISSOR_BOX), scissor_box(cmd));
      batch_move32(e, IDVS(LOW_DEPTH_CLAMP), fui(MIN2(vp->minDepth, vp->maxDepth)));
      batch_move32(e, IDVS(HIGH_DEPTH_CLAMP), fui(MAX2(vp->minDepth, vp->maxDepth)));
   }
   if (d->dirty & MALI_GFX_DIRTY_PASS) {
      batch_move64(e, IDVS(TILER_CTX), r->tiler);
      batch_move64(e, IDVS(TSD_0), r->tsd);
   }
   if (di->indexed && (d->dirty & MALI_GFX_DIRTY_INDEX)) {
      batch_move64(e, IDVS(INDEX_BUFFER), d->ib.addr);
      batch_move32(e, IDVS(INDEX_BUFFER_SIZE), d->ib.size);
      d->dirty &= ~MALI_GFX_DIRTY_INDEX;
   }

   d->dirty &= MALI_GFX_DIRTY_INDEX;
   return true;
}

static void
draw(struct mali_cmd_buffer *cmd, const struct draw_info *di)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;

   if (!di->count || !di->instance_count)
      return;
   if (!r->active || !d->pipeline || !d->pipeline->vs) {
      vk_logw(VK_LOG_OBJS(cmd), "libmali: draw outside a render pass or without a pipeline");
      return;
   }
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   struct idvs_batch e;
   e.d = d;
   e.n = 0;
   if (!prepare_draw(cmd, di, &e)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);
   batch_move32(&e, IDVS(INDEX_COUNT), di->count);
   batch_move32(&e, IDVS(INSTANCE_COUNT), di->instance_count);
   batch_move32(&e, IDVS(INDEX_OFFSET), di->indexed ? di->first_index : 0);
   batch_move32(&e, IDVS(VERTEX_OFFSET), (uint32_t)di->vertex_offset);
   /* Zero-based instance IDs; firstInstance reaches the shader as a
    * system value (panvk). */
   batch_move32(&e, IDVS(INSTANCE_OFFSET), 0);

   struct mali_primitive_flags_packed ovr;
   pan_pack_nodefaults(&ovr, PRIMITIVE_FLAGS, cfg) {
      cfg.index_type = di->indexed ? index_type(d->ib.index_size) : MALI_INDEX_TYPE_NONE;
   }

   /* Per-draw timing ("draws" mode). */
   uint32_t mh = 0;
   if (unlikely(cmd->measure) && cmd->measure->draws) {
      batch_flush(b, &e);
      mh = mali_measure_begin(cmd, MALI_MEASURE_DRAW, MALI_SUBQUEUE_VERTEX_TILER,
                              cmd->draws);
      mali_measure_info(cmd, mh, di->count, di->instance_count, di->indexed,
                        cmd->passes - 1);
      uint64_t key = 0;
      if (d->pipeline->fs)
         memcpy(&key, d->pipeline->fs->key, sizeof(key));
      mali_measure_shader(cmd, mh, key);
      mali_measure_extra(cmd, mh, d->blend_shader_mask);
   }

   /* One RUN_IDVS per tiler context: a layered pass has one per 8
    * layers, and the draw's primitives may go to any of them. */
   if (r->td_count > 1) {
      batch_flush(b, &e);
      struct cs_index ctx = cs_reg64(b, IDVS(TILER_CTX));
      for (uint32_t i = 0; i < r->td_count; i++) {
         if (i)
            cs_add_imm64(b, ctx, ctx, pan_size(TILER_CONTEXT));
         emit_run_idvs(b, ovr.opaque[0]);
      }
      cs_move64_to(b, ctx, r->tiler);
   } else {
      b->req_resource_mask |= CS_IDVS_RES;
      assert(e.n < ARRAY_SIZE(e.ins));
      e.ins[e.n++] = run_idvs_ins(ovr.opaque[0]);
      batch_flush(b, &e);
   }
   r->draws++;
   cmd->draws++;
   if (unlikely(mh))
      mali_measure_end(cmd, mh);
}

VKAPI_ATTR void VKAPI_CALL
mali_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
             uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const struct draw_info di = {
      .count = vertexCount,
      .instance_count = instanceCount,
      .vertex_offset = (int32_t)firstVertex,
      .first_instance = firstInstance,
   };
   draw(cmd, &di);
}

VKAPI_ATTR void VKAPI_CALL
mali_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                    uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset,
                    uint32_t firstInstance)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const struct draw_info di = {
      .count = indexCount,
      .instance_count = instanceCount,
      .first_index = firstIndex,
      .vertex_offset = vertexOffset,
      .first_instance = firstInstance,
      .indexed = true,
   };
   draw(cmd, &di);
}

/* ---------------------------------------------------------------------- */
/* Full-screen draws                                                       */

void
mali_cmd_run_fullscreen(struct mali_cmd_buffer *cmd, uint64_t dcd, const VkRect2D *rect,
                        uint32_t base_layer, uint32_t layer_count)
{
   struct mali_render_state *r = &cmd->gfx.render;
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);

   if (!mali_cmd_render_tiler(cmd)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   int minx = MAX2(rect->offset.x, 0), miny = MAX2(rect->offset.y, 0);
   int maxx = MIN2(rect->offset.x + (int)rect->extent.width, (int)r->desc.width) - 1;
   int maxy = MIN2(rect->offset.y + (int)rect->extent.height, (int)r->desc.height) - 1;
   if (maxx < minx || maxy < miny)
      return;
   struct mali_scissor_packed sc;
   pan_pack(&sc, SCISSOR, cfg) {
      cfg.scissor_minimum_x = minx;
      cfg.scissor_minimum_y = miny;
      cfg.scissor_maximum_x = maxx;
      cfg.scissor_maximum_y = maxy;
   }
   cs_move64_to(b, cs_reg64(b, IDVS(SCISSOR_BOX)), sc.opaque[0] | (uint64_t)sc.opaque[1] << 32);
   cs_move32_to(b, cs_reg32(b, IDVS(TILER_FLAGS2)), 0);

   struct cs_index dcd_reg = mali_cs_scratch_reg64(b, 0);
   cs_move64_to(b, dcd_reg, dcd);

   /* RUN_FULLSCREEN draws into the layers its tiler flags' view mask
    * selects inside the current tiler context (panvk cmd_run_fullscreen). */
   const uint32_t first_td = base_layer / MALI_LAYERS_PER_TILER_CTX;
   const uint32_t last_td = (base_layer + layer_count - 1) / MALI_LAYERS_PER_TILER_CTX;
   for (uint32_t td = first_td; td <= last_td && td < r->td_count; td++) {
      const uint32_t td_base = td * MALI_LAYERS_PER_TILER_CTX;
      const uint32_t first = MAX2(base_layer, td_base);
      const uint32_t last = MIN2(base_layer + layer_count, td_base + MALI_LAYERS_PER_TILER_CTX);
      struct mali_primitive_flags_packed tf;
      pan_pack(&tf, PRIMITIVE_FLAGS, cfg) {
         cfg.low_depth_cull = false;
         cfg.high_depth_cull = false;
         cfg.view_mask = BITFIELD_RANGE(first - td_base, last - first);
      }
      cs_move64_to(b, cs_reg64(b, IDVS(TILER_CTX)), r->tiler + td * pan_size(TILER_CONTEXT));
      cs_move32_to(b, cs_reg32(b, IDVS(TILER_FLAGS)), tf.opaque[0]);
      cs_run_fullscreen(b, 0, dcd_reg);
   }

   r->draws++;
   cmd->gfx.draw.dirty |= MALI_GFX_DIRTY_PIPELINE | MALI_GFX_DIRTY_VIEWPORT |
                          MALI_GFX_DIRTY_PASS;
   /* The scissor, tiler flags and tiler context registers changed behind
    * the draws' record of them. */
   cmd->gfx.draw.regs_valid = 0;
}

/* A primitive barrier over the whole framebuffer: later fragments of the
 * pass wait for earlier ones in each tile (panvk cmd_fb_barrier). */
void
mali_cmd_fb_barrier(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_ptr zsd = mali_cmd_alloc(cmd, pan_size(DEPTH_STENCIL), 32);
   struct mali_ptr dcd = mali_cmd_alloc(cmd, pan_size(DRAW), 64);
   if (!zsd.cpu || !dcd.cpu)
      return;

   pan_cast_and_pack(zsd.cpu, DEPTH_STENCIL, cfg) {
      cfg.stencil_test_enable = false;
      cfg.depth_write_enable = false;
      cfg.depth_function = MALI_FUNC_ALWAYS;
   }
   pan_cast_and_pack(dcd.cpu, DRAW, cfg) {
      cfg.flags_0.allow_forward_pixel_to_kill = false;
      cfg.flags_0.allow_forward_pixel_to_be_killed = false;
      cfg.flags_0.primitive_barrier = true;
      cfg.flags_0.occlusion_query = MALI_OCCLUSION_MODE_DISABLED;
      cfg.flags_2.read_mask = 0;
      cfg.flags_2.write_mask = 0;
      cfg.flags_2.no_shader_depth_read = true;
      cfg.flags_2.no_shader_stencil_read = true;
      cfg.depth_stencil = zsd.gpu;
   }
   const VkRect2D all = {{0, 0}, {r->desc.width, r->desc.height}};
   mali_cmd_run_fullscreen(cmd, dcd.gpu, &all, 0, MAX2(r->desc.layer_count, 1));
}

/* ---------------------------------------------------------------------- */
/* vkCmdClearAttachments                                                   */

static enum mali_meta_fs_type
fs_type(enum pipe_format f)
{
   if (util_format_is_pure_uint(f))
      return MALI_META_FS_UINT;
   if (util_format_is_pure_sint(f))
      return MALI_META_FS_SINT;
   return MALI_META_FS_FLOAT;
}

/*
 * A full-screen draw per rectangle with a shader that writes the clear
 * values (the blob's path; panvk goes through vk_meta and ends in the
 * same RUN_FULLSCREEN). Depth and stencil are written by the shader (ZS
 * emit), so no depth-bias trick is needed.
 */
VKAPI_ATTR void VKAPI_CALL
mali_CmdClearAttachments(VkCommandBuffer commandBuffer, uint32_t attachmentCount,
                         const VkClearAttachment *pAttachments, uint32_t rectCount,
                         const VkClearRect *pRects)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   struct mali_render_state *r = &cmd->gfx.render;

   if (!r->active || !rectCount || vk_command_buffer_has_error(&cmd->vk))
      return;

   struct mali_meta_fs_key key = {0};
   struct mali_meta_fs_push push = {0};
   bool any = false;

   for (uint32_t i = 0; i < attachmentCount; i++) {
      const VkClearAttachment *a = &pAttachments[i];
      if (a->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
         const uint32_t rt = a->colorAttachment;
         if (rt == VK_ATTACHMENT_UNUSED || rt >= r->desc.rt_count || !r->desc.rt[rt].image)
            continue;
         key.rt_op[rt] = MALI_META_FS_CLEAR;
         key.rt_type[rt] = fs_type(r->desc.rt[rt].format);
         memcpy(push.clear_color[rt], &a->clearValue.color, 16);
         any = true;
      }
      if ((a->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) && r->desc.z.image) {
         key.z_op = MALI_META_FS_CLEAR;
         push.clear_depth = a->clearValue.depthStencil.depth;
         any = true;
      }
      if ((a->aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) && r->desc.s.image) {
         key.s_op = MALI_META_FS_CLEAR;
         push.clear_stencil = a->clearValue.depthStencil.stencil;
         any = true;
      }
   }
   if (!any)
      return;

   if (!mali_cmd_render_tiler(cmd)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   const struct mali_shader *fs = mali_meta_fs_get(cmd, &key);
   struct mali_ptr dcd = mali_cmd_alloc(cmd, pan_size(DRAW), 64);
   if (!fs || !dcd.cpu ||
       !mali_meta_fs_dcd(cmd, fs, &key, &push, NULL, 0, NULL, false, dcd.cpu)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   for (uint32_t i = 0; i < rectCount; i++) {
      mali_cmd_run_fullscreen(cmd, dcd.gpu, &pRects[i].rect, pRects[i].baseArrayLayer,
                              pRects[i].layerCount);
   }
}
