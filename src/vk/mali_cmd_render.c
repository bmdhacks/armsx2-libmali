/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Render passes: vkCmdBeginRendering / vkCmdEndRendering (the runtime
 * turns VkRenderPass objects into these), the framebuffer and tiler
 * descriptors, the fragment jobs and the tiler heap protocol.
 *
 * Framebuffer and tiler descriptor shapes, scoreboards and the
 * vertex/tiler -> fragment synchronization follow panvk v11
 * (csf/panvk_vX_cmd_draw.c: get_tiler_desc, get_fb_descs, flush_tiling,
 * issue_fragment_jobs; lib/pan_fb.c for the descriptor fields). Where the
 * blob and panvk differ and both are valid, the blob's choice is taken
 * when it is simpler or is what this kernel was tested with:
 *
 *  - no tiler out-of-memory exception handler and no incremental
 *    rendering (the queue group has none);
 *  - the tiler heap and geometry buffer belong to the queue; the device
 *    has one queue, so a ONE_TIME_SUBMIT command buffer gets them packed
 *    into its tiler contexts on the CPU, and any other command buffer
 *    stores them (and zeroes the words the tiler writes) from the stream
 *    at run time, as the blob does for every pass, so it can run again;
 *  - exactly three heap-counter steps per pass that tiles, in order:
 *    HEAP_OPERATION{Vertex/Tiler Started} before the first draw, a
 *    deferred HEAP_OPERATION{Vertex/Tiler Completed} after FINISH_TILING,
 *    and FINISH_FRAGMENT with the increment bit on the last tiler context
 *    (the blob; panvk uses a separate HEAP_OPERATION{Fragment Completed}
 *    for more than one tiler context). kbase's tiler out-of-memory handler
 *    rejects the event unless frag_end <= vt_end < vt_start. A pass that
 *    draws nothing does not touch the counters at all.
 *
 * Transaction elimination (CRC) follows panvk v11's seed scheme; see
 * "CRC" below. AFBC targets get panvk's AFBC render-target and
 * depth/stencil descriptors, and the pass's bounding box grows to whole
 * superblocks with the border loaded (pan_align_fb_tiling_area). MSAA and
 * multiview are not used; the pass refuses more samples than one with a
 * log line.
 */

#include "mali_cmd_buffer.h"
#include "mali_measure.h"

#include <string.h>

#include "drm-uapi/drm_fourcc.h"
#include "util/format/u_format.h"
#include "util/format_srgb.h"
#include "util/rounding.h"
#include "util/u_math.h"
#include "util/u_pack_color.h"
#include "vk_format.h"
#include "vk_log.h"

#include "model/pan_model.h"
#include "pan_format.h"

#include "mali_image.h"
#include "mali_queue.h"
#include "mali_vk.h"

#if PAN_ARCH != 11
#error "mali_cmd_render.c is written for arch v11"
#endif

/* Frame shader DCD slots (panvk: colour preload 0, depth/stencil 1). */
#define FS_DCD_COLOR 0
#define FS_DCD_ZS    1
#define FS_DCD_COUNT 3

/* ---------------------------------------------------------------------- */
/* Formats                                                                 */

static enum mali_zs_format
zs_write_format(enum pipe_format f)
{
   switch (f) {
   case PIPE_FORMAT_Z16_UNORM: return MALI_ZS_FORMAT_D16;
   case PIPE_FORMAT_Z24X8_UNORM: return MALI_ZS_FORMAT_D24X8;
   case PIPE_FORMAT_Z24_UNORM_S8_UINT: return MALI_ZS_FORMAT_D24S8;
   default: return MALI_ZS_FORMAT_D32;
   }
}

static enum mali_z_internal_format
z_internal_format(enum pipe_format f)
{
   switch (f) {
   case PIPE_FORMAT_Z16_UNORM: return MALI_Z_INTERNAL_FORMAT_D16;
   case PIPE_FORMAT_Z24X8_UNORM:
   case PIPE_FORMAT_Z24_UNORM_S8_UINT: return MALI_Z_INTERNAL_FORMAT_D24;
   default: return MALI_Z_INTERNAL_FORMAT_D32;
   }
}

/* Bytes a colour target takes per pixel in the tile buffer. */
static unsigned
tib_bytes(enum pipe_format f)
{
   const struct pan_blendable_format *bf = GENX(pan_blendable_format_from_pipe_format)(f);
   return pan_format_tib_size(f, bf->internal);
}

static enum mali_color_format
raw_writeback_format(unsigned bits)
{
   switch (bits) {
   case 8: return MALI_COLOR_FORMAT_RAW8;
   case 16: return MALI_COLOR_FORMAT_RAW16;
   case 24: return MALI_COLOR_FORMAT_RAW24;
   case 32: return MALI_COLOR_FORMAT_RAW32;
   case 48: return MALI_COLOR_FORMAT_RAW48;
   case 64: return MALI_COLOR_FORMAT_RAW64;
   case 96: return MALI_COLOR_FORMAT_RAW96;
   default: return MALI_COLOR_FORMAT_RAW128;
   }
}

/* Writeback format, tile-buffer format and swizzle of a render target
 * (pan_desc.c get_rt_formats: the render swizzle is the inverse of the
 * texture one). */
static void
rt_formats(enum pipe_format f, uint32_t *writeback, uint32_t *internal, uint32_t *swizzle)
{
   const struct util_format_description *desc = util_format_description(f);
   const struct pan_blendable_format *bf = GENX(pan_blendable_format_from_pipe_format)(f);
   unsigned char sw[4] = {PIPE_SWIZZLE_X, PIPE_SWIZZLE_Y, PIPE_SWIZZLE_Z, PIPE_SWIZZLE_W};

   if (bf->internal) {
      *internal = bf->internal;
      *writeback = bf->writeback;
      for (unsigned c = 0; c < 4; c++) {
         if (desc->swizzle[c] < 4)
            sw[desc->swizzle[c]] = c;
      }
   } else {
      /* Raw formats, the internal size a power of two from RAW8. */
      unsigned bits = desc->block.bits;
      *internal = MALI_COLOR_BUFFER_INTERNAL_FORMAT_RAW8 + util_logbase2_ceil(bits) - 3;
      *writeback = raw_writeback_format(bits);
   }
   /* Mali channel numbers are the pipe swizzle values. */
   *swizzle = sw[0] | sw[1] << 3 | sw[2] << 6 | sw[3] << 9;
}

/* The tile-buffer layout of a clear colour (pan_clear.c). */
struct tib_layout {
   unsigned int_r, frac_r, int_g, frac_g, int_b, frac_b, int_a, frac_a;
};

static const struct tib_layout tib_layouts[] = {
   [MALI_COLOR_BUFFER_INTERNAL_FORMAT_R8G8B8A8] = {8, 0, 8, 0, 8, 0, 8, 0},
   [MALI_COLOR_BUFFER_INTERNAL_FORMAT_R10G10B10A2] = {10, 0, 10, 0, 10, 0, 2, 0},
   [MALI_COLOR_BUFFER_INTERNAL_FORMAT_R8G8B8A2] = {8, 2, 8, 2, 8, 2, 2, 0},
   [MALI_COLOR_BUFFER_INTERNAL_FORMAT_R4G4B4A4] = {4, 4, 4, 4, 4, 4, 4, 4},
   [MALI_COLOR_BUFFER_INTERNAL_FORMAT_R5G6B5A0] = {5, 5, 6, 4, 5, 5, 0, 2},
   [MALI_COLOR_BUFFER_INTERNAL_FORMAT_R5G5B5A1] = {5, 5, 5, 5, 5, 5, 1, 1},
};

static uint32_t
to_fixed(float f, unsigned bits_int, unsigned bits_frac)
{
   uint32_t m = (1u << bits_int) - 1;
   return (uint32_t)_mesa_roundevenf(f * (float)m) << bits_frac;
}

/* A clear value in the render target's tile-buffer format (pan_pack_color,
 * not dithered). */
static void
pack_clear_color(enum pipe_format f, const VkClearColorValue *v, uint32_t out[4])
{
   const struct pan_blendable_format *bf = GENX(pan_blendable_format_from_pipe_format)(f);
   const unsigned size = util_format_get_blocksize(f);

   if (util_format_is_float(f) || !bf->internal || util_format_is_pure_integer(f)) {
      union util_color c = {0};
      if (util_format_is_pure_integer(f)) {
         util_format_pack_rgba(f, &c, v->uint32, 1);
      } else {
         util_pack_color(v->float32, f, &c);
      }
      if (size == 1) {
         uint32_t b = c.ui[0] & 0xff;
         b |= b << 8;
         b |= b << 16;
         out[0] = out[1] = out[2] = out[3] = b;
      } else if (size == 2) {
         uint32_t w = c.ui[0] & 0xffff;
         out[0] = out[1] = out[2] = out[3] = w | w << 16;
      } else if (size <= 4) {
         out[0] = out[1] = out[2] = out[3] = c.ui[0];
      } else if (size <= 8) {
         out[0] = out[2] = c.ui[0];
         out[1] = out[3] = c.ui[1];
      } else {
         memcpy(out, c.ui, 16);
      }
      return;
   }

   float r = SATURATE(v->float32[0]), g = SATURATE(v->float32[1]);
   float b = SATURATE(v->float32[2]), a = SATURATE(v->float32[3]);
   if (!util_format_has_alpha(f))
      a = 1.0f;
   if (util_format_is_srgb(f)) {
      r = util_format_linear_to_srgb_float(r);
      g = util_format_linear_to_srgb_float(g);
      b = util_format_linear_to_srgb_float(b);
   }
   assert(bf->internal < ARRAY_SIZE(tib_layouts));
   const struct tib_layout l = tib_layouts[bf->internal];
   const unsigned cr = l.int_r + l.frac_r;
   const unsigned cg = cr + l.int_g + l.frac_g;
   const unsigned cb = cg + l.int_b + l.frac_b;
   uint32_t w = to_fixed(r, l.int_r, l.frac_r) | to_fixed(g, l.int_g, l.frac_g) << cr |
                to_fixed(b, l.int_b, l.frac_b) << cg | to_fixed(a, l.int_a, l.frac_a) << cb;
   out[0] = out[1] = out[2] = out[3] = w;
}

void
mali_pack_opaque_blend(enum pipe_format format, unsigned rt, uint32_t out[4])
{
   pan_cast_and_pack(out, BLEND, cfg) {
      cfg.round_to_fb_precision = true;
      cfg.srgb = util_format_is_srgb(format);
      cfg.internal.mode = MALI_BLEND_MODE_OPAQUE;
      cfg.equation.rgb.a = MALI_BLEND_OPERAND_A_SRC;
      cfg.equation.rgb.b = MALI_BLEND_OPERAND_B_SRC;
      cfg.equation.rgb.c = MALI_BLEND_OPERAND_C_ZERO;
      cfg.equation.alpha.a = MALI_BLEND_OPERAND_A_SRC;
      cfg.equation.alpha.b = MALI_BLEND_OPERAND_B_SRC;
      cfg.equation.alpha.c = MALI_BLEND_OPERAND_C_ZERO;
      cfg.equation.color_mask = 0xf;
      cfg.internal.fixed_function.num_comps = 4;
      cfg.internal.fixed_function.conversion.memory_format =
         GENX(pan_dithered_format_from_pipe_format)(format, false);
      cfg.internal.fixed_function.rt = rt;
   }
}

/* ---------------------------------------------------------------------- */
/* Begin                                                                   */

/* Width and height of a tile of `pixels` (a power of two). */
static void
tile_dims(uint32_t pixels, uint32_t *w, uint32_t *h)
{
   unsigned log2 = util_logbase2(pixels);
   *w = 1u << DIV_ROUND_UP(log2, 2);
   *h = pixels / *w;
}

/* pan_select_fb_tile_size for single-sampled framebuffers: the largest
 * power-of-two tile whose colour and depth fit half the tile buffer (the
 * other half lets the GPU pipeline tiles). */
static void
select_tile_size(struct mali_cmd_buffer *cmd, struct mali_render_state *r)
{
   const struct mali_physical_device *pdev = mali_device_physical(cmd->dev);
   const struct pan_model *model = pan_get_model(pdev->props.gpu_id, 0);
   const uint32_t rt_budget = model ? model->tilebuffer.color_size / 2 : 16384;
   const uint32_t z_budget = model ? model->tilebuffer.z_size / 2 : 8192;

   uint32_t rt_bytes = 0;
   for (unsigned i = 0; i < r->rt_count; i++) {
      if (r->desc.rt[i].image)
         rt_bytes += tib_bytes(r->desc.rt[i].format);
   }
   rt_bytes = MAX2(rt_bytes, 1);

   uint32_t tile = MIN2(rt_budget >> util_logbase2_ceil(rt_bytes), z_budget >> 2);
   if (tile < 16)
      tile *= 2;
   tile = MIN2(tile, 32 * 32);   /* v10/v11 effective tile size limit */
   r->tile_size = tile;
   r->cbuf_alloc = ALIGN_POT(rt_bytes * tile, 1024);

   uint32_t off = 0;
   for (unsigned i = 0; i < r->rt_count; i++) {
      r->rt_offset[i] = off;
      if (r->desc.rt[i].image)
         off += tib_bytes(r->desc.rt[i].format) * tile;
   }
}

static bool
target_needs_border_load(const struct mali_fb_target *t)
{
   return t->image && t->load != MALI_ATT_LOAD_LOAD;
}

static bool
target_is_afbc(const struct mali_fb_target *t)
{
   return t->image && mali_image_is_afbc(t->image);
}

/* AFBC superblock size: the GPU writes whole superblocks of an AFBC target. */
#define AFBC_BLOCK 16

void
mali_cmd_render_begin(struct mali_cmd_buffer *cmd, const struct mali_render_desc *desc)
{
   struct mali_render_state *r = &cmd->gfx.render;

   memset(r, 0, sizeof(*r));
   r->active = true;
   r->desc = *desc;
   r->rt_count = MAX2(desc->rt_count, 1);
   r->crc_rt = -1;
   cmd->passes++;

   select_tile_size(cmd, r);

   /* Bounding box: the render area inside the framebuffer. */
   const VkRect2D *a = &desc->area;
   r->minx = MIN2((uint32_t)MAX2(a->offset.x, 0), desc->width - 1);
   r->miny = MIN2((uint32_t)MAX2(a->offset.y, 0), desc->height - 1);
   r->maxx = MIN2((uint32_t)MAX2(a->offset.x, 0) + a->extent.width, desc->width) - 1;
   r->maxy = MIN2((uint32_t)MAX2(a->offset.y, 0) + a->extent.height, desc->height) - 1;
   if (r->maxx < r->minx)
      r->maxx = r->minx;
   if (r->maxy < r->miny)
      r->maxy = r->miny;

   struct mali_fb_target *targets[10];
   unsigned n = 0;
   for (unsigned i = 0; i < desc->rt_count; i++)
      targets[n++] = &r->desc.rt[i];
   targets[n++] = &r->desc.z;
   targets[n++] = &r->desc.s;

   /* The fragment job writes back whole tiles; pixels of a partly covered
    * tile outside the render area must keep their contents, so a target
    * that is not loaded anyway gets loaded there (panvk's border load).
    * An AFBC target is written in whole superblocks, so the bounding box
    * grows to them and a tile smaller than a superblock counts as one. */
   uint32_t tw, th;
   tile_dims(r->tile_size, &tw, &th);
   bool afbc_store = false;
   for (unsigned i = 0; i < n; i++)
      afbc_store |= target_is_afbc(targets[i]) && targets[i]->store;
   if (afbc_store) {
      tw = MAX2(tw, AFBC_BLOCK);
      th = MAX2(th, AFBC_BLOCK);
   }
   r->partial_tiles = (r->minx % tw) || (r->miny % th) ||
                      ((r->maxx + 1) % tw && r->maxx + 1 != desc->width) ||
                      ((r->maxy + 1) % th && r->maxy + 1 != desc->height);
   if (afbc_store) {
      r->minx = ROUND_DOWN_TO(r->minx, AFBC_BLOCK);
      r->miny = ROUND_DOWN_TO(r->miny, AFBC_BLOCK);
      r->maxx = MIN2(ALIGN_POT(r->maxx + 1, AFBC_BLOCK), desc->width) - 1;
      r->maxy = MIN2(ALIGN_POT(r->maxy + 1, AFBC_BLOCK), desc->height) - 1;
   }

   bool any_preload = false;
   for (unsigned i = 0; i < n; i++) {
      struct mali_fb_target *t = targets[i];
      if (!t->image)
         continue;
      t->preload = t->load == MALI_ATT_LOAD_LOAD ||
                   (r->partial_tiles && target_needs_border_load(t));
      any_preload |= t->preload;
   }

   if (any_preload) {
      /* Loads go through the texture unit: invalidate its caches after
       * earlier attachment writes (panvk cmd_init_render_state; the blob
       * sets r49 for the same invalidate). */
      const VkMemoryBarrier2 mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
         .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
         .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      };
      const VkDependencyInfo dep = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &mb,
      };
      struct mali_cs_deps deps = {0};
      mali_cmd_add_deps(cmd, &dep, &deps);
      mali_cmd_emit_barrier(cmd, &deps);
   }

   cmd->gfx.draw.dirty = MALI_GFX_DIRTY_ALL;
}

/* ---------------------------------------------------------------------- */
/* Tiler contexts                                                          */

static bool
alloc_tsd(struct mali_cmd_buffer *cmd, struct mali_render_state *r)
{
   if (r->tsd)
      return true;
   struct mali_ptr p = mali_cmd_alloc(cmd, pan_size(LOCAL_STORAGE), 64);
   if (!p.cpu)
      return false;
   r->tsd = p.gpu;
   r->tsd_cpu = p.cpu;
   return true;
}

/* v11 cs_iter_sb_update with nothing inside (panvk cs_next_iter_sb). */
static void
next_iter_sb(struct mali_cmd_buffer *cmd, struct cs_builder *b)
{
   const struct mali_csf_device *csf = cmd->dev->csf;
   struct cs_index next_sb = mali_cs_scratch_reg32(b, 0);
   struct cs_index sb_mask = mali_cs_scratch_reg32(b, 1);

   cs_next_sb_entry(b, next_sb, MALI_CS_SCOREBOARD_TYPE_ENDPOINT,
                    MALI_CS_NEXT_SB_ENTRY_FORMAT_INDEX);
   cs_move32_to(b, sb_mask, 0);
   cs_bit_set32(b, sb_mask, sb_mask, next_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_WAIT, sb_mask);
   cs_move32_to(b, sb_mask, csf->sb.all_iters_mask);
   cs_bit_clear32(b, sb_mask, sb_mask, next_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_STREAM, sb_mask);
}

bool
mali_cmd_render_tiler(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   assert(r->active);
   if (r->tiler)
      return true;
   if (!alloc_tsd(cmd, r))
      return false;

   const uint32_t layers = MAX2(r->desc.layer_count, 1);
   const uint32_t td_count = DIV_ROUND_UP(layers, MALI_LAYERS_PER_TILER_CTX);
   struct mali_ptr p = mali_cmd_alloc(cmd, td_count * pan_size(TILER_CONTEXT), 64);
   if (!p.cpu)
      return false;

   /* Words 2-4 on the CPU (as the blob). Polygon list, completed
    * top/bottom and the private state start at zero; the tiler writes
    * them. A command buffer that runs once gets the queue's heap and
    * geometry buffer here too and needs no stream work; one that may run
    * again has the stream below store them and zero the rest before
    * every run. */
   STATIC_ASSERT(MALI_QUEUE_COUNT == 1);
   const struct mali_csf_queue *q = cmd->dev->csf->queue;
   const bool once = cmd->usage & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
   for (uint32_t i = 0; i < td_count; i++) {
      pan_cast_and_pack((uint8_t *)p.cpu + i * pan_size(TILER_CONTEXT), TILER_CONTEXT, cfg) {
         if (once) {
            cfg.heap = mali_queue_heap_desc_va(q);
            cfg.geometry_buffer_size = MALI_TILER_GEOM_BUF_SIZE;
            cfg.geometry_buffer = mali_queue_geom_buf_va(q);
         }
         cfg.hierarchy_mask = MALI_TILER_HIERARCHY_MASK;
         cfg.sample_pattern = MALI_SAMPLE_PATTERN_SINGLE_SAMPLED;
         cfg.first_provoking_vertex = true;
         cfg.fb_width = r->desc.width;
         cfg.fb_height = r->desc.height;
         cfg.layer_count = MIN2(layers - i * MALI_LAYERS_PER_TILER_CTX,
                                MALI_LAYERS_PER_TILER_CTX);
         cfg.layer_offset = -(int32_t)(i * MALI_LAYERS_PER_TILER_CTX);
      }
   }
   r->tiler = p.gpu;
   r->tiler_cpu = p.cpu;
   r->td_count = td_count;

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);
   if (!once) {
      struct cs_index zero_lo = mali_cs_scratch_reg_tuple(b, 0, 2);
      struct cs_index heap_geom = mali_cs_scratch_reg_tuple(b, 2, 4);
      struct cs_index zeros = mali_cs_scratch_reg_tuple(b, 6, 8);
      struct cs_index addr = mali_cs_scratch_reg64(b, 14);

      cs_move64_to(b, zero_lo, 0);
      for (unsigned i = 0; i < 8; i += 2)
         cs_move64_to(b, mali_cs_scratch_reg64(b, 6 + i), 0);
      STATIC_ASSERT(offsetof(struct mali_cs_subqueue_ctx, geom_buf) ==
                    offsetof(struct mali_cs_subqueue_ctx, tiler_heap) + 8);
      cs_load_to(b, heap_geom, mali_cs_subqueue_ctx_reg(b), BITFIELD_MASK(4),
                 offsetof(struct mali_cs_subqueue_ctx, tiler_heap));

      for (uint32_t i = 0; i < td_count; i++) {
         cs_move64_to(b, addr, r->tiler + i * pan_size(TILER_CONTEXT));
         cs_store(b, zero_lo, addr, BITFIELD_MASK(2), 0);          /* polygon list */
         cs_store(b, heap_geom, addr, BITFIELD_MASK(4), 6 * 4);    /* heap, geometry buffer */
         cs_store(b, mali_cs_scratch_reg_tuple(b, 6, 4), addr, BITFIELD_MASK(4),
                  10 * 4);                                         /* completed top/bottom */
         cs_store(b, zeros, addr, BITFIELD_MASK(8), 24 * 4);       /* private state */
      }
      cs_flush_stores(b);
   }

   /* Timing: the vertex/tiler side of the pass starts here. */
   if (unlikely(cmd->measure))
      r->measure_vt = mali_measure_begin(cmd, MALI_MEASURE_PASS_VT,
                                         MALI_SUBQUEUE_VERTEX_TILER, cmd->passes - 1);

   /* The endpoint scoreboard entry the pass's draws signal, then the first
    * heap-counter step. */
   next_iter_sb(cmd, b);
   cs_vt_start(b, cs_now());

   cmd->gfx.draw.dirty |= MALI_GFX_DIRTY_PASS;
   return true;
}

/* ---------------------------------------------------------------------- */
/* CRC (transaction elimination)                                           */

/*
 * The fragment unit can keep a 64-bit CRC per 16x16 tile of one colour
 * target in a table next to the image (mali_image_layout.c). With CRC
 * read on, a tile whose new CRC equals the stored one is not written to
 * memory; with CRC write on, the new CRCs are stored.
 *
 * A stored CRC is only right while the tile's memory holds what that CRC
 * was computed from. Everything that writes the image other than a pass
 * that keeps its CRCs must therefore make the stored CRCs useless. panvk
 * v11 does this without touching the table: the low 16 bits of the CRC
 * Clear Color word are the seed the hardware starts each CRC from, taken
 * from a word in the image's CRC header when the pass runs; an invalidation
 * adds one to that word, so every CRC stored before it stops matching. The
 * blob does the same with a device-wide stamp counter per slice. We take
 * panvk's per-image counter: no device-global state and one
 * load/add/store per invalidation.
 *
 * The seed is 16 bits, so a CRC stored 65,536 invalidations ago matches
 * again if its tile was not rendered since and the pass renders exactly the
 * tile contents it was computed from while memory holds something else
 * (panvk has the same limit).
 */

struct mali_crc_options mali_crc_options = {
   .empty_tile_read = true,
   .empty_tile_write = true,
};

/* Seed word in the CRC header. */
#define CRC_SEED_OFFSET 0
#define CRC_SEED_MASK 0xffff
/* Fragment-stream scratch registers of the seed code: above the ones the
 * tiling wait and the barrier flush use (0-3), so the pass patch's store
 * can still be reading them while those run. */
#define CRC_SCRATCH 8

void
mali_cmd_crc_invalidate(struct mali_cmd_buffer *cmd, const struct mali_image *image)
{
   if (!image || !image->crc_header)
      return;
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_FRAGMENT);
   struct cs_index addr = mali_cs_scratch_reg64(b, CRC_SCRATCH);
   struct cs_index seed = mali_cs_scratch_reg32(b, CRC_SCRATCH + 2);

   /* A pending store may be the pass patch or an earlier bump of the same
    * word. The store is waited for at once: cs_builder does not track the
    * registers a pending store reads, and bumps are rare. */
   cs_flush_stores(b);
   cs_move64_to(b, addr, image->crc_header);
   cs_load32_to(b, seed, addr, CRC_SEED_OFFSET);
   cs_add_imm32(b, seed, seed, 1);
   cs_store32(b, seed, addr, CRC_SEED_OFFSET);
   cs_flush_stores(b);
}

/* Can target t keep its CRCs in this pass? panvk pan_fb_store_target_
 * should_crc on v11: level 0 of an image with CRC state, stored, tiles of
 * at least 16x16 pixels. One layer, since CRC images have one. */
static bool
target_can_crc(const struct mali_render_state *r, const struct mali_fb_target *t)
{
   return t->image && t->image->crc_header && t->store && t->level == 0 && t->plane == 0 &&
          r->tile_size >= 16 * 16 && MAX2(r->desc.layer_count, 1) == 1;
}

/* The hardware keeps CRCs for one render target: the first that can. */
static int
select_crc_rt(const struct mali_render_state *r)
{
   if (mali_crc_options.disable)
      return -1;
   for (unsigned i = 0; i < r->desc.rt_count; i++) {
      if (target_can_crc(r, &r->desc.rt[i]))
         return i;
   }
   return -1;
}

/*
 * The CRC Clear Color (pan_fb_crc_clear_color): a hash of the packed clear
 * values of the targets the pass clears, above bit 16, with bit 62 set so
 * it is never zero. With empty-tile write the hardware stores it as the
 * CRC of a tile that has no geometry and is only cleared; with empty-tile
 * read such a tile is not written when its stored CRC already equals it.
 * Bits 0-15 are the seed, filled in by the fragment stream.
 */
static uint64_t
crc_clear_color(const struct mali_render_state *r)
{
   static const uint64_t primes[4] = {16381, 16369, 16363, 16361};
   uint64_t hash = 0;
   for (unsigned i = 0; i < r->desc.rt_count; i++) {
      const struct mali_fb_target *t = &r->desc.rt[i];
      if (!t->image || t->load != MALI_ATT_LOAD_CLEAR)
         continue;
      uint32_t clear[4];
      pack_clear_color(t->format, &r->desc.clear_color[i], clear);
      for (unsigned c = 0; c < 4; c++)
         hash ^= primes[c] * clear[c] * (i + 1);
   }
   return ((1ull << 46) | hash) << 16;
}

/*
 * Fragment stream, before the pass's fragment job: put the CRC target's
 * current seed into the CRC Clear Color of the FBD, and invalidate every
 * other stored target that has CRC state (its tiles are written without
 * CRCs). fbd is the FBD's address, untagged.
 */
static void
emit_crc_updates(struct mali_cmd_buffer *cmd, const struct mali_render_state *r, uint64_t fbd)
{
   for (unsigned i = 0; i < r->desc.rt_count; i++) {
      const struct mali_fb_target *t = &r->desc.rt[i];
      if ((int)i != r->crc_rt && t->image && t->store && t->level == 0 && t->plane == 0)
         mali_cmd_crc_invalidate(cmd, t->image);
   }
   if (r->crc_rt < 0)
      return;

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_FRAGMENT);
   struct cs_index addr = mali_cs_scratch_reg64(b, CRC_SCRATCH);
   struct cs_index seed = mali_cs_scratch_reg32(b, CRC_SCRATCH + 2);
   struct cs_index mask = mali_cs_scratch_reg32(b, CRC_SCRATCH + 3);
   struct cs_index lo = mali_cs_scratch_reg32(b, CRC_SCRATCH + 4);

   cs_flush_stores(b);
   cs_move64_to(b, addr, r->desc.rt[r->crc_rt].image->crc_header);
   cs_load32_to(b, seed, addr, CRC_SEED_OFFSET);
   cs_move32_to(b, mask, CRC_SEED_MASK);
   cs_move32_to(b, lo, r->crc_clear_lo & ~CRC_SEED_MASK);
   cs_and32(b, seed, seed, mask);
   cs_or32(b, lo, lo, seed);
   /* CRC Clear Color is word 2 of the ZS/CRC extension, which follows
    * the Framebuffer Parameters. */
   cs_move64_to(b, addr, fbd + pan_size(FRAMEBUFFER) + 2 * 4);
   cs_store32(b, lo, addr, 0);
   /* The caller waits for the store before the fragment job. */
}

/* ---------------------------------------------------------------------- */
/* Framebuffer descriptors                                                 */

static void
target_mem(const struct mali_fb_target *t, unsigned layer, uint64_t *base,
           uint32_t *row_stride, uint64_t *surf_stride)
{
   const struct mali_image *img = t->image;
   const struct mali_image_plane_layout *pl = &img->planes[t->plane].layout;
   const struct mali_image_slice *s = &pl->slices[t->level];

   *base = img->base + s->offset;
   if (img->vk.image_type == VK_IMAGE_TYPE_3D)
      *base += (t->layer + layer) * s->surface_stride;
   else
      *base += (uint64_t)(t->layer + layer) * pl->array_stride;
   *row_stride = s->row_stride;
   *surf_stride = s->surface_stride;
}

/* get_afbc_att_mem_props: the layer's first header, the body offset from
 * it and the header row stride. */
static void
target_afbc_mem(const struct mali_fb_target *t, unsigned layer, uint64_t *header,
                uint32_t *body_offset, uint32_t *row_stride)
{
   const struct mali_image *img = t->image;
   const struct mali_image_plane_layout *pl = &img->planes[t->plane].layout;
   const struct mali_image_slice *s = &pl->slices[t->level];

   *header = img->base + s->offset + (uint64_t)(t->layer + layer) * pl->array_stride;
   *body_offset = mali_afbc_body_offset(s);
   *row_stride = s->row_stride;
}

/* pan_needs_afbc_reverse_issue_order: a tile smaller than a superblock in
 * either direction shares superblocks with its neighbours. */
static bool
afbc_reverse_issue_order(const struct mali_render_state *r)
{
   uint32_t tw, th;
   tile_dims(r->tile_size, &tw, &th);
   return tw < AFBC_BLOCK || th < AFBC_BLOCK;
}

static enum mali_block_format
target_block_format(const struct mali_fb_target *t)
{
   if (mali_image_is_afbc(t->image))
      return MALI_BLOCK_FORMAT_AFBC_TILED;
   return t->image->vk.drm_format_mod == DRM_FORMAT_MOD_LINEAR ?
             MALI_BLOCK_FORMAT_LINEAR : MALI_BLOCK_FORMAT_TILED_U_INTERLEAVED;
}

/* Does the colour frame shader clear some target (so every tile has to
 * be written back)? */
static bool
clears_in_area(const struct mali_fb_target *t)
{
   return t->image && t->preload && t->load == MALI_ATT_LOAD_CLEAR;
}

/* Tiles without geometry are written back only for targets that are
 * cleared (by the tile buffer or by the frame shader). */
static bool
clean_tile_write(const struct mali_fb_target *t)
{
   return t->image && t->store && (t->load == MALI_ATT_LOAD_CLEAR || t->always_write);
}

static uint32_t
fbd_size(const struct mali_render_state *r, bool has_zs)
{
   return pan_size(FRAMEBUFFER) + (has_zs ? pan_size(ZS_CRC_EXTENSION) : 0) +
          r->rt_count * pan_size(RENDER_TARGET);
}

/* A single-sampled sample position table (pan_samples.c: positions in
 * 1/256 pixel, centred; 64 entries of 4 bytes). */
static uint64_t
sample_positions(struct mali_cmd_buffer *cmd)
{
   struct mali_ptr p = mali_cmd_alloc(cmd, 64 * 4, 64);
   if (!p.cpu)
      return 0;
   uint16_t pos[128];
   memset(pos, 0, sizeof(pos));
   pos[0] = pos[1] = 128;       /* sample 0 */
   pos[64] = pos[65] = 128;     /* origin */
   memcpy(p.cpu, pos, sizeof(pos));
   return p.gpu;
}

/* The frame shaders of the pass: colour preload in DCD slot 0 (Intersect,
 * or Always when a clear inside the area is done by the shader), depth and
 * stencil in slot 1 (Early ZS Always), as panvk v9+. */
static bool
build_frame_shaders(struct mali_cmd_buffer *cmd, struct mali_render_state *r,
                    uint64_t *dcds, enum mali_pre_post_frame_shader_mode modes[3])
{
   struct mali_meta_fs_key ckey = {0}, zkey = {0};
   uint32_t ctex[10][8], ztex[2][8];
   unsigned nctex = 0, nztex = 0;
   bool color = false, zs = false, color_always = false;
   const bool layered = r->desc.layer_count > 1;

   struct mali_meta_fs_push push = {
      .area = {(int32_t)r->desc.area.offset.x, (int32_t)r->desc.area.offset.y,
               (int32_t)(r->desc.area.offset.x + r->desc.area.extent.width) - 1,
               (int32_t)(r->desc.area.offset.y + r->desc.area.extent.height) - 1},
      .clear_depth = r->desc.clear_depth,
      .clear_stencil = r->desc.clear_stencil,
   };

   for (unsigned i = 0; i < r->desc.rt_count && !r->desc.blit; i++) {
      const struct mali_fb_target *t = &r->desc.rt[i];
      if (!t->image || !t->preload)
         continue;
      ckey.rt_op[i] = t->load == MALI_ATT_LOAD_CLEAR ? MALI_META_FS_CLEAR_IN_AREA
                                                      : MALI_META_FS_LOAD;
      ckey.rt_type[i] = util_format_is_pure_uint(t->format) ? MALI_META_FS_UINT :
                        util_format_is_pure_sint(t->format) ? MALI_META_FS_SINT :
                                                              MALI_META_FS_FLOAT;
      memcpy(push.clear_color[i], &r->desc.clear_color[i], 16);
      if (!mali_cmd_pack_plane_texture(cmd, t->image, t->plane, t->format, t->level,
                                       t->layer, MAX2(r->desc.layer_count, 1),
                                       ctex[nctex++]))
         return false;
      color = true;
      color_always |= clears_in_area(t);
   }
   ckey.layered = zkey.layered = layered;

   const struct mali_fb_target *zt = &r->desc.z, *st = &r->desc.s;
   if (zt->image && zt->preload) {
      zkey.z_op = zt->load == MALI_ATT_LOAD_CLEAR ? MALI_META_FS_CLEAR_IN_AREA
                                                  : MALI_META_FS_LOAD;
      if (!mali_cmd_pack_plane_texture(cmd, zt->image, zt->plane, zt->format, zt->level,
                                       zt->layer, MAX2(r->desc.layer_count, 1),
                                       ztex[nztex++]))
         return false;
      zs = true;
   }
   if (st->image && st->preload) {
      zkey.s_op = st->load == MALI_ATT_LOAD_CLEAR ? MALI_META_FS_CLEAR_IN_AREA
                                                  : MALI_META_FS_LOAD;
      if (!mali_cmd_pack_plane_texture(cmd, st->image, st->plane, st->format, st->level,
                                       st->layer, MAX2(r->desc.layer_count, 1),
                                       ztex[nztex++]))
         return false;
      zs = true;
   }

   if (!color && !zs && !r->desc.blit)
      return true;

   struct mali_ptr p = mali_cmd_alloc(cmd, FS_DCD_COUNT * pan_size(DRAW), 64);
   if (!p.cpu)
      return false;
   memset(p.cpu, 0, FS_DCD_COUNT * pan_size(DRAW));

   if (r->desc.blit) {
      const struct mali_render_blit *bl = r->desc.blit;
      const struct mali_shader *fs = mali_meta_fs_get(cmd, bl->key);
      if (!fs || !mali_meta_fs_dcd(cmd, fs, bl->key, bl->push, bl->textures,
                                   bl->texture_count, bl->sampler, true,
                                   (uint8_t *)p.cpu + FS_DCD_COLOR * pan_size(DRAW)))
         return false;
      modes[FS_DCD_COLOR] = MALI_PRE_POST_FRAME_SHADER_MODE_ALWAYS;
   } else if (color) {
      const struct mali_shader *fs = mali_meta_fs_get(cmd, &ckey);
      if (!fs || !mali_meta_fs_dcd(cmd, fs, &ckey, &push, (const uint32_t(*)[8])ctex,
                                   nctex, NULL, true,
                                   (uint8_t *)p.cpu + FS_DCD_COLOR * pan_size(DRAW)))
         return false;
      modes[FS_DCD_COLOR] = color_always ? MALI_PRE_POST_FRAME_SHADER_MODE_ALWAYS
                                         : MALI_PRE_POST_FRAME_SHADER_MODE_INTERSECT;
   }
   if (zs) {
      const struct mali_shader *fs = mali_meta_fs_get(cmd, &zkey);
      if (!fs || !mali_meta_fs_dcd(cmd, fs, &zkey, &push, (const uint32_t(*)[8])ztex,
                                   nztex, NULL, true,
                                   (uint8_t *)p.cpu + FS_DCD_ZS * pan_size(DRAW)))
         return false;
      /* Intersect: tiles without geometry keep their memory and need no
       * reload. panvk uses Early ZS Always (the reload runs a tile or more
       * ahead) without having measured it; on ARMSX2's many small passes it
       * reloads and writes back every tile of the render area in every
       * pass. A clear inside the area has to reach every tile. */
      const bool zs_clears = (zt->image && zt->preload && zt->load == MALI_ATT_LOAD_CLEAR) ||
                             (st->image && st->preload && st->load == MALI_ATT_LOAD_CLEAR);
      modes[FS_DCD_ZS] = zs_clears ? MALI_PRE_POST_FRAME_SHADER_MODE_ALWAYS
                                   : MALI_PRE_POST_FRAME_SHADER_MODE_INTERSECT;
   }
   *dcds = p.gpu;
   return true;
}

static void
pack_zs_ext(const struct mali_render_state *r, unsigned layer, uint64_t crc_clear, void *out)
{
   const struct mali_fb_target *zt = &r->desc.z, *st = &r->desc.s;
   struct mali_zs_crc_extension_packed desc, part;

   pan_pack(&desc, ZS_CRC_EXTENSION, cfg) {
      cfg.zs.msaa = MALI_MSAA_SINGLE;
      cfg.s.msaa = MALI_MSAA_SINGLE;
      cfg.zs.clean_tile_write_enable = clean_tile_write(zt);
      cfg.s.clean_tile_write_enable = clean_tile_write(st);
   }

   if (r->crc_rt >= 0) {
      const struct mali_image *img = r->desc.rt[r->crc_rt].image;
      const struct mali_image_crc_layout *c = &img->planes[0].layout.crc;
      pan_pack(&part, ZS_CRC_EXTENSION, cfg) {
         cfg.crc.base = img->crc_header + MALI_CRC_HEADER_SIZE;
         cfg.crc.row_stride = c->row_stride;
         cfg.crc.render_target = r->crc_rt;
         cfg.crc.clear_color = crc_clear;
      }
      pan_merge(&desc, &part, ZS_CRC_EXTENSION);
   }

   if (zt->image && zt->store && target_is_afbc(zt)) {
      uint64_t header;
      uint32_t body, row;
      target_afbc_mem(zt, layer, &header, &body, &row);
      pan_pack(&part, ZS_CRC_EXTENSION, cfg) {
         cfg.afbc_zs.write_format = zs_write_format(zt->format);
         cfg.afbc_zs.block_format = MALI_BLOCK_FORMAT_AFBC_TILED;
         cfg.afbc_zs.reverse_issue_order = afbc_reverse_issue_order(r);
         cfg.afbc_zs.header = header;
         cfg.afbc_zs.body_offset = body;
         cfg.afbc_zs.header_row_stride = row;
      }
      pan_merge(&desc, &part, ZS_CRC_EXTENSION);
   } else if (zt->image && zt->store) {
      uint64_t base, surf;
      uint32_t row;
      target_mem(zt, layer, &base, &row, &surf);
      pan_pack(&part, ZS_CRC_EXTENSION, cfg) {
         cfg.zs.write_format = zs_write_format(zt->format);
         cfg.zs.block_format = target_block_format(zt);
         cfg.zs.base = base;
         cfg.zs.row_stride = row;
         cfg.zs.surface_stride = (uint32_t)surf;
         cfg.zs.surface_stride_hi = (uint32_t)(surf >> 32);
      }
      pan_merge(&desc, &part, ZS_CRC_EXTENSION);
   }
   if (st->image && st->store && target_is_afbc(st)) {
      uint64_t header;
      uint32_t body, row;
      target_afbc_mem(st, layer, &header, &body, &row);
      pan_pack(&part, ZS_CRC_EXTENSION, cfg) {
         cfg.afbc_s.write_format = MALI_S_FORMAT_S8;
         cfg.afbc_s.block_format = MALI_BLOCK_FORMAT_AFBC_TILED;
         cfg.afbc_s.reverse_issue_order = afbc_reverse_issue_order(r);
         cfg.afbc_s.header = header;
         cfg.afbc_s.body_offset = body;
         cfg.afbc_s.header_row_stride = row;
      }
      pan_merge(&desc, &part, ZS_CRC_EXTENSION);
   } else if (st->image && st->store) {
      uint64_t base, surf;
      uint32_t row;
      target_mem(st, layer, &base, &row, &surf);
      pan_pack(&part, ZS_CRC_EXTENSION, cfg) {
         cfg.s.write_format = MALI_S_FORMAT_S8;
         cfg.s.block_format = target_block_format(st);
         cfg.s.base = base;
         cfg.s.row_stride = row;
         cfg.s.surface_stride = (uint32_t)surf;
         cfg.s.surface_stride_hi = (uint32_t)(surf >> 32);
      }
      pan_merge(&desc, &part, ZS_CRC_EXTENSION);
   }
   memcpy(out, &desc, sizeof(desc));
}

/* pan_emit_afbc_color_attachment for v9+. */
static void
pack_rt_afbc(const struct mali_render_state *r, const struct mali_fb_target *t, unsigned i,
             unsigned layer, const uint32_t clear[4], void *out)
{
   uint64_t header;
   uint32_t body, row;
   target_afbc_mem(t, layer, &header, &body, &row);

   pan_cast_and_pack(out, AFBC_RGB_RENDER_TARGET, cfg) {
      cfg.internal_buffer_offset = r->rt_offset[i];
      cfg.dithering_enable = true;
      cfg.clean_tile_write_enable = clean_tile_write(t);
      cfg.clear.color_0 = clear[0];
      cfg.clear.color_1 = clear[1];
      cfg.clear.color_2 = clear[2];
      cfg.clear.color_3 = clear[3];
      cfg.write_enable = t->store;
      rt_formats(t->format, &cfg.writeback_format, &cfg.internal_format, &cfg.swizzle);
      /* No AFBC with RAW24 on v9+: RAW32, the compression mode says 24. */
      if (cfg.writeback_format == MALI_COLOR_FORMAT_RAW24)
         cfg.writeback_format = MALI_COLOR_FORMAT_RAW32;
      cfg.srgb = util_format_is_srgb(t->format);
      cfg.writeback_block_format = MALI_BLOCK_FORMAT_AFBC_TILED;
      cfg.writeback_msaa = MALI_MSAA_SINGLE;
      cfg.yuv_transform = t->image->vk.drm_format_mod & AFBC_FORMAT_MOD_YTR;
      cfg.wide_block = false;
      cfg.split_block = false;
      cfg.reverse_issue_order = afbc_reverse_issue_order(r);
      cfg.header = header;
      cfg.body_offset = body;
      cfg.row_stride = row;
      cfg.compression_mode = mali_image_afbc_hw_mode(t->image, t->plane);
   }
}

static void
pack_rt(const struct mali_render_state *r, unsigned i, unsigned layer, void *out)
{
   const struct mali_fb_target *t = i < r->desc.rt_count ? &r->desc.rt[i] : NULL;

   if (!t || !t->image) {
      /* The hardware wants at least one target; a missing one sits past
       * the allocation, where writes are dropped (pan_fb.c). */
      pan_cast_and_pack(out, RGB_RENDER_TARGET, cfg) {
         cfg.internal_buffer_offset = r->cbuf_alloc;
         cfg.internal_format = MALI_COLOR_BUFFER_INTERNAL_FORMAT_R8G8B8A8;
         cfg.write_enable = false;
         cfg.writeback_block_format = MALI_BLOCK_FORMAT_TILED_U_INTERLEAVED;
      }
      return;
   }

   uint32_t clear[4] = {0};
   if (t->load == MALI_ATT_LOAD_CLEAR)
      pack_clear_color(t->format, &r->desc.clear_color[i], clear);

   if (target_is_afbc(t)) {
      pack_rt_afbc(r, t, i, layer, clear, out);
      return;
   }

   uint64_t base, surf;
   uint32_t row;
   target_mem(t, layer, &base, &row, &surf);

   pan_cast_and_pack(out, RGB_RENDER_TARGET, cfg) {
      cfg.internal_buffer_offset = r->rt_offset[i];
      cfg.dithering_enable = true;
      cfg.clean_tile_write_enable = clean_tile_write(t);
      cfg.clear.color_0 = clear[0];
      cfg.clear.color_1 = clear[1];
      cfg.clear.color_2 = clear[2];
      cfg.clear.color_3 = clear[3];
      cfg.write_enable = t->store;
      cfg.writeback_block_format = target_block_format(t);
      rt_formats(t->format, &cfg.writeback_format, &cfg.internal_format, &cfg.swizzle);
      cfg.srgb = util_format_is_srgb(t->format);
      cfg.writeback_msaa = MALI_MSAA_SINGLE;
      cfg.writeback_buffer.base = base;
      cfg.writeback_buffer.row_stride = row;
      cfg.writeback_buffer.surface_stride = (uint32_t)surf;
      cfg.writeback_buffer.surface_stride_hi = (uint32_t)(surf >> 32);
   }
}

static enum mali_pre_post_frame_shader_mode
fix_frame_mode(enum mali_pre_post_frame_shader_mode mode, bool clean_tile)
{
   /* pan_fix_frame_shader_mode: with clean-tile writes, every tile runs. */
   if (clean_tile && mode == MALI_PRE_POST_FRAME_SHADER_MODE_INTERSECT)
      return MALI_PRE_POST_FRAME_SHADER_MODE_ALWAYS;
   return mode;
}

/* The pass's framebuffer descriptors, one per layer. Returns the tagged
 * pointer for FBD_POINTER, 0 on failure. */
static uint64_t
build_fbds(struct mali_cmd_buffer *cmd, struct mali_render_state *r, uint32_t *size_out)
{
   const struct mali_fb_target *zt = &r->desc.z, *st = &r->desc.s;
   r->crc_rt = select_crc_rt(r);
   const bool crc = r->crc_rt >= 0;
   const bool has_zs = zt->image || st->image || crc;
   const uint32_t layers = MAX2(r->desc.layer_count, 1);
   const uint32_t size = fbd_size(r, has_zs);
   const uint64_t crc_clear = crc ? crc_clear_color(r) : 0;
   r->crc_clear_lo = (uint32_t)crc_clear;

   uint64_t dcds = 0;
   enum mali_pre_post_frame_shader_mode modes[3] = {0};
   if (!build_frame_shaders(cmd, r, &dcds, modes))
      return 0;

   bool clean = clean_tile_write(zt) || clean_tile_write(st);
   for (unsigned i = 0; i < r->desc.rt_count; i++)
      clean |= clean_tile_write(&r->desc.rt[i]);
   /* panvk runs the preloads of a pass with a CRC target on every tile
    * (pan_emit_fb_desc: "Pre-frame shaders that preload a CRC-enabled RT
    * must run in ALWAYS mode"). We keep Intersect: on the device a LOAD
    * pass with CRC leaves tiles without geometry alone and does not stamp
    * them (vk_render_smoke part 9), and Always would read back every tile
    * of ARMSX2's full-target passes. The option restores panvk's rule. */
   const bool always_preload = clean || (crc && mali_crc_options.always_preload);

   uint64_t samples = sample_positions(cmd);
   struct mali_ptr p = mali_cmd_alloc(cmd, (uint64_t)size * layers, 64);
   if (!p.cpu || !samples)
      return 0;

   for (uint32_t l = 0; l < layers; l++) {
      uint8_t *fbd = (uint8_t *)p.cpu + (uint64_t)l * size;

      pan_section_pack(fbd, FRAMEBUFFER, PARAMETERS, cfg) {
         cfg.pre_frame_0 = fix_frame_mode(modes[0], always_preload);
         cfg.pre_frame_1 = fix_frame_mode(modes[1], always_preload);
         cfg.post_frame = MALI_PRE_POST_FRAME_SHADER_MODE_NEVER;
         cfg.frame_shader_dcds = dcds;
         cfg.sample_locations = samples;
         /* The layer inside its tiler context selects the primitive list;
          * the frame argument reaches the fragment shader as the layer. */
         cfg.internal_layer_index = l % MALI_LAYERS_PER_TILER_CTX;
         cfg.frame_argument = l;
         cfg.width = r->desc.width;
         cfg.height = r->desc.height;
         cfg.bound_min_x = r->minx;
         cfg.bound_min_y = r->miny;
         cfg.bound_max_x = r->maxx;
         cfg.bound_max_y = r->maxy;
         cfg.sample_count = 1;
         cfg.sample_pattern = MALI_SAMPLE_PATTERN_SINGLE_SAMPLED;
         cfg.tie_break_rule = MALI_TIE_BREAK_RULE_MINUS_180_IN_0_OUT;
         cfg.effective_tile_size = r->tile_size;
         cfg.point_sprite_coord_origin_max_y = false;
         cfg.first_provoking_vertex = true;
         cfg.render_target_count = r->rt_count;
         cfg.color_buffer_allocation = r->cbuf_alloc;
         if (st->image) {
            cfg.s_clear = st->load == MALI_ATT_LOAD_CLEAR ? r->desc.clear_stencil : 0;
            cfg.s_write_enable = st->store;
         }
         if (zt->image) {
            cfg.z_internal_format = z_internal_format(zt->format);
            cfg.z_clear = zt->load == MALI_ATT_LOAD_CLEAR ? r->desc.clear_depth : 0;
            cfg.z_write_enable = zt->store;
         } else {
            cfg.z_internal_format = MALI_Z_INTERNAL_FORMAT_D24;
         }
         cfg.has_zs_crc_extension = has_zs;
         if (crc) {
            cfg.crc_read_enable = true;
            cfg.crc_write_enable = true;
            cfg.empty_tile_read_enable = mali_crc_options.empty_tile_read && r->rt_count == 1;
            cfg.empty_tile_write_enable = mali_crc_options.empty_tile_write;
         }
         cfg.tiler = r->tiler ? r->tiler + (l / MALI_LAYERS_PER_TILER_CTX) *
                                              pan_size(TILER_CONTEXT) : 0;
      }
      pan_section_pack(fbd, FRAMEBUFFER, PADDING, cfg);

      uint8_t *next = fbd + pan_size(FRAMEBUFFER);
      if (has_zs) {
         pack_zs_ext(r, l, crc_clear, next);
         next += pan_size(ZS_CRC_EXTENSION);
      }
      for (unsigned i = 0; i < r->rt_count; i++)
         pack_rt(r, i, l, next + i * pan_size(RENDER_TARGET));
   }

   struct mali_framebuffer_pointer_packed tag;
   pan_pack(&tag, FRAMEBUFFER_POINTER, cfg) {
      cfg.zs_crc_extension_present = has_zs;
      cfg.render_target_count = r->rt_count;
   }
   *size_out = size;
   return p.gpu | tag.opaque[0];
}

/* ---------------------------------------------------------------------- */
/* End                                                                     */

/* The pass's thread storage: TLS for the largest per-thread need of its
 * shaders, no workgroup memory. */
static void
fill_tsd(struct mali_cmd_buffer *cmd, struct mali_render_state *r)
{
   if (!r->tsd_cpu)
      return;
   uint64_t tls = r->tls_size ? mali_cmd_tls_buffer(cmd, r->tls_size) : 0;
   pan_cast_and_pack(r->tsd_cpu, LOCAL_STORAGE, cfg) {
      if (tls) {
         cfg.tls_size = util_logbase2_ceil(DIV_ROUND_UP(r->tls_size, 16));
         cfg.tls_address_mode = MALI_ADDRESS_MODE_PACKED;
         cfg.tls_base_pointer = tls >> 8;
      }
      cfg.wls_instances = MALI_LOCAL_STORAGE_NO_WORKGROUP_MEM;
   }
}

/* Vertex/tiler side of the end (panvk flush_tiling): finish tiling, the
 * deferred HEAP_OPERATION{Vertex/Tiler Completed} behind every draw of the
 * pass, and the signal the fragment side waits on: a deferred increment of
 * the shared scoreboard entry (the blob on this GPU), or without shared
 * entries a deferred SYNC_ADD64 on the vertex/tiler barrier counter
 * (panvk). */
static void
flush_tiling(struct mali_cmd_buffer *cmd)
{
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);

   /* The draws' copies of uniform-buffer words into fragment FAU blocks
    * (mali_cmd_draw.c emit_ubo_push) must land before the fragment job
    * reads them. */
   cs_flush_stores(b);
   if (cmd->gfx.render.ubo_stores) {
      cs_wait_slot(b, MALI_SB_LS);
      cmd->gfx.render.ubo_stores = false;
   }
   cs_finish_tiling(b);
   if (cmd->dev->csf->shared_sb) {
      cs_vt_end(b, cs_defer_indirect());
      mali_cs_shared_sb_inc(b, MALI_SHARED_SB_VT_TO_FRAG, cs_defer_indirect());
      return;
   }

   struct cs_index addr = mali_cs_scratch_reg64(b, 0);
   struct cs_index one = mali_cs_scratch_reg64(b, 2);
   cs_move64_to(b, addr, mali_queue_syncobj_va(cmd->dev->csf->queue,
                                                MALI_SUBQUEUE_VERTEX_TILER));
   cs_move64_to(b, one, 1);
   cs_vt_end(b, cs_defer_indirect());
   cs_sync64_add(b, true, MALI_CS_SYNC_SCOPE_CSG, one, addr, cs_defer_indirect());
   cmd->cs[MALI_SUBQUEUE_VERTEX_TILER].relative_sync_point++;
}

static void
issue_fragment_jobs(struct mali_cmd_buffer *cmd, struct mali_render_state *r,
                    uint64_t fbd_ptr, uint32_t fbd_sz)
{
   const struct mali_csf_device *csf = cmd->dev->csf;
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_FRAGMENT);

   cs_move32_to(b, cs_sr_reg32(b, FRAGMENT, BBOX_MIN), r->miny << 16 | r->minx);
   cs_move32_to(b, cs_sr_reg32(b, FRAGMENT, BBOX_MAX), r->maxy << 16 | r->maxx);

   /* The CRC seed patch first: seeds are only read and bumped on this
    * stream, so stream order is all it needs, and its load and store then
    * overlap the tiling wait. */
   emit_crc_updates(cmd, r, fbd_ptr & ~(uint64_t)63);

   /* Wait for the tiling of this pass: take one off the shared entry the
    * vertex/tiler side increments per pass (a pass that tiled nothing has
    * no increment and waits for nothing), or, without shared entries,
    * panvk's wait_finish_tiling: the vertex/tiler counter passes progress +
    * signals so far. Then the barrier wait and cache flush put off until
    * this job; in this order the tiling wait, which the firmware takes
    * microseconds to pass even when it is already satisfied, overlaps the
    * previous fragment job. */
   if (csf->shared_sb) {
      if (r->tiler)
         mali_cs_shared_sb_dec(b, MALI_SHARED_SB_VT_TO_FRAG);
   } else {
      struct cs_index addr = mali_cs_scratch_reg64(b, 0);
      struct cs_index ref = mali_cs_scratch_reg64(b, 2);
      cs_move64_to(b, addr, mali_queue_syncobj_va(cmd->dev->csf->queue,
                                                  MALI_SUBQUEUE_VERTEX_TILER));
      cs_add_imm64(b, ref, mali_cs_progress_seqno_reg(b, MALI_SUBQUEUE_VERTEX_TILER),
                   cmd->cs[MALI_SUBQUEUE_VERTEX_TILER].relative_sync_point);
      cs_sync64_wait(b, false, MALI_CS_CONDITION_GREATER, ref, addr);
   }
   mali_cmd_frag_flush_pending(cmd);
   /* The fragment job reads the patched FBD from memory. */
   if (r->crc_rt >= 0)
      cs_flush_stores(b);

   /* Timing: the fragment side, from the end of the tiling wait to
    * the end of the pass's fragment jobs. */
   if (unlikely(cmd->measure))
      r->measure_frag = mali_measure_begin(cmd, MALI_MEASURE_PASS_FRAG,
                                           MALI_SUBQUEUE_FRAGMENT, cmd->passes - 1);

   struct cs_index fbd = cs_sr_reg64(b, FRAGMENT, FBD_POINTER);
   cs_move64_to(b, fbd, fbd_ptr);
   const uint32_t layers = MAX2(r->desc.layer_count, 1);
   for (uint32_t l = 0; l < layers; l++) {
      if (l)
         cs_add_imm64(b, fbd, fbd, fbd_sz);
      cs_run_fragment(b, false, MALI_TILE_RENDER_ORDER_Z_ORDER);
   }
   if (unlikely(r->measure_frag))
      mali_measure_end(cmd, r->measure_frag);

   /* The heap chunks of the pass go back after the fragment job, in pass
    * order: wait on the current iterator entry (the RUN_FRAGMENTs and the
    * previous pass's FINISH_FRAGMENT), signal the next (panvk). Exactly one
    * fragment-completed increment per tiled pass, on the last context. */
   struct cs_index next_sb = mali_cs_scratch_reg32(b, 0);
   struct cs_index sb_mask = mali_cs_scratch_reg32(b, 1);
   struct cs_index tiler = mali_cs_scratch_reg64(b, 2);
   struct cs_index completed = mali_cs_scratch_reg_tuple(b, 4, 4);
   struct cs_index top = mali_cs_scratch_reg64(b, 4);
   struct cs_index bottom = mali_cs_scratch_reg64(b, 6);

   cs_next_sb_entry(b, next_sb, MALI_CS_SCOREBOARD_TYPE_ENDPOINT,
                    MALI_CS_NEXT_SB_ENTRY_FORMAT_INDEX);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_SEL_DEFERRED, next_sb);
   if (r->td_count) {
      cs_move64_to(b, tiler, r->tiler);
      for (uint32_t i = 0; i < r->td_count; i++) {
         cs_load_to(b, completed, tiler, BITFIELD_MASK(4),
                    i * pan_size(TILER_CONTEXT) + 10 * 4);
         cs_finish_fragment(b, i == r->td_count - 1, top, bottom, cs_defer_indirect());
      }
   }
   cs_set_state_imm32(b, MALI_CS_SET_STATE_TYPE_SB_SEL_DEFERRED, MALI_SB_DEFERRED_SYNC);

   /* panvk also adds one to the fragment counter here, for its descriptor
    * ring buffer; nothing waits on that counter per pass in this driver
    * (barriers signal it themselves), so the signal is left out, as the
    * blob does. */

   cs_move32_to(b, sb_mask, 0);
   cs_bit_set32(b, sb_mask, sb_mask, next_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_WAIT, sb_mask);
   cs_move32_to(b, sb_mask, csf->sb.all_iters_mask);
   cs_bit_clear32(b, sb_mask, sb_mask, next_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_STREAM, sb_mask);
}

void
mali_cmd_render_end(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   if (!r->active)
      return;

   /* Nothing tiled and nothing cleared or border-loaded: the attachments
    * keep their contents, no fragment job is needed. */
   bool work = r->tiler != 0;
   const struct mali_fb_target *ts[] = {&r->desc.z, &r->desc.s};
   for (unsigned i = 0; i < 2; i++)
      work |= ts[i]->image && ts[i]->store && ts[i]->load == MALI_ATT_LOAD_CLEAR;
   for (unsigned i = 0; i < r->desc.rt_count; i++) {
      const struct mali_fb_target *t = &r->desc.rt[i];
      work |= t->image && t->store && (t->load == MALI_ATT_LOAD_CLEAR || t->always_write);
   }

   if (work && !vk_command_buffer_has_error(&cmd->vk)) {
      bool ok = alloc_tsd(cmd, r);
      uint32_t fbd_sz = 0;
      uint64_t fbds = ok ? build_fbds(cmd, r, &fbd_sz) : 0;
      if (!fbds) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      } else {
         fill_tsd(cmd, r);
         if (r->tiler)
            flush_tiling(cmd);
         if (unlikely(r->measure_vt))
            mali_measure_end(cmd, r->measure_vt);
         issue_fragment_jobs(cmd, r, fbds, fbd_sz);
      }
   }
   if (unlikely(cmd->measure)) {
      const uint32_t w = r->desc.width, h = r->desc.height;
      mali_measure_info(cmd, r->measure_vt, w, h, r->desc.rt_count, r->draws);
      mali_measure_info(cmd, r->measure_frag, w, h, r->desc.rt_count, r->draws);
   }

   r->active = false;
   cmd->gfx.draw.dirty = MALI_GFX_DIRTY_ALL;
}

/* ---------------------------------------------------------------------- */
/* vkCmdBeginRendering / vkCmdEndRendering                                 */

static enum mali_att_load
att_load(VkAttachmentLoadOp op)
{
   switch (op) {
   case VK_ATTACHMENT_LOAD_OP_LOAD: return MALI_ATT_LOAD_LOAD;
   case VK_ATTACHMENT_LOAD_OP_CLEAR: return MALI_ATT_LOAD_CLEAR;
   default: return MALI_ATT_LOAD_DONT_CARE;
   }
}

static bool
att_store(const VkRenderingAttachmentInfo *att)
{
   return att->storeOp != VK_ATTACHMENT_STORE_OP_DONT_CARE &&
          att->storeOp != VK_ATTACHMENT_STORE_OP_NONE;
}

static void
set_target(struct mali_fb_target *t, const struct mali_image_view *view, unsigned plane,
           enum pipe_format format, const VkRenderingAttachmentInfo *att)
{
   const struct mali_image *image = container_of(view->vk.image, struct mali_image, vk);
   t->image = image;
   t->plane = plane;
   t->level = view->vk.base_mip_level;
   t->layer = view->vk.base_array_layer;
   t->format = format;
   t->load = att_load(att->loadOp);
   t->store = att_store(att);
}

VKAPI_ATTR void VKAPI_CALL
mali_CmdBeginRendering(VkCommandBuffer commandBuffer, const VkRenderingInfo *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   struct mali_render_desc d = {
      .area = info->renderArea,
      .layer_count = info->viewMask ? util_last_bit(info->viewMask) : info->layerCount,
      .rt_count = info->colorAttachmentCount,
   };
   uint32_t w = UINT32_MAX, h = UINT32_MAX;
   bool bound = false;

   if (info->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT))
      vk_logw(VK_LOG_OBJS(cmd), "libmali: suspended/resumed rendering is not supported");
   if (info->viewMask)
      vk_logw(VK_LOG_OBJS(cmd), "libmali: multiview rendering is not supported");

   assert(info->colorAttachmentCount <= MALI_MAX_RTS);
   for (uint32_t i = 0; i < info->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[i];
      VK_FROM_HANDLE(mali_image_view, view, att->imageView);
      if (!view)
         continue;
      if (view->vk.image->samples > 1)
         vk_logw(VK_LOG_OBJS(cmd), "libmali: multisampled attachments are not supported");
      set_target(&d.rt[i], view, 0, vk_format_to_pipe_format(view->vk.view_format), att);
      d.clear_color[i] = att->clearValue.color;
      w = MIN2(w, view->vk.extent.width);
      h = MIN2(h, view->vk.extent.height);
      bound = true;
   }

   const VkRenderingAttachmentInfo *za = info->pDepthAttachment;
   const VkRenderingAttachmentInfo *sa = info->pStencilAttachment;
   if (za && za->imageView) {
      VK_FROM_HANDLE(mali_image_view, view, za->imageView);
      const struct mali_image *img = container_of(view->vk.image, struct mali_image, vk);
      if (vk_format_has_depth(img->vk.format)) {
         unsigned plane = mali_image_aspect_plane(img, VK_IMAGE_ASPECT_DEPTH_BIT);
         set_target(&d.z, view, plane, img->planes[plane].format, za);
         d.clear_depth = za->clearValue.depthStencil.depth;
      }
      w = MIN2(w, view->vk.extent.width);
      h = MIN2(h, view->vk.extent.height);
      bound = true;
   }
   if (sa && sa->imageView) {
      VK_FROM_HANDLE(mali_image_view, view, sa->imageView);
      const struct mali_image *img = container_of(view->vk.image, struct mali_image, vk);
      /* A depth-only view of a depth/stencil image still names the
       * image's stencil plane (ARMSX2 attaches D32S8 through one). */
      if (vk_format_has_stencil(img->vk.format)) {
         unsigned plane = mali_image_aspect_plane(img, VK_IMAGE_ASPECT_STENCIL_BIT);
         set_target(&d.s, view, plane, img->planes[plane].format, sa);
         d.clear_stencil = sa->clearValue.depthStencil.stencil;
      }
      w = MIN2(w, view->vk.extent.width);
      h = MIN2(h, view->vk.extent.height);
      bound = true;
   }

   if (bound) {
      d.width = w;
      d.height = h;
   } else {
      /* No attachments: the framebuffer is the render area (panvk). */
      d.width = info->renderArea.offset.x + info->renderArea.extent.width;
      d.height = info->renderArea.offset.y + info->renderArea.extent.height;
   }
   d.width = MAX2(d.width, 1);
   d.height = MAX2(d.height, 1);
   d.layer_count = MAX2(d.layer_count, 1);

   mali_cmd_render_begin(cmd, &d);
}

VKAPI_ATTR void VKAPI_CALL
mali_CmdEndRendering(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   mali_cmd_render_end(cmd);
}
