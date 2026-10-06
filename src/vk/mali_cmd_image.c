/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Image transfers ARMSX2 records:
 *
 *  - vkCmdCopyImage, vkCmdCopyBufferToImage, vkCmdCopyImageToBuffer: raw
 *    block copies in a compute shader that addresses linear and
 *    u-interleaved surfaces itself (mali_meta_copy_get), on the compute
 *    subqueue like the buffer copies (the blob copies with compute too).
 *    Compressed images copy whole blocks.
 *  - vkCmdClearColorImage, vkCmdClearDepthStencilImage: a render pass per
 *    mip level that clears through the tile buffer and draws nothing, so
 *    only a fragment job runs (no tiler work, no heap-counter steps).
 *  - vkCmdBlitImage: a render pass on the destination whose frame shader
 *    samples the source inside the destination rectangle and reloads the
 *    destination outside it; one fragment job, no tiler work.
 *
 * AFBC images (render targets and depth buffers) have no raw addressing,
 * so copies that touch one go another way, as panvk's
 * (copy_to_image_use_gfx_pipeline, panvk_meta_copy_get_image_properties):
 *
 *  - into an AFBC plane: a render pass on it whose frame shader writes the
 *    source blocks as a uint format of the block size (the tile buffer
 *    compresses them); the source is read through the texture unit if it
 *    is AFBC too, else raw from memory;
 *  - out of an AFBC plane into anything else: the compute copy with the
 *    source read through the texture unit (texel fetch, same uint format).
 *
 * Depth and stencil planes are copied as colour (R32_UINT / R8_UINT), as
 * panvk does; the compression mode stays the plane's.
 *
 * Built per arch. On the job manager (v9) the transfer passes are marked
 * as fragment-side transfers of their batch (barriers from transfer stages
 * wait for them, mali_jm.h), and an image -> image copy whose compute job
 * would close the open batch (a barrier makes it wait for this batch's
 * fragment work, typically a copy of what a pass just rendered) is done by
 * the fragment-side copy pass instead, which stays in the batch behind a
 * barrier in its fragment chain.
 */

#if PAN_ARCH >= 10
#include "mali_cmd_buffer.h"
#else
#include "mali_jm.h"
#endif

#include <string.h>

#include "drm-uapi/drm_fourcc.h"
#include "util/format/u_format.h"
#include "util/u_math.h"
#include "vk_buffer.h"
#include "vk_format.h"
#include "vk_image.h"
#include "vk_log.h"

#include "mali_arch.h"
#include "mali_image.h"
#if PAN_ARCH >= 10
#include "mali_queue.h"
#endif
#include "mali_vk.h"

/* ---------------------------------------------------------------------- */
/* Surfaces                                                                */

struct copy_surface {
   uint64_t base;          /* block (0, 0) of the first layer */
   uint64_t layer_stride;
   uint32_t row_stride;
   uint32_t tile_shift;    /* 0 linear, 4 (texels) or 2 (compressed blocks) */
   unsigned block_w, block_h, block_bytes;
};

static unsigned
aspect_plane(const struct mali_image *img, VkImageAspectFlags aspect)
{
   return mali_image_aspect_plane(img, aspect);
}

static void
image_surface(const struct mali_image *img, VkImageAspectFlags aspect, unsigned level,
              unsigned layer, struct copy_surface *s)
{
   const unsigned plane = aspect_plane(img, aspect);
   const struct mali_image_plane *p = &img->planes[plane];
   const struct mali_image_slice *sl = &p->layout.slices[level];
   const enum pipe_format f = p->format;

   s->block_w = util_format_get_blockwidth(f);
   s->block_h = util_format_get_blockheight(f);
   s->block_bytes = util_format_get_blocksize(f);
   s->row_stride = sl->row_stride;
   if (img->vk.image_type == VK_IMAGE_TYPE_3D) {
      s->layer_stride = sl->surface_stride;
      s->base = img->base + sl->offset + (uint64_t)layer * sl->surface_stride;
   } else {
      s->layer_stride = p->layout.array_stride;
      s->base = img->base + sl->offset + (uint64_t)layer * p->layout.array_stride;
   }
   if (img->vk.drm_format_mod == DRM_FORMAT_MOD_LINEAR)
      s->tile_shift = 0;
   else
      s->tile_shift = util_format_is_compressed(f) ? 2 : 4;
}

/* The buffer side of a buffer/image copy: rows of blocks of the image
 * plane's format (the stencil aspect is one byte, depth four for D32). */
static void
buffer_surface(const struct vk_buffer *buf, const VkBufferImageCopy2 *r,
               const struct copy_surface *img, struct copy_surface *s)
{
   const uint32_t row_len = r->bufferRowLength ? r->bufferRowLength : r->imageExtent.width;
   const uint32_t img_h = r->bufferImageHeight ? r->bufferImageHeight : r->imageExtent.height;
   *s = *img;
   s->base = vk_buffer_address(buf, r->bufferOffset);
   s->tile_shift = 0;
   s->row_stride = DIV_ROUND_UP(row_len, img->block_w) * img->block_bytes;
   s->layer_stride = (uint64_t)DIV_ROUND_UP(img_h, img->block_h) * s->row_stride;
}

static void
dispatch_copy(struct mali_cmd_buffer *cmd, const struct copy_surface *src, int sx, int sy,
              const struct copy_surface *dst, int dx, int dy, uint32_t w_blocks,
              uint32_t h_blocks, uint32_t layers)
{
   if (!w_blocks || !h_blocks || !layers)
      return;
   const unsigned bytes = src->block_bytes;
   if (!util_is_power_of_two_nonzero(bytes) || bytes > 16) {
      vk_logw(VK_LOG_OBJS(cmd), "malisx2: copies of %u-byte blocks are not supported", bytes);
      return;
   }
   const struct mali_shader *s = MALI_PER_ARCH(meta_copy_get)(cmd, util_logbase2(bytes));
   if (!s)
      return;

   struct mali_meta_copy_push push = {
      .src = src->base,
      .dst = dst->base,
      .src_layer_stride = src->layer_stride,
      .dst_layer_stride = dst->layer_stride,
      .src_row_stride = src->row_stride,
      .dst_row_stride = dst->row_stride,
      .src_x = sx,
      .src_y = sy,
      .dst_x = dx,
      .dst_y = dy,
      .width = w_blocks,
      .height = h_blocks,
      .src_tile_shift = src->tile_shift,
      .dst_tile_shift = dst->tile_shift,
   };
   const uint32_t base[3] = {0, 0, 0};
   const uint32_t groups[3] = {DIV_ROUND_UP(w_blocks, MALI_META_COPY_WG),
                               DIV_ROUND_UP(h_blocks, MALI_META_COPY_WG), layers};
   MALI_PER_ARCH(cmd_dispatch_shader)(cmd, s, NULL, &push, sizeof(push), base, groups);
}

/* An internal render pass (image clear, blit, fragment-side copy). */
static void
transfer_pass(struct mali_cmd_buffer *cmd, const struct mali_render_desc *d)
{
   MALI_PER_ARCH(cmd_render_begin)(cmd, d);
   MALI_PER_ARCH(cmd_render_end)(cmd);
#if PAN_ARCH < 10
   mali_jm_cmd_mark_frag_transfer(cmd);
#endif
}

/* ---------------------------------------------------------------------- */
/* Copies that touch AFBC                                                  */

/* The uint format of a block size (vk_meta_get_uint_format_for_blk_size). */
static enum pipe_format
uint_format(unsigned bytes)
{
   switch (bytes) {
   case 1: return PIPE_FORMAT_R8_UINT;
   case 2: return PIPE_FORMAT_R16_UINT;
   case 4: return PIPE_FORMAT_R32_UINT;
   case 8: return PIPE_FORMAT_R32G32_UINT;
   default: return PIPE_FORMAT_R32G32B32A32_UINT;
   }
}

/* How a copy reads a plane through the texture unit: its block as uint;
 * stencil as stencil (the stencil preload reads it the same way). */
static enum pipe_format
copy_read_format(const struct mali_image *img, unsigned plane)
{
   const enum pipe_format f = img->planes[plane].format;
   if (f == PIPE_FORMAT_S8_UINT)
      return f;
   return uint_format(util_format_get_blocksize(f));
}

/* The source of a copy into an AFBC plane: an AFBC image plane (read as a
 * texture) or a raw surface. Block coordinates. */
struct copy_src {
   const struct mali_image *img;    /* AFBC source, or NULL for raw */
   unsigned plane, level, layer;
   struct copy_surface raw;
   int x, y;
};

static bool
rects_overlap(int ax0, int ay0, int ax1, int ay1, int bx0, int by0, int bx1, int by1)
{
   return ax0 < bx1 && bx0 < ax1 && ay0 < by1 && by0 < ay1;
}

static void copy_from_afbc(struct mali_cmd_buffer *cmd, const struct mali_image *img,
                           unsigned plane, unsigned level, unsigned layer, int sx, int sy,
                           const struct copy_surface *dst, int dx, int dy, uint32_t w,
                           uint32_t h, uint32_t layers);

/*
 * Blocks [dx, dx + w) x [dy, dy + h) of layers [layer, layer + layers) of an
 * AFBC plane from `src`: a render pass over the rectangle (grown to whole
 * superblocks, the rest reloaded) with the COPY frame shader. On v9 also
 * the fragment-side copy into any colour plane (copy_on_fragment).
 */
static void
copy_to_afbc(struct mali_cmd_buffer *cmd, const struct mali_image *img, unsigned plane,
             unsigned level, unsigned layer, int dx, int dy, uint32_t w, uint32_t h,
             uint32_t layers, const struct copy_src *src)
{
   if (!w || !h || !layers)
      return;
   const unsigned elem = util_format_get_blocksize(img->planes[plane].format);

   /* A copy inside one image whose source shares superblocks with what the
    * pass writes would read blocks while other tiles rewrite them: go
    * through memory. */
   if (src->img == img && src->plane == plane && src->level == level &&
       rects_overlap(ROUND_DOWN_TO(src->x, 16), ROUND_DOWN_TO(src->y, 16),
                     ALIGN_POT(src->x + w, 16), ALIGN_POT(src->y + h, 16),
                     ROUND_DOWN_TO(dx, 16), ROUND_DOWN_TO(dy, 16), ALIGN_POT(dx + w, 16),
                     ALIGN_POT(dy + h, 16)) &&
       src->layer < layer + layers && layer < src->layer + layers) {
      const uint64_t row = (uint64_t)w * elem, size = row * h * layers;
      struct mali_ptr tmp = mali_cmd_alloc(cmd, size, 64);
      if (!tmp.cpu)
         return;
      struct copy_src bounce = {
         .raw = {.base = tmp.gpu, .layer_stride = row * h, .row_stride = row,
                 .tile_shift = 0, .block_w = 1, .block_h = 1, .block_bytes = elem},
      };
      copy_from_afbc(cmd, src->img, src->plane, src->level, src->layer, src->x, src->y,
                     &bounce.raw, 0, 0, w, h, layers);
#if PAN_ARCH < 10
      mali_jm_cmd_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                          VK_ACCESS_2_SHADER_READ_BIT);
#else
      const VkMemoryBarrier2 mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
         .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
      };
      const VkDependencyInfo dep = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &mb,
      };
      struct mali_cs_deps deps = {0};
      mali_cmd_add_deps(cmd, &dep, &deps);
      mali_cmd_emit_barrier(cmd, &deps);
#endif
      copy_to_afbc(cmd, img, plane, level, layer, dx, dy, w, h, layers, &bounce);
      return;
   }

   struct mali_meta_fs_key key = {
      .layered = layers > 1,
      .copy_src_tex = src->img != NULL,
      .copy_elem_log2 = util_logbase2(elem),
   };
   key.rt_op[0] = MALI_META_FS_COPY;
   key.rt_type[0] = MALI_META_FS_UINT;

   struct mali_meta_fs_push push = {
      .area = {dx, dy, dx + (int32_t)w - 1, dy + (int32_t)h - 1},
      .copy_delta = {src->x - dx, src->y - dy},
      .copy_row_stride = src->raw.row_stride,
      .copy_tile_shift = src->raw.tile_shift,
      .copy_src = src->raw.base,
      .copy_layer_stride = src->raw.layer_stride,
   };

   uint32_t tex[2][8];
   unsigned ntex = 0;
   if (src->img &&
       !MALI_PER_ARCH(cmd_pack_plane_texture)(cmd, src->img, src->plane,
                                              copy_read_format(src->img, src->plane), src->level,
                                              src->layer, layers, tex[ntex++]))
      goto oom;
   if (!MALI_PER_ARCH(cmd_pack_plane_texture)(cmd, img, plane, copy_read_format(img, plane), level,
                                              layer, layers, tex[ntex++]))
      goto oom;

   const struct mali_render_blit blit = {
      .key = &key,
      .push = &push,
      .textures = (const uint32_t(*)[8])tex,
      .texture_count = ntex,
   };
   struct mali_render_desc d = {
      .width = u_minify(img->vk.extent.width, level),
      .height = u_minify(img->vk.extent.height, level),
      .area = {{dx, dy}, {w, h}},
      .layer_count = layers,
      .rt_count = 1,
      .blit = &blit,
   };
   d.rt[0] = (struct mali_fb_target){
      .image = img,
      .plane = plane,
      .level = level,
      .layer = layer,
      .format = uint_format(elem),
      .load = MALI_ATT_LOAD_DONT_CARE,
      .store = true,
      .always_write = true,
   };
   transfer_pass(cmd, &d);
   return;

oom:
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
}

/* Blocks of an AFBC plane into a raw surface: the compute copy reading
 * through the texture unit. */
static void
copy_from_afbc(struct mali_cmd_buffer *cmd, const struct mali_image *img, unsigned plane,
               unsigned level, unsigned layer, int sx, int sy, const struct copy_surface *dst,
               int dx, int dy, uint32_t w, uint32_t h, uint32_t layers)
{
   if (!w || !h || !layers)
      return;
   const unsigned elem = util_format_get_blocksize(img->planes[plane].format);
   const struct mali_shader *s = MALI_PER_ARCH(meta_copy_tex_get)(cmd, util_logbase2(elem));
   if (!s)
      return;

   uint32_t tex[8];
   if (!MALI_PER_ARCH(cmd_pack_plane_texture)(cmd, img, plane, copy_read_format(img, plane), level,
                                              layer, layers, tex)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   const struct mali_meta_copy_push push = {
      .dst = dst->base,
      .dst_layer_stride = dst->layer_stride,
      .dst_row_stride = dst->row_stride,
      .src_x = sx,
      .src_y = sy,
      .dst_x = dx,
      .dst_y = dy,
      .width = w,
      .height = h,
      .dst_tile_shift = dst->tile_shift,
   };
   const uint32_t groups[3] = {DIV_ROUND_UP(w, MALI_META_COPY_WG),
                               DIV_ROUND_UP(h, MALI_META_COPY_WG), layers};
   MALI_PER_ARCH(cmd_dispatch_meta)(cmd, s, &push, sizeof(push), groups,
                          (const uint32_t(*)[8])tex, 1);
}

static uint32_t
subresource_layers(const struct vk_image *img, const VkImageSubresourceLayers *sr,
                   const VkExtent3D *extent)
{
   if (img->image_type == VK_IMAGE_TYPE_3D)
      return extent->depth;
   return vk_image_subresource_layer_count(img, sr);
}

/* A copy into level 0 of an image with CRC state writes it behind the
 * CRCs' back (panvk copy_image_touches_crc): bump the seed once per copy
 * command. */
static bool
copy_touches_crc(const struct mali_image *img, const VkImageSubresourceLayers *sr)
{
   return img->crc_header && sr->mipLevel == 0 && (sr->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) &&
          !mali_crc_options.skip_copy_invalidate;
}

/*
 * On v9: copy this colour region with the fragment-side copy pass rather
 * than the compute copy, because the compute job would close the open
 * batch. Uncompressed 2D colour planes only (the pass renders the plane as
 * a uint format of its texel size); anything else stays compute.
 */
static bool
copy_on_fragment(struct mali_cmd_buffer *cmd, const struct mali_image *src,
                 const struct mali_image *dst, VkImageAspectFlags aspect,
                 const struct copy_surface *s)
{
#if PAN_ARCH < 10
   return aspect == VK_IMAGE_ASPECT_COLOR_BIT && src->vk.image_type == VK_IMAGE_TYPE_2D &&
          dst->vk.image_type == VK_IMAGE_TYPE_2D && s->block_w == 1 && s->block_h == 1 &&
          util_is_power_of_two_nonzero(s->block_bytes) && s->block_bytes <= 16 &&
          mali_jm_cmd_transfer_needs_frag(cmd);
#else
   return false;
#endif
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdCopyImage2)(VkCommandBuffer commandBuffer, const VkCopyImageInfo2 *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_image, src, info->srcImage);
   VK_FROM_HANDLE(mali_image, dst, info->dstImage);

   for (uint32_t i = 0; i < info->regionCount; i++) {
      if (copy_touches_crc(dst, &info->pRegions[i].dstSubresource)) {
         MALI_PER_ARCH(cmd_crc_invalidate)(cmd, dst);
         break;
      }
   }

   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageCopy2 *r = &info->pRegions[i];
      u_foreach_bit(a, r->srcSubresource.aspectMask) {
         const VkImageAspectFlags aspect = BITFIELD_BIT(a);
         /* Multi-aspect copies pair the aspects in order. */
         const VkImageAspectFlags daspect =
            r->dstSubresource.aspectMask == r->srcSubresource.aspectMask ?
               aspect : r->dstSubresource.aspectMask;
         const bool src3d = src->vk.image_type == VK_IMAGE_TYPE_3D;
         const bool dst3d = dst->vk.image_type == VK_IMAGE_TYPE_3D;
         struct copy_surface s, d;
         image_surface(src, aspect, r->srcSubresource.mipLevel,
                       src3d ? r->srcOffset.z : r->srcSubresource.baseArrayLayer, &s);
         image_surface(dst, daspect, r->dstSubresource.mipLevel,
                       dst3d ? r->dstOffset.z : r->dstSubresource.baseArrayLayer, &d);
         const uint32_t layers = subresource_layers(&src->vk, &r->srcSubresource, &r->extent);
         /* The extent is in source texels; both sides count blocks of the
          * same size (size-compatible formats). */
         const int sx = r->srcOffset.x / s.block_w, sy = r->srcOffset.y / s.block_h;
         const int dx = r->dstOffset.x / d.block_w, dy = r->dstOffset.y / d.block_h;
         const uint32_t w = DIV_ROUND_UP(r->extent.width, s.block_w);
         const uint32_t h = DIV_ROUND_UP(r->extent.height, s.block_h);
         const unsigned splane = aspect_plane(src, aspect);
         const unsigned dplane = aspect_plane(dst, daspect);
         if (mali_image_is_afbc(dst) || copy_on_fragment(cmd, src, dst, aspect, &s)) {
            const struct copy_src cs = {
               .img = mali_image_is_afbc(src) ? src : NULL,
               .plane = splane,
               .level = r->srcSubresource.mipLevel,
               .layer = r->srcSubresource.baseArrayLayer,
               .raw = s,
               .x = sx,
               .y = sy,
            };
            copy_to_afbc(cmd, dst, dplane, r->dstSubresource.mipLevel,
                         r->dstSubresource.baseArrayLayer, dx, dy, w, h, layers, &cs);
         } else if (mali_image_is_afbc(src)) {
            copy_from_afbc(cmd, src, splane, r->srcSubresource.mipLevel,
                           r->srcSubresource.baseArrayLayer, sx, sy, &d, dx, dy, w, h,
                           layers);
         } else {
            dispatch_copy(cmd, &s, sx, sy, &d, dx, dy, w, h, layers);
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdCopyBufferToImage2)(VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2 *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, buf, info->srcBuffer);
   VK_FROM_HANDLE(mali_image, img, info->dstImage);

   for (uint32_t i = 0; i < info->regionCount; i++) {
      if (copy_touches_crc(img, &info->pRegions[i].imageSubresource)) {
         MALI_PER_ARCH(cmd_crc_invalidate)(cmd, img);
         break;
      }
   }

   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkBufferImageCopy2 *r = &info->pRegions[i];
      const bool is3d = img->vk.image_type == VK_IMAGE_TYPE_3D;
      struct copy_surface d, s;
      image_surface(img, r->imageSubresource.aspectMask, r->imageSubresource.mipLevel,
                    is3d ? r->imageOffset.z : r->imageSubresource.baseArrayLayer, &d);
      buffer_surface(buf, r, &d, &s);
      const int dx = r->imageOffset.x / d.block_w, dy = r->imageOffset.y / d.block_h;
      const uint32_t w = DIV_ROUND_UP(r->imageExtent.width, d.block_w);
      const uint32_t h = DIV_ROUND_UP(r->imageExtent.height, d.block_h);
      const uint32_t layers = subresource_layers(&img->vk, &r->imageSubresource, &r->imageExtent);
      if (mali_image_is_afbc(img)) {
         const struct copy_src cs = {.raw = s, .x = 0, .y = 0};
         copy_to_afbc(cmd, img, aspect_plane(img, r->imageSubresource.aspectMask),
                      r->imageSubresource.mipLevel, r->imageSubresource.baseArrayLayer, dx,
                      dy, w, h, layers, &cs);
      } else {
         dispatch_copy(cmd, &s, 0, 0, &d, dx, dy, w, h, layers);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdCopyImageToBuffer2)(VkCommandBuffer commandBuffer, const VkCopyImageToBufferInfo2 *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_image, img, info->srcImage);
   VK_FROM_HANDLE(vk_buffer, buf, info->dstBuffer);

   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkBufferImageCopy2 *r = &info->pRegions[i];
      const bool is3d = img->vk.image_type == VK_IMAGE_TYPE_3D;
      struct copy_surface s, d;
      image_surface(img, r->imageSubresource.aspectMask, r->imageSubresource.mipLevel,
                    is3d ? r->imageOffset.z : r->imageSubresource.baseArrayLayer, &s);
      buffer_surface(buf, r, &s, &d);
      const int sx = r->imageOffset.x / s.block_w, sy = r->imageOffset.y / s.block_h;
      const uint32_t w = DIV_ROUND_UP(r->imageExtent.width, s.block_w);
      const uint32_t h = DIV_ROUND_UP(r->imageExtent.height, s.block_h);
      const uint32_t layers = subresource_layers(&img->vk, &r->imageSubresource, &r->imageExtent);
      if (mali_image_is_afbc(img))
         copy_from_afbc(cmd, img, aspect_plane(img, r->imageSubresource.aspectMask),
                        r->imageSubresource.mipLevel, r->imageSubresource.baseArrayLayer,
                        sx, sy, &d, 0, 0, w, h, layers);
      else
         dispatch_copy(cmd, &s, sx, sy, &d, 0, 0, w, h, layers);
   }
}

/* ---------------------------------------------------------------------- */
/* Clears                                                                  */

static uint32_t
level_layers(const struct mali_image *img, const VkImageSubresourceRange *range,
             unsigned level)
{
   if (img->vk.image_type == VK_IMAGE_TYPE_3D)
      return u_minify(img->vk.extent.depth, level);
   return vk_image_subresource_layer_count(&img->vk, range);
}

static void
clear_pass(struct mali_cmd_buffer *cmd, struct mali_render_desc *d, const struct mali_image *img,
           unsigned level)
{
   d->width = u_minify(img->vk.extent.width, level);
   d->height = u_minify(img->vk.extent.height, level);
   d->area = (VkRect2D){{0, 0}, {d->width, d->height}};
   transfer_pass(cmd, d);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdClearColorImage)(VkCommandBuffer commandBuffer, VkImage image,
                                  VkImageLayout imageLayout, const VkClearColorValue *pColor,
                                  uint32_t rangeCount, const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_image, img, image);
   const enum pipe_format format = img->planes[0].format;

   if (cmd->gfx.render.active)
      return;
   for (uint32_t i = 0; i < rangeCount; i++) {
      const VkImageSubresourceRange *r = &pRanges[i];
      const uint32_t levels = vk_image_subresource_level_count(&img->vk, r);
      for (uint32_t l = 0; l < levels; l++) {
         const unsigned level = r->baseMipLevel + l;
         struct mali_render_desc d = {
            .layer_count = level_layers(img, r, level),
            .rt_count = 1,
         };
         d.rt[0] = (struct mali_fb_target){
            .image = img,
            .plane = 0,
            .level = level,
            .layer = img->vk.image_type == VK_IMAGE_TYPE_3D ? 0 : r->baseArrayLayer,
            .format = format,
            .load = MALI_ATT_LOAD_CLEAR,
            .store = true,
         };
         d.clear_color[0] = *pColor;
         clear_pass(cmd, &d, img, level);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdClearDepthStencilImage)(VkCommandBuffer commandBuffer, VkImage image,
                                         VkImageLayout imageLayout,
                                         const VkClearDepthStencilValue *pDepthStencil,
                                         uint32_t rangeCount, const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_image, img, image);

   if (cmd->gfx.render.active)
      return;
   for (uint32_t i = 0; i < rangeCount; i++) {
      const VkImageSubresourceRange *r = &pRanges[i];
      const uint32_t levels = vk_image_subresource_level_count(&img->vk, r);
      for (uint32_t l = 0; l < levels; l++) {
         const unsigned level = r->baseMipLevel + l;
         struct mali_render_desc d = {
            .layer_count = level_layers(img, r, level),
            .clear_depth = pDepthStencil->depth,
            .clear_stencil = pDepthStencil->stencil,
         };
         const struct mali_fb_target base = {
            .image = img,
            .level = level,
            .layer = r->baseArrayLayer,
            .load = MALI_ATT_LOAD_CLEAR,
            .store = true,
         };
         if ((r->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) && vk_format_has_depth(img->vk.format)) {
            d.z = base;
            d.z.plane = aspect_plane(img, VK_IMAGE_ASPECT_DEPTH_BIT);
            d.z.format = img->planes[d.z.plane].format;
         }
         if ((r->aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) &&
             vk_format_has_stencil(img->vk.format)) {
            d.s = base;
            d.s.plane = aspect_plane(img, VK_IMAGE_ASPECT_STENCIL_BIT);
            d.s.format = img->planes[d.s.plane].format;
         }
         clear_pass(cmd, &d, img, level);
      }
   }
}

/* ---------------------------------------------------------------------- */
/* Blits                                                                   */

static enum mali_meta_fs_type
fs_type(enum pipe_format f)
{
   if (util_format_is_pure_uint(f))
      return MALI_META_FS_UINT;
   if (util_format_is_pure_sint(f))
      return MALI_META_FS_SINT;
   return MALI_META_FS_FLOAT;
}

/* dst pixel centre d maps to source texel coordinate
 * s0 + (d - d0) * (s1 - s0) / (d1 - d0); normalized by the source size. */
static void
blit_axis(int32_t s0, int32_t s1, int32_t d0, int32_t d1, uint32_t size, float *scale,
          float *offset)
{
   const float k = (float)(s1 - s0) / (float)(d1 - d0);
   *scale = k / (float)size;
   *offset = ((float)s0 - (float)d0 * k) / (float)size;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdBlitImage2)(VkCommandBuffer commandBuffer, const VkBlitImageInfo2 *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_image, src, info->srcImage);
   VK_FROM_HANDLE(mali_image, dst, info->dstImage);

   if (cmd->gfx.render.active)
      return;

   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageBlit2 *r = &info->pRegions[i];
      if (!(r->dstSubresource.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT)) {
         vk_logw(VK_LOG_OBJS(cmd), "malisx2: depth/stencil blits are not supported");
         continue;
      }
      const unsigned slevel = r->srcSubresource.mipLevel;
      const unsigned dlevel = r->dstSubresource.mipLevel;
      const uint32_t layers = vk_image_subresource_layer_count(&dst->vk, &r->dstSubresource);
      const enum pipe_format sfmt = src->planes[0].format;
      const enum pipe_format dfmt = dst->planes[0].format;

      const int32_t dx0 = MIN2(r->dstOffsets[0].x, r->dstOffsets[1].x);
      const int32_t dx1 = MAX2(r->dstOffsets[0].x, r->dstOffsets[1].x);
      const int32_t dy0 = MIN2(r->dstOffsets[0].y, r->dstOffsets[1].y);
      const int32_t dy1 = MAX2(r->dstOffsets[0].y, r->dstOffsets[1].y);
      if (dx1 <= dx0 || dy1 <= dy0)
         continue;

      struct mali_meta_fs_key key = {
         .layered = layers > 1,
      };
      key.rt_op[0] = MALI_META_FS_BLIT;
      key.rt_type[0] = fs_type(dfmt);

      struct mali_meta_fs_push push = {
         .area = {dx0, dy0, dx1 - 1, dy1 - 1},
      };
      blit_axis(r->srcOffsets[0].x, r->srcOffsets[1].x, r->dstOffsets[0].x,
                r->dstOffsets[1].x, u_minify(src->vk.extent.width, slevel),
                &push.blit_scale[0], &push.blit_offset[0]);
      blit_axis(r->srcOffsets[0].y, r->srcOffsets[1].y, r->dstOffsets[0].y,
                r->dstOffsets[1].y, u_minify(src->vk.extent.height, slevel),
                &push.blit_scale[1], &push.blit_offset[1]);

      uint32_t tex[2][8];
      if (!MALI_PER_ARCH(cmd_pack_plane_texture)(cmd, src, 0, sfmt, slevel,
                                                 r->srcSubresource.baseArrayLayer, layers,
                                                 tex[0]) ||
          !MALI_PER_ARCH(cmd_pack_plane_texture)(cmd, dst, 0, dfmt, dlevel,
                                                 r->dstSubresource.baseArrayLayer, layers,
                                                 tex[1])) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
      uint32_t sampler[8];
      const bool nearest = info->filter == VK_FILTER_NEAREST;
      pan_cast_and_pack(sampler, SAMPLER, cfg) {
         cfg.seamless_cube_map = false;
         cfg.normalized_coordinates = true;
         cfg.minify_nearest = nearest;
         cfg.magnify_nearest = nearest;
         cfg.clamp_integer_array_indices = true;
      }

      const struct mali_render_blit blit = {
         .key = &key,
         .push = &push,
         .textures = (const uint32_t(*)[8])tex,
         .texture_count = 2,
         .sampler = sampler,
      };
      struct mali_render_desc d = {
         .width = u_minify(dst->vk.extent.width, dlevel),
         .height = u_minify(dst->vk.extent.height, dlevel),
         .area = {{dx0, dy0}, {(uint32_t)(dx1 - dx0), (uint32_t)(dy1 - dy0)}},
         .layer_count = layers,
         .rt_count = 1,
         .blit = &blit,
      };
      d.rt[0] = (struct mali_fb_target){
         .image = dst,
         .plane = 0,
         .level = dlevel,
         .layer = r->dstSubresource.baseArrayLayer,
         .format = dfmt,
         .load = MALI_ATT_LOAD_DONT_CARE,
         .store = true,
         .always_write = true,
      };
      transfer_pass(cmd, &d);
   }
}
