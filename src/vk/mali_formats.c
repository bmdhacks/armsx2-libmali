/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Format support: vkGetPhysicalDeviceFormatProperties(2) and
 * vkGetPhysicalDeviceImageFormatProperties(2).
 *
 * The answers come from Mesa's format table for our architecture
 * (pan_format.c, vendored) and the GPU's compressed-format bits
 * (TEXTURE_FEATURES_0), derived the way panvk does
 * (panvk_physical_device.c), because the descriptors that use the formats
 * are packed from the same table. Where the blob's answer differs and the
 * blob's is the safer one we follow it: integer formats get no blend or
 * linear-filter bit, and combined depth/stencil formats have no
 * linear-tiling features. YCbCr formats, sparse images, protected images
 * and external memory are not supported.
 */

#define PAN_ARCH MALI_PAN_ARCH

#include "mali_vk.h"
#include "mali_image.h"

#include "pan_format.h"

#include "util/format/u_format.h"
#include "util/log.h"
#include "vk_format.h"
#include "vk_util.h"

static const struct pan_format *
pan_fmt(enum pipe_format p)
{
   return GENX(pan_format_from_pipe_format)(p);
}

/* Compressed formats are optional in the hardware: TEXTURE_FEATURES_0 has
 * one bit per compressed texture format (pan_query_compressed_formats). */
static uint32_t
compressed_format_bits(const struct mali_physical_device *pdev)
{
   return pdev->props.texture_features[0] ? pdev->props.texture_features[0]
                                          : pdev->props.raw_texture_features[0];
}

static bool
pipe_format_supported(const struct mali_physical_device *pdev, enum pipe_format p)
{
   if (p == PIPE_FORMAT_NONE || !pan_fmt(p)->hw)
      return false;
   /* No YCbCr conversion support (and Mesa's subsampled RGB formats are
    * YUV to the hardware). */
   if (pan_format_is_yuv(p))
      return false;
   if (util_format_is_compressed(p) &&
       !(compressed_format_bits(pdev) & BITFIELD_BIT(pan_fmt(p)->texfeat_bit)))
      return false;
   return true;
}

unsigned
mali_format_planes(VkFormat format, enum pipe_format planes[MALI_IMAGE_MAX_PLANES])
{
   switch (format) {
   /* Combined depth/stencil in two planes, as panvk does on arch 9 and
    * later (Z32_S8X24 does not exist there) and as the blob keeps stencil
    * in a second slice array. */
   case VK_FORMAT_D32_SFLOAT_S8_UINT:
      planes[0] = PIPE_FORMAT_Z32_FLOAT;
      planes[1] = PIPE_FORMAT_S8_UINT;
      return 2;
   case VK_FORMAT_D24_UNORM_S8_UINT:
      planes[0] = PIPE_FORMAT_Z24X8_UNORM;
      planes[1] = PIPE_FORMAT_S8_UINT;
      return 2;
   default:
      break;
   }

   if (vk_format_get_plane_count(format) != 1)
      return 0;
   planes[0] = vk_format_to_pipe_format(format);
   return planes[0] == PIPE_FORMAT_NONE ? 0 : 1;
}

VkFormatFeatureFlags
mali_format_image_features(const struct mali_physical_device *pdev, VkFormat format)
{
   enum pipe_format planes[MALI_IMAGE_MAX_PLANES];
   const unsigned nplanes = mali_format_planes(format, planes);
   if (!nplanes)
      return 0;
   for (unsigned i = 0; i < nplanes; i++)
      if (!pipe_format_supported(pdev, planes[i]))
         return 0;

   const enum pipe_format p = vk_format_to_pipe_format(format);
   if (!pipe_format_supported(pdev, p))
      return 0;

   const struct pan_format *f = pan_fmt(p);
   const bool integer = util_format_is_pure_integer(p);
   VkFormatFeatureFlags feat = 0;

   if (f->bind & PAN_BIND_SAMPLER_VIEW) {
      feat |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT |
              VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
      /* Integer and scaled formats filter nearest only. */
      if (!integer && !util_format_is_scaled(p))
         feat |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
   }

   if (f->bind & PAN_BIND_RENDER_TARGET) {
      feat |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
      /* The blob reports no blending for integer formats (R32_UINT is
       * 0xcc87); Vulkan never blends them. */
      if (!integer)
         feat |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
   }

   if (f->bind & PAN_BIND_STORAGE_IMAGE) {
      feat |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
      if (p == PIPE_FORMAT_R32_UINT || p == PIPE_FORMAT_R32_SINT)
         feat |= VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT;
   }

   if (f->bind & PAN_BIND_DEPTH_STENCIL)
      feat |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;

   return feat;
}

/* Linear images of a combined depth/stencil format are not offered (the
 * blob reports optimal tiling only for them). */
static VkFormatFeatureFlags
linear_features(const struct mali_physical_device *pdev, VkFormat format)
{
   enum pipe_format planes[MALI_IMAGE_MAX_PLANES];
   if (mali_format_planes(format, planes) > 1)
      return 0;
   return mali_format_image_features(pdev, format);
}

VkFormatFeatureFlags
mali_format_buffer_features(const struct mali_physical_device *pdev, VkFormat format)
{
   const enum pipe_format p = vk_format_to_pipe_format(format);
   if (vk_format_get_plane_count(format) != 1 || !pipe_format_supported(pdev, p))
      return 0;

   const struct pan_format *f = pan_fmt(p);
   VkFormatFeatureFlags feat = 0;

   /* sRGB vertex formats are rejected, as panvk does (Vulkan-Docs issue
    * 2214). */
   if ((f->bind & PAN_BIND_VERTEX_BUFFER) && !util_format_is_srgb(p))
      feat |= VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT;
   if (f->bind & PAN_BIND_TEXEL_BUFFER)
      feat |= VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT |
              VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT;
   if ((f->bind & PAN_BIND_TEXEL_BUFFER) &&
       (p == PIPE_FORMAT_R32_UINT || p == PIPE_FORMAT_R32_SINT))
      feat |= VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
   return feat;
}

VKAPI_ATTR void VKAPI_CALL
mali_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice, VkFormat format,
                                        VkFormatProperties2 *pFormatProperties)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, physicalDevice);

   pFormatProperties->formatProperties = (VkFormatProperties) {
      .linearTilingFeatures = linear_features(pdev, format),
      .optimalTilingFeatures = mali_format_image_features(pdev, format),
      .bufferFeatures = mali_format_buffer_features(pdev, format),
   };

   /* No extension structure we know is filled: FormatProperties3 and DRM
    * format modifier lists come with extensions we do not expose. */
   vk_foreach_struct(stype, ext, pFormatProperties->pNext)
      vk_debug_ignored_stype(stype);
}

static VkResult
image_format_properties(const struct mali_physical_device *pdev,
                        const VkPhysicalDeviceImageFormatInfo2 *info,
                        VkImageFormatProperties *props)
{
   const VkImageStencilUsageCreateInfo *stencil_usage_info =
      vk_find_struct_const(info->pNext, IMAGE_STENCIL_USAGE_CREATE_INFO);
   const VkImageUsageFlags usage =
      info->usage | (stencil_usage_info ? stencil_usage_info->stencilUsage : 0);

   if (info->flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                      VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
                      VK_IMAGE_CREATE_SPARSE_ALIASED_BIT |
                      VK_IMAGE_CREATE_PROTECTED_BIT |
                      VK_IMAGE_CREATE_DISJOINT_BIT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   VkFormatFeatureFlags feat;
   switch (info->tiling) {
   case VK_IMAGE_TILING_OPTIMAL:
      feat = mali_format_image_features(pdev, info->format);
      break;
   case VK_IMAGE_TILING_LINEAR:
      feat = linear_features(pdev, info->format);
      break;
   default:
      /* DRM format modifiers are not exposed. */
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }
   if (!feat)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   const struct vk_properties *lim = &pdev->vk.properties;
   VkExtent3D extent;
   uint32_t layers;
   switch (info->type) {
   case VK_IMAGE_TYPE_1D:
      extent = (VkExtent3D){lim->maxImageDimension1D, 1, 1};
      layers = lim->maxImageArrayLayers;
      break;
   case VK_IMAGE_TYPE_2D:
      extent = (VkExtent3D){lim->maxImageDimension2D, lim->maxImageDimension2D, 1};
      layers = lim->maxImageArrayLayers;
      break;
   case VK_IMAGE_TYPE_3D:
      extent = (VkExtent3D){lim->maxImageDimension3D, lim->maxImageDimension3D,
                            lim->maxImageDimension3D};
      layers = 1;
      break;
   default:
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }

   /* Unless EXTENDED_USAGE lets views use other formats, every usage needs
    * its format feature (as panvk). */
   if (!(info->flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT)) {
      const bool ds = vk_format_is_depth_or_stencil(info->format);
      if ((usage & VK_IMAGE_USAGE_SAMPLED_BIT) && !(feat & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) && !(feat & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      if ((usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) &&
          !(feat & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      if ((usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) &&
          !(feat & VK_FORMAT_FEATURE_TRANSFER_DST_BIT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      if (((usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
           ((usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) && !ds)) &&
          !(feat & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      if (((usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ||
           ((usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) && ds)) &&
          !(feat & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }

   /* Multisampling: optimal 2D attachments, not cube, not storage (panvk's
    * condition), with the blob's per-size rule. */
   VkSampleCountFlags samples = VK_SAMPLE_COUNT_1_BIT;
   if (info->tiling == VK_IMAGE_TILING_OPTIMAL && info->type == VK_IMAGE_TYPE_2D &&
       (feat & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
       !(info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
       !(usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
      enum pipe_format planes[MALI_IMAGE_MAX_PLANES];
      mali_format_planes(info->format, planes);
      samples = mali_physical_device_sample_counts(pdev, util_format_get_blocksize(planes[0]));
   }

   *props = (VkImageFormatProperties) {
      .maxExtent = extent,
      .maxMipLevels = MALI_IMAGE_MAX_LEVELS,
      .maxArrayLayers = layers,
      .sampleCounts = samples,
      /* An image has to fit in one allocation. */
      .maxResourceSize = pdev->vk.properties.maxMemoryAllocationSize,
   };
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetPhysicalDeviceImageFormatProperties2(VkPhysicalDevice physicalDevice,
                                             const VkPhysicalDeviceImageFormatInfo2 *pInfo,
                                             VkImageFormatProperties2 *pProps)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, physicalDevice);

   /*
    * Seam for the Android WSI: external memory (the loader's AHB queries)
    * is not supported yet, so an image that asks for a handle type is not
    * supported.
    */
   const VkPhysicalDeviceExternalImageFormatInfo *ext_info =
      vk_find_struct_const(pInfo->pNext, PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
   VkResult result = ext_info && ext_info->handleType ?
                        VK_ERROR_FORMAT_NOT_SUPPORTED :
                        image_format_properties(pdev, pInfo, &pProps->imageFormatProperties);
   if (result != VK_SUCCESS) {
      pProps->imageFormatProperties = (VkImageFormatProperties){0};
      return result;
   }

   vk_foreach_struct(stype, ext, pProps->pNext) {
      switch (stype) {
      case VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES: {
         VkExternalImageFormatProperties *e = ext;
         e->externalMemoryProperties = (VkExternalMemoryProperties){0};
         break;
      }
      case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES: {
         VkSamplerYcbcrConversionImageFormatProperties *y = ext;
         y->combinedImageSamplerDescriptorCount = 1;
         break;
      }
      default:
         vk_debug_ignored_stype(stype);
         break;
      }
   }
   return VK_SUCCESS;
}
