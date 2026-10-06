/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkPhysicalDevice: one per kbase device node. What we report matches
 * the blob's values, including the memory types.
 */

#include "mali_queue.h"
#include "mali_vk.h"
#include "mali_shader.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "util/log.h"
#include "util/macros.h"
#include "util/mesa-blake3.h"
#include "util/u_math.h"
#include "vk_alloc.h"
#include "vk_limits.h"
#include "vk_util.h"

#include "mali_version.h"
#include "mali_shader_cache_id.h"

/* Arm's PCI-style vendor ID, as the blob reports it. */
#define MALI_VENDOR_ID 0x13b5

/* The r44p1 blob's driverVersion, 44.1.0 in VK_MAKE_VERSION layout.
 * ARMSX2's Arm driver rules match on exactly this. */
#define MALI_DRIVER_VERSION VK_MAKE_VERSION(44, 1, 0)

/* driverInfo keeps the blob's "v1.r44p1-" prefix (ARMSX2 looks for
 * "r44p1"; the blob's own version parse still yields 44.1) and then says
 * what we are. It ends in the shader cache ID, not the commit: ARMSX2 keys
 * its pipeline cache file to driverInfo as well as pipelineCacheUUID, so a
 * commit here would empty the cache on every driver update. The commit is
 * logged when the physical device is created instead. */
#define MALI_DRIVER_INFO \
   "v1.r44p1-libmali." MALI_VERSION_STRING ".s" MALI_SHADER_CACHE_ID_SHORT

/*
 * What identity the G57 build (arch 9) reports to ARMSX2: the
 * working choice is to report the same scheme the G615 build uses
 * above, not the T820 blob's own values (driverVersion 40.0.0,
 * driverInfo "v1.r40p0-01eac0.<hash>"). Either way driverID is 9
 * (VK_DRIVER_ID_ARM_PROPRIETARY), so ARMSX2 treats both the same
 * way for that check. Switching to the blob's own values instead
 * of the G615 scheme is a one-line change to these two
 * definitions.
 */
#define MALI_V9_DRIVER_VERSION MALI_DRIVER_VERSION
/* The same scheme, ending in the v9 build's own shader cache ID. */
#define MALI_V9_DRIVER_INFO \
   "v1.r44p1-libmali." MALI_VERSION_STRING ".s" MALI_SHADER_CACHE_ID_V9_SHORT

/* Arch the driver was built for (meson.build pan_arch). */
#ifndef MALI_PAN_ARCH
#error "MALI_PAN_ARCH must be defined"
#endif

/* VkFence and binary VkSemaphore (mali_sync.c, built once per arch). */
static const struct vk_sync_type *const mali_v11_sync_types[] = { &mali_v11_sync_type, NULL };
static const struct vk_sync_type *const mali_v9_sync_types[] = { &mali_v9_sync_type, NULL };

void
mali_kbase_log_to_vk(void *user, const char *msg)
{
   mesa_logw("libmali: kbase: %s", msg);
}

/* ---------------------------------------------------------------------- */
/* Identity                                                                */

bool
mali_device_name(const struct mali_kbase_gpu_props *p, char *buf, size_t size)
{
   /* Arch 9: product 0x9001 is the G57, the only one the
    * T820 blob accepts; 0x9003 is Mesa's other G57 product
    * ID (the T820 blob has no name for it and would call it
    * "UNKNOWN", but our driver names the GPU from its own
    * table, not the blob's). The T820 blob reports the bare
    * name with no "MC<n>" suffix -- it has no "Mali-G57
    * MC%d" format string, unlike the G615's r44p1 build.
    * This driver reports that same bare name below, for
    * both product IDs on arch 9. */
   if (p->arch_major == 9) {
      const uint32_t product = p->product_id & 0xf00f;
      if (product != 0x9001 && product != 0x9003)
         return false;
      snprintf(buf, size, "Mali-G57");
      return true;
   }
   if (p->arch_major != MALI_PAN_ARCH)
      return false;

   /*
    * The blob's naming for arch 11: product 0xb003 is the G615;
    * 0xb002 is the G715, named G615 when it has fewer than 7 cores. The
    * blob's "G715-Immortalis" variant is not reproduced.
    */
   const char *model;
   switch (p->product_id & 0xf00f) {
   case 0xb003:
      model = "G615";
      break;
   case 0xb002:
      model = p->core_count < 7 ? "G615" : "G715";
      break;
   default:
      return false;
   }
   snprintf(buf, size, "Mali-%s MC%u", model, p->core_count);
   return true;
}

/* ---------------------------------------------------------------------- */
/* Timestamps                                                              */

/*
 * GPU timestamps count the Arm generic timer (the blob takes 13 MHz
 * from its platform config on the RG 477V); CNTFRQ_EL0 gives the same
 * frequency without a config store.
 */
static uint64_t
timestamp_frequency(void)
{
#if defined(__aarch64__)
   uint64_t hz;
   __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(hz));
   return hz;
#else
   return 0;
#endif
}

/* ---------------------------------------------------------------------- */
/* Memory                                                                  */

/*
 * One DEVICE_LOCAL heap whose size is the smaller of the memory kbase
 * reports and 2^VA bits, at least 1 GiB; three types (uncached coherent,
 * cached, lazily allocated). The cached type is host-coherent only when
 * the kernel runs the GPU in full ACE coherency (coherency mode 1). No
 * protected type.
 */
static uint64_t
heap_size(const struct mali_kbase_gpu_props *p)
{
   uint64_t size = p->gpu_available_memory_size;
   if (p->va_bits && p->va_bits < 64)
      size = MIN2(size, 1ull << p->va_bits);
   return MAX2(size, 1ull << 30);
}

static void
init_memory(struct mali_physical_device *pdev)
{
   const struct mali_kbase_gpu_props *p = &pdev->props;
   VkPhysicalDeviceMemoryProperties *m = &pdev->memory;
   const bool ace = p->coherency_mode == 1;

   m->memoryHeapCount = 1;
   m->memoryHeaps[0] = (VkMemoryHeap) {
      .size = heap_size(p),
      .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
   };

   m->memoryTypeCount = 3;
   m->memoryTypes[0] = (VkMemoryType) {
      .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      .heapIndex = 0,
   };
   m->memoryTypes[1] = (VkMemoryType) {
      .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                       (ace ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : 0),
      .heapIndex = 0,
   };
   m->memoryTypes[2] = (VkMemoryType) {
      .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                       VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT,
      .heapIndex = 0,
   };
}

/* ---------------------------------------------------------------------- */
/* Features, properties, extensions                                        */

/*
 * Only what ARMSX2 and the Android loader need.
 * VK_KHR_driver_properties is what makes ARMSX2 read driverID/driverInfo.
 */
static const struct vk_device_extension_table mali_device_extensions = {
   .KHR_driver_properties = true,
   /* Sync-file import and export: the Android loader signals present
    * fences by importing the release fence (swapchain_maintenance1). */
   .KHR_external_fence_fd = true,
   .KHR_external_semaphore_fd = true,
#ifdef VK_USE_PLATFORM_ANDROID_KHR
   /* Without it the Android loader builds no swapchain. */
   .ANDROID_native_buffer = true,
#endif
   /* ARMSX2's in-tile destination read (the blob advertises both names).
    * Without it ARMSX2 reads the destination through an input attachment
    * with a by-region barrier between primitives, splitting draws (3.5
    * times the draw calls on Stuntman) and running up to 6.9 times the
    * blob's frame time. Input attachments are tile reads already; with
    * the rasterization-order flags the pipeline turns off forward pixel
    * kill and early depth/stencil updates (mali_pipeline_state.c), as
    * panvk does on v10+. Not on v9 yet -- see
    * device_extensions_for_arch() below. */
   .EXT_rasterization_order_attachment_access = true,
   .ARM_rasterization_order_attachment_access = true,
};

/*
 * v9 (G57) drops rasterization-order attachment access: the
 * job-manager render-pass encoder has no DCD Flags 2 there, and it
 * is not implemented yet regardless. It stays off even once that
 * lands, until ARMSX2's G57 ROAA rule has an exemption for our
 * driver and a device measurement supports lifting it; we do not
 * touch the ARMSX2 side here.
 */
static struct vk_device_extension_table
device_extensions_for_arch(uint32_t arch)
{
   struct vk_device_extension_table ext = mali_device_extensions;
   if (arch == 9) {
      ext.EXT_rasterization_order_attachment_access = false;
      ext.ARM_rasterization_order_attachment_access = false;
   }
   return ext;
}

/*
 * The features we intend to implement for ARMSX2 on this GPU, not the
 * blob's full list. Values that decide ARMSX2's paths follow the blob
 * where we can back them.
 */
static void
get_features(struct vk_features *f, uint32_t arch)
{
   /* v9 (G57): rasterization-order attachment access is not advertised
    * yet (device_extensions_for_arch() above). */
   const bool roaa = arch != 9;

   *f = (struct vk_features) {
      /* Vulkan 1.0 */
      .robustBufferAccess = true,
      .fullDrawIndexUint32 = true,
      .imageCubeArray = true,
      .independentBlend = true,
      .geometryShader = false,        /* blob: true; no GS in our compiler */
      .tessellationShader = false,
      .sampleRateShading = true,
      .dualSrcBlend = false,          /* as the blob; ARMSX2 measured so */
      .logicOp = false,
      .multiDrawIndirect = false,
      .drawIndirectFirstInstance = false,
      .depthClamp = true,
      .depthBiasClamp = true,
      .fillModeNonSolid = false,
      .depthBounds = false,
      .wideLines = false,
      .largePoints = true,
      .alphaToOne = false,
      .multiViewport = false,
      .samplerAnisotropy = true,
      .textureCompressionETC2 = true,
      .textureCompressionASTC_LDR = true,
      .textureCompressionBC = false,
      .occlusionQueryPrecise = true,
      .pipelineStatisticsQuery = false,
      .vertexPipelineStoresAndAtomics = false,
      .fragmentStoresAndAtomics = true,
      .shaderImageGatherExtended = true,
      .shaderStorageImageExtendedFormats = true,
      .shaderStorageImageReadWithoutFormat = true,
      .shaderStorageImageWriteWithoutFormat = true,
      .shaderUniformBufferArrayDynamicIndexing = true,
      .shaderSampledImageArrayDynamicIndexing = true,
      .shaderStorageBufferArrayDynamicIndexing = true,
      .shaderStorageImageArrayDynamicIndexing = true,
      .shaderInt16 = true,

      /* Vulkan 1.1: multiview is required of a 1.1 device. */
      .multiview = true,

      /* VK_EXT_rasterization_order_attachment_access */
      .rasterizationOrderColorAttachmentAccess = roaa,
      .rasterizationOrderDepthAttachmentAccess = roaa,
      .rasterizationOrderStencilAttachmentAccess = roaa,
   };
}

/* Sample counts for a format of the given bytes per pixel: 1 and 4
 * samples when bytes x 512 fits the tile buffer, 8 at bytes x 1024, 16 at
 * bytes x 2048. */
static VkSampleCountFlags
sample_counts(unsigned bytes_per_pixel, unsigned tilebuf_bytes)
{
   VkSampleCountFlags s = VK_SAMPLE_COUNT_1_BIT;
   if (bytes_per_pixel * 512 <= tilebuf_bytes)
      s |= VK_SAMPLE_COUNT_4_BIT;
   if (bytes_per_pixel * 1024 <= tilebuf_bytes)
      s |= VK_SAMPLE_COUNT_8_BIT;
   if (bytes_per_pixel * 2048 <= tilebuf_bytes)
      s |= VK_SAMPLE_COUNT_16_BIT;
   return s;
}

/*
 * v11 (G615): 32 KiB when the low byte of core_features is 3 or 4, else
 * 16 KiB. v9 (G57): a fixed 16 KiB -- both writers of the T820 blob's
 * tile-buffer-budget global store the same constant regardless of
 * core_features, and Mesa's model table gives the G57 the same fixed
 * 16 KiB.
 */
static unsigned
tilebuf_budget(const struct mali_kbase_gpu_props *p)
{
   if (p->arch_major == 9)
      return 16384;
   unsigned variant = p->core_features & 0xff;
   return (variant == 3 || variant == 4) ? 32768 : 16384;
}

/* For image format queries (mali_formats.c, declared in mali_image.h). */
VkSampleCountFlags mali_physical_device_sample_counts(const struct mali_physical_device *pdev,
                                                      unsigned bytes_per_pixel);

VkSampleCountFlags
mali_physical_device_sample_counts(const struct mali_physical_device *pdev,
                                   unsigned bytes_per_pixel)
{
   return sample_counts(bytes_per_pixel, tilebuf_budget(&pdev->props));
}

static void
hash_uuid(uint8_t out[VK_UUID_SIZE], const void *data, size_t size)
{
   blake3_hash h;
   _mesa_blake3_compute(data, size, h);
   memcpy(out, h, VK_UUID_SIZE);
}

/*
 * pipelineCacheUUID must change whenever our compiler or the shader binary
 * format changes, and should not change otherwise: ARMSX2 throws its
 * whole pipeline cache away when the UUID differs, so a UUID tied to the
 * commit made every driver update start with an empty cache. It hashes
 * the architecture's shader cache ID (a hash of every source that can
 * change that architecture's shader binaries, src/vk/gen_shader_cache_id.py:
 * MALI_SHADER_CACHE_ID for the G615, MALI_SHADER_CACHE_ID_V9 for the G57)
 * and the GPU ID.
 */
static void
pipeline_cache_uuid(uint8_t out[VK_UUID_SIZE], const char *cache_id, size_t cache_id_size,
                    uint64_t gpu_id)
{
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, cache_id, cache_id_size);
   _mesa_blake3_update(&ctx, &gpu_id, sizeof(gpu_id));

   blake3_hash h;
   _mesa_blake3_final(&ctx, h);
   memcpy(out, h, VK_UUID_SIZE);
}

static void
get_properties(const struct mali_physical_device *pdev, const char *name,
               struct vk_properties *props)
{
   const struct mali_kbase_gpu_props *p = &pdev->props;
   const VkSampleCountFlags samples = sample_counts(4, tilebuf_budget(p));

   /* min(max_workgroup_size, max_barrier_size); 1024 on the device.
    * Fall back to Vulkan's minimum if the kernel did not say. */
   uint32_t wg = MIN2(p->max_workgroup_size, p->max_barrier_size);
   if (!wg)
      wg = 128;

   const uint64_t max_alloc = MIN2(heap_size(p), 8ull << 30);
   const float ts_period = pdev->timestamp_hz ? 1e9f / (float)pdev->timestamp_hz : 1.0f;

   /* The G57 build's own identity scheme (above). */
   const uint32_t driver_version =
      p->arch_major == 9 ? MALI_V9_DRIVER_VERSION : MALI_DRIVER_VERSION;
   const char *const driver_info =
      p->arch_major == 9 ? MALI_V9_DRIVER_INFO : MALI_DRIVER_INFO;

   *props = (struct vk_properties) {
      .apiVersion = MALI_API_VERSION,
      .driverVersion = driver_version,
      .vendorID = MALI_VENDOR_ID,
      .deviceID = (uint32_t)p->gpu_id,
      .deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,

      /* Hardware limits: the blob's values. */
      .maxImageDimension1D = 16384,
      .maxImageDimension2D = 16384,
      .maxImageDimension3D = 16384,
      .maxImageDimensionCube = 16384,
      .maxImageArrayLayers = 4096,
      .maxTexelBufferElements = 1u << 28,
      .maxStorageBufferRange = 1u << 31,
      .maxMemoryAllocationCount = 16384,
      .maxSamplerAllocationCount = 4000,
      .bufferImageGranularity = 1,
      .sparseAddressSpaceSize = 0,
      .maxComputeSharedMemorySize = 32768,
      .maxComputeWorkGroupCount = { 65535, 65535, 65535 },
      .maxComputeWorkGroupInvocations = wg,
      .maxComputeWorkGroupSize = { wg, wg, wg },
      .subPixelPrecisionBits = 8,
      .subTexelPrecisionBits = 8,
      .mipmapPrecisionBits = 8,
      .maxDrawIndexedIndexValue = UINT32_MAX,
      .maxDrawIndirectCount = 1,
      .maxSamplerLodBias = 126.0f,
      .maxSamplerAnisotropy = 16.0f,
      .maxViewports = 1,
      .maxViewportDimensions = { 16384, 16384 },
      .viewportBoundsRange = { -32768.0f, 32767.0f },
      .viewportSubPixelBits = 0,
      .minMemoryMapAlignment = 64,
      .minTexelBufferOffsetAlignment = 64,
      .minUniformBufferOffsetAlignment = 16,
      .minStorageBufferOffsetAlignment = 64,
      .minTexelOffset = -8,
      .maxTexelOffset = 7,
      .minTexelGatherOffset = -8,
      .maxTexelGatherOffset = 7,
      .minInterpolationOffset = -0.5f,
      .maxInterpolationOffset = 0.5f,
      .subPixelInterpolationOffsetBits = 4,
      .maxFramebufferWidth = 16384,
      .maxFramebufferHeight = 16384,
      .maxFramebufferLayers = 256,
      .framebufferColorSampleCounts = samples,
      .framebufferDepthSampleCounts = samples,
      .framebufferStencilSampleCounts = samples,
      .framebufferNoAttachmentsSampleCounts = samples,
      .maxColorAttachments = 8,
      .sampledImageColorSampleCounts = samples,
      .sampledImageIntegerSampleCounts = samples,
      .sampledImageDepthSampleCounts = samples,
      .sampledImageStencilSampleCounts = samples,
      .storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .maxSampleMaskWords = 1,
      .timestampComputeAndGraphics = pdev->timestamp_hz != 0,
      .timestampPeriod = ts_period,
      .maxClipDistances = 0,
      .maxCullDistances = 0,
      .maxCombinedClipAndCullDistances = 0,
      .discreteQueuePriorities = 2,
      .pointSizeRange = { 1.0f, 1024.0f },
      .lineWidthRange = { 1.0f, 1.0f },
      .pointSizeGranularity = 0.0625f,
      .lineWidthGranularity = 0.0f,
      .strictLines = true,
      .standardSampleLocations = true,
      .optimalBufferCopyOffsetAlignment = 64,
      .optimalBufferCopyRowPitchAlignment = 64,
      .nonCoherentAtomSize = 64,

      /*
       * Limits that follow from our descriptor and shader-interface design,
       * which is panvk's for v11: 7 sets, 2^20 descriptors per type, 16
       * dynamic uniform and 8 dynamic storage buffers, 16 vertex buffers,
       * 1 MiB uniform buffers (12-bit entry count of 16-byte entries),
       * 256 bytes of push constants.
       */
      .maxUniformBufferRange = 1u << 20,
      .maxPushConstantsSize = 256,
      .maxBoundDescriptorSets = 7,
      .maxPerStageDescriptorSamplers = 1u << 20,
      .maxPerStageDescriptorUniformBuffers = 1u << 20,
      .maxPerStageDescriptorStorageBuffers = 1u << 20,
      .maxPerStageDescriptorSampledImages = 1u << 20,
      .maxPerStageDescriptorStorageImages = 1u << 20,
      .maxPerStageDescriptorInputAttachments = 9,
      .maxPerStageResources = 5 * (1u << 20) + 9,
      .maxDescriptorSetSamplers = 1u << 20,
      .maxDescriptorSetUniformBuffers = 1u << 20,
      .maxDescriptorSetUniformBuffersDynamic = 16,
      .maxDescriptorSetStorageBuffers = 1u << 20,
      .maxDescriptorSetStorageBuffersDynamic = 8,
      .maxDescriptorSetSampledImages = 1u << 20,
      .maxDescriptorSetStorageImages = 1u << 20,
      .maxDescriptorSetInputAttachments = 9,
      .maxVertexInputAttributes = 16,
      .maxVertexInputBindings = 16,
      .maxVertexInputAttributeOffset = UINT32_MAX,
      .maxVertexInputBindingStride = MESA_VK_MAX_VERTEX_BINDING_STRIDE,
      .maxVertexOutputComponents = 128,
      .maxFragmentInputComponents = 128,
      .maxFragmentOutputAttachments = 8,
      .maxFragmentDualSrcAttachments = 0,
      .maxFragmentCombinedOutputResources = 8 + (1u << 12) + (1u << 8),

      /* Vulkan 1.1 */
      .subgroupSize = 16,
      /*
       * Only what a 1.1 device must have until we have checked which
       * subgroup operations kraid handles. Not in the vertex stage: the
       * hardware may run vertex invocations for indices that are never
       * drawn (panvk's reasoning).
       */
      .subgroupSupportedStages = VK_SHADER_STAGE_FRAGMENT_BIT |
                                 VK_SHADER_STAGE_COMPUTE_BIT,
      .subgroupSupportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT,
      .subgroupQuadOperationsInAllStages = false,
      .pointClippingBehavior = VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES,
      .maxMultiviewViewCount = 8,
      .maxMultiviewInstanceIndex = UINT32_MAX,
      .protectedNoFault = false,
      .maxPerSetDescriptors = UINT16_MAX,
      .maxMemoryAllocationSize = max_alloc,

      /* VK_KHR_driver_properties */
      .driverID = VK_DRIVER_ID_ARM_PROPRIETARY,
      .conformanceVersion = { 0, 0, 0, 0 },
   };

   snprintf(props->deviceName, sizeof(props->deviceName), "%s", name);
   /* The blob reports the device name as driverName too. */
   snprintf(props->driverName, sizeof(props->driverName), "%s", name);
   snprintf(props->driverInfo, sizeof(props->driverInfo), "%s", driver_info);

   /* deviceUUID: gpu_id little-endian, then 1. */
   uint32_t id = (uint32_t)p->gpu_id;
   memcpy(props->deviceUUID, &id, sizeof(id));
   props->deviceUUID[4] = 1;
   /* driverUUID: a hash of driverInfo, so never the blob's. */
   hash_uuid(props->driverUUID, driver_info, strlen(driver_info) + 1);
   if (p->arch_major == 9)
      pipeline_cache_uuid(props->pipelineCacheUUID, MALI_SHADER_CACHE_ID_V9,
                          sizeof(MALI_SHADER_CACHE_ID_V9), p->gpu_id);
   else
      pipeline_cache_uuid(props->pipelineCacheUUID, MALI_SHADER_CACHE_ID,
                          sizeof(MALI_SHADER_CACHE_ID), p->gpu_id);
}

/* ---------------------------------------------------------------------- */
/* Queries we implement ourselves                                          */

VKAPI_ATTR void VKAPI_CALL
mali_GetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice physicalDevice,
                                             uint32_t *pQueueFamilyPropertyCount,
                                             VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out, pQueueFamilyProperties,
                          pQueueFamilyPropertyCount);

   vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p) {
      p->queueFamilyProperties = (VkQueueFamilyProperties) {
         .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                       VK_QUEUE_TRANSFER_BIT,
         .queueCount = MALI_QUEUE_COUNT,
         .timestampValidBits = pdev->timestamp_hz ? 64 : 0,
         .minImageTransferGranularity = { 1, 1, 1 },
      };
      /* No pNext structure we know is filled: we expose no global
       * priorities and no other queue-family extension. */
   }
}

VKAPI_ATTR void VKAPI_CALL
mali_GetPhysicalDeviceMemoryProperties2(VkPhysicalDevice physicalDevice,
                                        VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, physicalDevice);
   pMemoryProperties->memoryProperties = pdev->memory;
}

/* ---------------------------------------------------------------------- */
/* Creation                                                                */

/* Collects the kbase layer's messages while probing a node, so a node
 * that does not exist costs no log line. */
struct probe_log {
   char text[512];
   size_t len;
};

static void
probe_log_append(void *user, const char *msg)
{
   struct probe_log *l = user;
   int n = snprintf(l->text + l->len, sizeof(l->text) - l->len, "%s%s",
                    l->len ? "; " : "", msg);
   if (n > 0)
      l->len = MIN2(l->len + (size_t)n, sizeof(l->text) - 1);
}

static VkResult
physical_device_create(struct mali_instance *instance, const char *path,
                       struct vk_physical_device **out)
{
   struct probe_log log = { .len = 0 };
   struct mali_kbase *kb;

   /*
    * Like the blob, open a kbase context just to read the GPU
    * properties, then close it again. Each VkDevice opens its own.
    */
   const struct mali_kbase_create_info info = {
      .device_path = path,
      .backend = instance->kbase_backend,
      .log = probe_log_append,
      .log_user = &log,
   };
   enum mali_kbase_result r = mali_kbase_create(&info, &kb);
   if (r == MALI_KBASE_ERROR_DEVICE_NOT_FOUND)
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   if (r != MALI_KBASE_SUCCESS) {
      mesa_logw("libmali: skipping %s: %s (%s)", path, mali_kbase_result_str(r),
                log.text);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }

   struct mali_physical_device *pdev =
      vk_zalloc(&instance->vk.alloc, sizeof(*pdev), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!pdev) {
      mali_kbase_destroy(kb);
      return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   snprintf(pdev->path, sizeof(pdev->path), "%s", path);
   pdev->props = kb->props;
   pdev->cs_work_registers = kb->glb.cs_work_registers;
   pdev->arch = pdev->props.arch_major;
   const enum mali_kbase_frontend frontend = kb->frontend;
   mali_kbase_destroy(kb);

   /* The arch has to match the frontend the kernel speaks: CSF with v11,
    * the job manager with v9. */
   const bool arch_ok =
      (pdev->arch == MALI_PAN_ARCH && frontend == MALI_KBASE_FRONTEND_CSF) ||
      (pdev->arch == 9 && frontend == MALI_KBASE_FRONTEND_JM);
   char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
   if (!arch_ok || !mali_device_name(&pdev->props, name, sizeof(name))) {
      mesa_logw("libmali: skipping %s: GPU product 0x%04x (arch %u) is not "
                "one this driver was built for (arch %u)", path,
                pdev->props.product_id, pdev->props.arch_major, MALI_PAN_ARCH);
      vk_free(&instance->vk.alloc, pdev);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }

   pdev->timestamp_hz = timestamp_frequency();
   init_memory(pdev);
   /* driverInfo carries the shader cache ID, not the commit; say which build
    * this is here. No "libmali:" prefix: the corpus tools count those lines
    * as warnings. */
   mesa_logi("libmali build %s, driverInfo %s", MALI_GIT_SHA,
            pdev->arch == 9 ? MALI_V9_DRIVER_INFO : MALI_DRIVER_INFO);

   struct vk_features features;
   struct vk_properties props;
   get_features(&features, pdev->arch);
   get_properties(pdev, name, &props);

   const struct vk_device_extension_table extensions =
      device_extensions_for_arch(pdev->arch);

   struct vk_physical_device_dispatch_table dispatch;
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch, &mali_physical_device_entrypoints, true);

   VkResult result = vk_physical_device_init(&pdev->vk, &instance->vk,
                                             &extensions, &features,
                                             &props, &dispatch);
   if (result != VK_SUCCESS) {
      vk_free(&instance->vk.alloc, pdev);
      return result;
   }
   vk_physical_device_dispatch_table_from_entrypoints(
      &pdev->vk.dispatch_table, &mali_waist_physical_device_entrypoints, false);

   pdev->vk.supported_sync_types = pdev->arch == 9 ? mali_v9_sync_types : mali_v11_sync_types;

   /* Compiled shaders are what pipeline caches hold. */
   pdev->vk.pipeline_cache_import_ops = pdev->arch == 9 ? mali_v9_pipeline_cache_import_ops
                                                        : mali_v11_pipeline_cache_import_ops;

   *out = &pdev->vk;
   return VK_SUCCESS;
}

VkResult
mali_physical_devices_enumerate(struct vk_instance *vk_instance)
{
   struct mali_instance *instance = container_of(vk_instance, struct mali_instance, vk);
   const bool os_backend = instance->kbase_backend == mali_kbase_os_backend();

   for (unsigned i = 0; i < MALI_MAX_PHYSICAL_DEVICES; i++) {
      char path[32];
      snprintf(path, sizeof(path), "/dev/mali%u", i);

      /* The blob checks each node with access() first. With a test
       * backend the node exists only in the fake, so we ask it. */
      if (os_backend && access(path, F_OK) != 0)
         continue;

      struct vk_physical_device *pdev;
      VkResult result = physical_device_create(instance, path, &pdev);
      if (result == VK_ERROR_INCOMPATIBLE_DRIVER)
         continue;
      if (result != VK_SUCCESS)
         return result;
      list_addtail(&pdev->link, &vk_instance->physical_devices.list);
   }
   return VK_SUCCESS;
}

void
mali_physical_device_destroy(struct vk_physical_device *vk_pdev)
{
   struct mali_physical_device *pdev =
      container_of(vk_pdev, struct mali_physical_device, vk);
   vk_physical_device_finish(&pdev->vk);
   vk_free(&pdev->vk.instance->alloc, pdev);
}
