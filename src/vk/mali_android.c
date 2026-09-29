/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Android only: the HAL module the Android Vulkan loader opens, and the
 * VK_ANDROID_native_buffer entry points it drives swapchains with. The
 * work is in mali_wsi.c; this file unwraps Android's structures.
 *
 * The loader (/system/lib64/libvulkan.so) loads the driver with
 * android_load_sphal_library or android_dlopen_ext (adrenotools hooks both
 * to substitute this library), looks up the symbol HMI, and opens device
 * "vk0" through it; from then on it only uses the three functions of
 * hwvulkan_device_t and vkGetInstanceProcAddr. Surfaces and swapchains
 * are the loader's; per swapchain image it calls vkCreateImage with a
 * VkNativeBufferANDROID, per acquire vkAcquireImageANDROID, per present
 * vkQueueSignalReleaseImageANDROID.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/system_properties.h>
#include <unistd.h>

#include <hardware/hwvulkan.h>

#include "util/log.h"
#include "vk_log.h"
#include "vk_util.h"

#include "mali_image.h"
#include "mali_vk.h"
#include "mali_wsi.h"

#define wsi_errorf(obj, result, ...)                                            \
   (mesa_loge("libmali: WSI: " __VA_ARGS__), vk_errorf(obj, result, __VA_ARGS__))

/* ---------------------------------------------------------------------- */
/* The HAL module                                                          */

static int mali_hal_open(const struct hw_module_t *mod, const char *id,
                         struct hw_device_t **dev);

static_assert(HWVULKAN_DISPATCH_MAGIC == ICD_LOADER_MAGIC,
              "dispatchable handles start with the loader magic");

static hw_module_methods_t mali_hal_methods = {
   .open = mali_hal_open,
};

__attribute__((visibility("default"))) struct hwvulkan_module_t HAL_MODULE_INFO_SYM = {
   .common = {
      .tag = HARDWARE_MODULE_TAG,
      .module_api_version = HWVULKAN_MODULE_API_VERSION_0_1,
      .hal_api_version = HARDWARE_MAKE_API_VERSION(1, 0),
      .id = HWVULKAN_HARDWARE_MODULE_ID,
      .name = "libmali Vulkan driver (kbase)",
      .author = "ARMSX2 and bmdhacks",
      .methods = &mali_hal_methods,
   },
};

static int
mali_hal_close(struct hw_device_t *dev)
{
   /* Called when the loader unloads the driver. */
   free(container_of(dev, hwvulkan_device_t, common));
   return 0;
}

static int
mali_hal_open(const struct hw_module_t *mod, const char *id, struct hw_device_t **dev)
{
   if (mod != &HAL_MODULE_INFO_SYM.common || strcmp(id, HWVULKAN_DEVICE_0) != 0)
      return -1;

   hwvulkan_device_t *hal = calloc(1, sizeof(*hal));
   if (!hal)
      return -1;
   *hal = (hwvulkan_device_t){
      .common = {
         .tag = HARDWARE_DEVICE_TAG,
         .version = HWVULKAN_DEVICE_API_VERSION_0_1,
         .module = &HAL_MODULE_INFO_SYM.common,
         .close = mali_hal_close,
      },
      .EnumerateInstanceExtensionProperties =
         (PFN_vkEnumerateInstanceExtensionProperties)vk_icdGetInstanceProcAddr(
            NULL, "vkEnumerateInstanceExtensionProperties"),
      .CreateInstance =
         (PFN_vkCreateInstance)vk_icdGetInstanceProcAddr(NULL, "vkCreateInstance"),
      .GetInstanceProcAddr =
         (PFN_vkGetInstanceProcAddr)vk_icdGetInstanceProcAddr(NULL, "vkGetInstanceProcAddr"),
   };
   *dev = &hal->common;
   return 0;
}

/* ---------------------------------------------------------------------- */
/* Gralloc usage                                                           */

/*
 * The gralloc usage bits that keep Arm's gralloc from choosing AFBC, from
 * the property the blob reads. The property may not be readable from an
 * application's SELinux domain; then the RG 477V's value, which is Arm
 * gralloc's.
 */
static uint64_t
no_afbc_usage(void)
{
   static uint64_t value;
   static bool known;
   if (!known) {
      char buf[PROP_VALUE_MAX] = {0};
      value = MALI_GRALLOC_NO_AFBC_DEFAULT;
      if (__system_property_get("ro.vendor.arm.gralloc.no_afbc_usage_flags", buf) > 0) {
         char *end;
         unsigned long long v = strtoull(buf, &end, 0);
         if (end != buf)
            value = v;
      } else {
         mesa_logi("libmali: ro.vendor.arm.gralloc.no_afbc_usage_flags not readable, "
                   "using 0x%llx", (unsigned long long)value);
      }
      known = true;
   }
   return value;
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetSwapchainGrallocUsage2ANDROID(VkDevice _device, VkFormat format,
                                      VkImageUsageFlags imageUsage,
                                      VkSwapchainImageUsageFlagsANDROID swapchainImageUsage,
                                      uint64_t *grallocConsumerUsage,
                                      uint64_t *grallocProducerUsage)
{
   VK_FROM_HANDLE(mali_device, dev, _device);

   /* Shared (front-buffer) presentation is not offered: sharedImage is
    * false, so the loader never asks for it. */
   if (swapchainImageUsage & VK_SWAPCHAIN_IMAGE_USAGE_SHARED_BIT_ANDROID)
      return wsi_errorf(dev, VK_ERROR_FORMAT_NOT_SUPPORTED,
                       "shared presentable images are not supported");
   return mali_wsi_gralloc_usage(mali_device_physical(dev), format, imageUsage,
                                 no_afbc_usage(), grallocConsumerUsage,
                                 grallocProducerUsage);
}

/* The gralloc0 form: 32 bits, so no room for the no-AFBC bit (the blob
 * drops it too). The loader uses it only if ...Usage2 is missing. */
VKAPI_ATTR VkResult VKAPI_CALL
mali_GetSwapchainGrallocUsageANDROID(VkDevice _device, VkFormat format,
                                     VkImageUsageFlags imageUsage, int *grallocUsage)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   uint64_t consumer, producer;
   VkResult result = mali_wsi_gralloc_usage(mali_device_physical(dev), format, imageUsage,
                                            0, &consumer, &producer);
   if (result == VK_SUCCESS)
      *grallocUsage = (int)(consumer | producer);
   return result;
}

/* ---------------------------------------------------------------------- */
/* Swapchain images                                                        */

/*
 * The buffer's dma-buf among the handle's fds. Where it sits is gralloc's
 * business: MediaTek's gralloc puts a "gralloc_extra" anonymous inode
 * first, then the dma-buf, then a memfd of shared metadata (seen on the
 * RG 477V). The blob asks the gralloc mapper for it. We look at the
 * names: a dma-buf's is "/dmabuf:<buffer name>". (fstatfs for the
 * dma-buf filesystem's magic is refused by SELinux there.)
 */
static int
native_buffer_dmabuf_fd(const native_handle_t *h)
{
   for (int i = 0; i < h->numFds; i++) {
      char path[32], target[16] = {0};
      snprintf(path, sizeof(path), "/proc/self/fd/%d", h->data[i]);
      if (readlink(path, target, sizeof(target) - 1) > 8 && !strncmp(target, "/dmabuf:", 8))
         return h->data[i];
   }
   return -1;
}

VkResult
mali_android_bind_native_buffer(struct mali_device *dev, struct mali_image *image,
                                const VkNativeBufferANDROID *nb,
                                const VkAllocationCallbacks *alloc)
{
   if (!nb->handle || nb->handle->numFds < 1)
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "native buffer without a dma-buf fd");
   if (nb->stride <= 0)
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "native buffer with stride %d", nb->stride);
   const int fd = native_buffer_dmabuf_fd(nb->handle);
   if (fd >= 0)
      return mali_wsi_image_bind_buffer(dev, image, fd, (uint32_t)nb->stride, alloc);

   /* No name to go by (/proc not readable?): the one kbase can import. */
   mesa_logw("libmali: WSI: no fd of the native buffer is named as a dma-buf; "
             "trying to import each of its %d fds", nb->handle->numFds);
   VkResult result = VK_ERROR_INVALID_EXTERNAL_HANDLE;
   for (int i = 0; i < nb->handle->numFds && result != VK_SUCCESS; i++)
      result = mali_wsi_image_bind_buffer(dev, image, nb->handle->data[i],
                                          (uint32_t)nb->stride, alloc);
   return result;
}

/* ---------------------------------------------------------------------- */
/* Acquire and present                                                     */

VKAPI_ATTR VkResult VKAPI_CALL
mali_AcquireImageANDROID(VkDevice _device, VkImage image, int nativeFenceFd,
                         VkSemaphore semaphore, VkFence fence)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   return mali_wsi_acquire(dev, nativeFenceFd, semaphore, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_QueueSignalReleaseImageANDROID(VkQueue _queue, uint32_t waitSemaphoreCount,
                                    const VkSemaphore *pWaitSemaphores, VkImage image,
                                    int *pNativeFenceFd)
{
   VK_FROM_HANDLE(mali_queue, queue, _queue);
   return mali_wsi_release(queue, waitSemaphoreCount, pWaitSemaphores, pNativeFenceFd);
}
