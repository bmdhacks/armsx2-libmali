/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * What the Android window system needs from the driver, in terms that do
 * not need Android's headers, so the host tests can call it. The Android
 * entry points in mali_android.c unwrap the loader's structures and call
 * these.
 */

#ifndef MALI_WSI_H
#define MALI_WSI_H

#include <stdbool.h>
#include <stdint.h>

#include "vulkan/vulkan_core.h"

struct mali_device;
struct mali_image;
struct mali_physical_device;
struct mali_queue;

/* gralloc usage bits (hardware/gralloc.h, gralloc1.h; the 64-bit gralloc 4
 * BufferUsage has the same values). */
#define MALI_GRALLOC_USAGE_HW_TEXTURE 0x100ull  /* GPU_TEXTURE (consumer) */
#define MALI_GRALLOC_USAGE_HW_RENDER  0x200ull  /* GPU_RENDER_TARGET (producer) */

/* The RG 477V's ro.vendor.arm.gralloc.no_afbc_usage_flags, used when the
 * property cannot be read. */
#define MALI_GRALLOC_NO_AFBC_DEFAULT 0x0200000000000000ull

/*
 * The gralloc usage for swapchain images of this format and image usage
 * (vkGetSwapchainGrallocUsage2ANDROID semantics). As the blob: GPU texture
 * as consumer usage and GPU render target as producer usage whatever the
 * image usage. Unlike the blob, the producer usage always carries
 * no_afbc: we render linear images only, so gralloc must not pick AFBC.
 * VK_ERROR_FORMAT_NOT_SUPPORTED for a format or usage we cannot render or
 * write to linear images with.
 */
VkResult mali_wsi_gralloc_usage(struct mali_physical_device *pdev, VkFormat format,
                                VkImageUsageFlags usage, uint64_t no_afbc,
                                uint64_t *consumer, uint64_t *producer);

/*
 * Give a swapchain image the memory of a gralloc buffer: a linear layout
 * with the buffer's row stride (in pixels, as the loader passes it), the
 * dma-buf imported through kbase (a dup of dmabuf_fd; the caller keeps
 * its fd) and bound. The image owns the import from then on.
 */
VkResult mali_wsi_image_bind_buffer(struct mali_device *dev, struct mali_image *image,
                                    int dmabuf_fd, uint32_t stride_px,
                                    const VkAllocationCallbacks *alloc);

/*
 * vkAcquireImageANDROID: the display's release fence (a sync file, or -1
 * for "already signalled") becomes the payload of the semaphore and/or
 * the fence. Consumes fd on every path, success or not.
 */
VkResult mali_wsi_acquire(struct mali_device *dev, int fd, VkSemaphore semaphore,
                          VkFence fence);

/*
 * vkQueueSignalReleaseImageANDROID: one sync file that signals when every
 * semaphore's payload has signalled; the semaphores are waited on
 * (binary semaphore payloads are consumed). *out_fd is -1 when nothing is
 * pending.
 */
VkResult mali_wsi_release(struct mali_queue *queue, uint32_t count,
                          const VkSemaphore *semaphores, int *out_fd);

#ifdef VK_USE_PLATFORM_ANDROID_KHR
#include "vulkan/vk_android_native_buffer.h"

/* A swapchain image created or bound with a VkNativeBufferANDROID
 * (mali_android.c): mali_wsi_image_bind_buffer on the buffer's dma-buf. */
VkResult mali_android_bind_native_buffer(struct mali_device *dev, struct mali_image *image,
                                         const VkNativeBufferANDROID *nb,
                                         const VkAllocationCallbacks *alloc);
#endif

#endif
