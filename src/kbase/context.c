/*
 * Copyright 2026 The libmali authors
 * SPDX-License-Identifier: MIT
 */

/*
 * Context creation and teardown. The order follows the blob's context
 * init, minus the steps that are the blob's own bookkeeping or belong to
 * the command-stream layer:
 *
 *   blob step                          here
 *   2  open + VERSION_CHECK            open, version handshake
 *   3  SET_FLAGS                       SET_FLAGS
 *   4  tracking page (UK < 1.18)       not needed at UK 1.20
 *   5  GET_GPUPROPS                    GET_GPUPROPS + decode
 *   7  CS_GET_GLB_IFACE (twice)        same
 *   8  GPU ID vs supported list        left to the caller
 *   9  map the user register page      same
 *   10-12 page pool, event thread      command-stream layer
 *   (after the context) MEM_EXEC_INIT  MEM_EXEC_INIT
 *   (after the context) MEM_JIT_INIT   MEM_JIT_INIT (context flags bit 0 or
 *                                      2, which the Vulkan context has)
 */

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "kbase_priv.h"

#define DEFAULT_DEVICE_PATH "/dev/mali0"

static enum mali_kbase_result
open_device(struct mali_kbase *kb, const char *path)
{
   int fd = kb->backend->open(kb->backend->priv, path);
   if (fd >= 0) {
      kb->fd = fd;
      return MALI_KBASE_SUCCESS;
   }

   int err = errno;
   KB_LOG(kb, "cannot open %s: %s", path, strerror(err));
   switch (err) {
   case ENOENT:
   case ENODEV:
   case ENXIO:
      return MALI_KBASE_ERROR_DEVICE_NOT_FOUND;
   case EACCES:
   case EPERM:
      return MALI_KBASE_ERROR_PERMISSION_DENIED;
   default:
      return MALI_KBASE_ERROR_KERNEL;
   }
}

/*
 * We propose UK 1.20, the r44p1 interface, and accept only 1.20 back. The
 * kernel answers min(its minor, ours) when the major matches and its own
 * version otherwise, so "1.20" means "a kernel at 1.20 or newer" (the
 * RG 477V is 1.21). The blob accepts older minors and switches ioctl
 * variants for them; we implement only the 1.20 variants (MEM_ALLOC_EX,
 * the 112-byte group create, tiler heap init with buf_desc_va, no
 * tracking page), so an older kernel is refused here, loudly, instead of
 * failing later in some ioctl.
 */
static enum mali_kbase_result
version_handshake(struct mali_kbase *kb)
{
   struct kb_ioctl_version_check vc = {
      .major = KB_UK_VERSION_MAJOR,
      .minor = KB_UK_VERSION_MINOR,
   };
   long ret = kb_ioctl(kb, KB_IOCTL_VERSION_CHECK, &vc);
   if (ret < 0) {
      KB_LOG(kb, "refusing device: VERSION_CHECK (CSF ioctl 52) failed: %s; "
                 "not a CSF kbase kernel?",
             strerror((int)-ret));
      return MALI_KBASE_ERROR_INCOMPATIBLE_KERNEL;
   }
   if (vc.major != KB_UK_VERSION_MAJOR || vc.minor != KB_UK_VERSION_MINOR) {
      KB_LOG(kb, "refusing device: kernel answered UK %u.%u, this driver "
                 "speaks only UK %u.%u (kbase r44p1)",
             vc.major, vc.minor, KB_UK_VERSION_MAJOR, KB_UK_VERSION_MINOR);
      return MALI_KBASE_ERROR_INCOMPATIBLE_KERNEL;
   }
   kb->uk_major = vc.major;
   kb->uk_minor = vc.minor;
   return MALI_KBASE_SUCCESS;
}

static enum mali_kbase_result
set_flags(struct mali_kbase *kb, uint32_t mmu_group)
{
   /* The blob sends its context flags masked to what the kernel accepts:
    * only the MMU group (bits 3..6) is ever set, and it is 0 on the
    * device. CCTX_EMBEDDED and CSF_EVENT_THREAD are user-space only. */
   struct kb_ioctl_set_flags sf = {
      .create_flags = (mmu_group << KB_CONTEXT_MMU_GROUP_ID_SHIFT) &
                      KB_CONTEXT_MMU_GROUP_ID_MASK,
   };
   long ret = kb_ioctl(kb, KB_IOCTL_SET_FLAGS, &sf);
   if (ret < 0) {
      KB_LOG(kb, "SET_FLAGS 0x%x failed: %s", sf.create_flags, strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   return MALI_KBASE_SUCCESS;
}

static enum mali_kbase_result
load_gpu_props(struct mali_kbase *kb)
{
   /* First call with no buffer returns the stream size. */
   struct kb_ioctl_get_gpuprops gp = {0};
   long size = kb_ioctl(kb, KB_IOCTL_GET_GPUPROPS, &gp);
   if (size <= 0) {
      KB_LOG(kb, "GET_GPUPROPS size query failed: %s",
             size < 0 ? strerror((int)-size) : "empty stream");
      return size < 0 ? kb_result_from_errno((int)-size, KB_ERRNO_GENERIC)
                      : MALI_KBASE_ERROR_MALFORMED_PROPERTIES;
   }

   void *buf = calloc(1, (size_t)size);
   if (!buf)
      return MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY;

   gp.buffer = (uint64_t)(uintptr_t)buf;
   gp.size = (uint32_t)size;
   long got = kb_ioctl(kb, KB_IOCTL_GET_GPUPROPS, &gp);
   enum mali_kbase_result r;
   if (got < 0) {
      KB_LOG(kb, "GET_GPUPROPS failed: %s", strerror((int)-got));
      r = kb_result_from_errno((int)-got, KB_ERRNO_GENERIC);
   } else {
      if (got > size)
         got = size;
      r = mali_kbase_gpu_props_decode(buf, (size_t)got, &kb->props);
      if (r != MALI_KBASE_SUCCESS)
         KB_LOG(kb, "GET_GPUPROPS stream of %ld bytes is truncated", got);
   }
   free(buf);
   return r;
}

static enum mali_kbase_result
load_glb_iface(struct mali_kbase *kb)
{
   /* First call with zero counts returns the header and the counts. */
   union kb_ioctl_cs_get_glb_iface q;
   memset(&q, 0, sizeof(q));
   long ret = kb_ioctl(kb, KB_IOCTL_CS_GET_GLB_IFACE, &q);
   if (ret < 0) {
      KB_LOG(kb, "CS_GET_GLB_IFACE failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }

   struct mali_kbase_glb_iface *g = &kb->glb;
   uint32_t groups = q.out.group_num;
   uint32_t streams = q.out.total_stream_num;
   g->groups = calloc(groups ? groups : 1, sizeof(*g->groups));
   g->streams = calloc(streams ? streams : 1, sizeof(*g->streams));
   if (!g->groups || !g->streams)
      return MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY;

   memset(&q, 0, sizeof(q));
   q.in.max_group_num = groups;
   q.in.max_total_stream_num = streams;
   q.in.groups_ptr = (uint64_t)(uintptr_t)g->groups;
   q.in.streams_ptr = (uint64_t)(uintptr_t)g->streams;
   ret = kb_ioctl(kb, KB_IOCTL_CS_GET_GLB_IFACE, &q);
   if (ret < 0) {
      KB_LOG(kb, "CS_GET_GLB_IFACE failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }

   g->version = q.out.glb_version;
   g->features = q.out.features;
   g->prfcnt_size = q.out.prfcnt_size;
   g->instr_features = q.out.instr_features;
   g->group_num = groups < q.out.group_num ? groups : q.out.group_num;
   g->total_stream_num =
      streams < q.out.total_stream_num ? streams : q.out.total_stream_num;
   /* STREAM_FEATURES bits 7:0 hold the work register count minus one: the
    * RG 477V reports 0x5f and the blob uses r0..r95 (upstream panthor
    * decodes the field the same way). */
   g->cs_work_registers =
      g->total_stream_num ? (g->streams[0].features & 0xff) + 1 : 0;
   return MALI_KBASE_SUCCESS;
}

static enum mali_kbase_result
map_user_reg_page(struct mali_kbase *kb)
{
   void *p = kb->backend->mmap(kb->backend->priv, KB_PAGE_SIZE, PROT_READ, kb->fd,
                               KB_MEM_CSF_USER_REG_PAGE_HANDLE);
   if (p == MAP_FAILED) {
      KB_LOG(kb, "mapping the user register page failed: %s", strerror(errno));
      return MALI_KBASE_ERROR_MAP_FAILED;
   }
   kb->user_reg_page = p;
   return MALI_KBASE_SUCCESS;
}

static enum mali_kbase_result
exec_init(struct mali_kbase *kb)
{
   /* On r44p1 CSF kernels the executable zone always exists and this only
    * validates the size; the blob issues it anyway, with the full 4 GiB,
    * and so do we. */
   struct kb_ioctl_mem_exec_init ei = {.va_pages = KB_EXEC_VA_PAGES};
   long ret = kb_ioctl(kb, KB_IOCTL_MEM_EXEC_INIT, &ei);
   if (ret < 0) {
      KB_LOG(kb, "MEM_EXEC_INIT failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   return MALI_KBASE_SUCCESS;
}

static enum mali_kbase_result
jit_init(struct mali_kbase *kb)
{
   /* The blob issues this for its Vulkan contexts (create flags bits 0 and
    * 2). We do not use JIT memory, but on a 64-bit CSF context this is
    * what creates the CUSTOM_VA zone, and the kernel allocates tiler heap
    * contexts and chunks there: without it CS_TILER_HEAP_INIT fails with
    * ENOMEM (found on the RG 477V). It must come before the first
    * allocation. Group 0: the blob takes it from config option 0x3b,
    * whose default is not decoded. */
   struct kb_ioctl_mem_jit_init ji = {
      .va_pages = KB_JIT_VA_PAGES,
      .max_allocations = KB_JIT_MAX_ALLOCATIONS,
      .trim_level = KB_JIT_TRIM_LEVEL,
      .group_id = 0,
      .phys_pages = KB_JIT_VA_PAGES,
   };
   long ret = kb_ioctl(kb, KB_IOCTL_MEM_JIT_INIT, &ji);
   if (ret < 0) {
      KB_LOG(kb, "MEM_JIT_INIT failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   return MALI_KBASE_SUCCESS;
}

static void
release(struct mali_kbase *kb)
{
   if (kb->user_reg_page)
      kb->backend->munmap(kb->backend->priv, (void *)kb->user_reg_page, KB_PAGE_SIZE);
   free(kb->glb.groups);
   free(kb->glb.streams);
   if (kb->fd >= 0)
      kb->backend->close(kb->backend->priv, kb->fd);
   free(kb);
}

enum mali_kbase_result
mali_kbase_create(const struct mali_kbase_create_info *info, struct mali_kbase **out)
{
   static const struct mali_kbase_create_info defaults = {0};
   if (!out)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   *out = NULL;
   if (!info)
      info = &defaults;
   if (info->mmu_group > 15)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;

   struct mali_kbase *kb = calloc(1, sizeof(*kb));
   if (!kb)
      return MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY;
   kb->fd = -1;
   kb->backend = info->backend ? info->backend : mali_kbase_os_backend();
   kb->log = info->log;
   kb->log_user = info->log_user;

   long page = kb->backend->page_size(kb->backend->priv);
   if (page != KB_PAGE_SIZE) {
      KB_LOG(kb, "refusing device: page size is %ld, only 4096 is supported", page);
      release(kb);
      return MALI_KBASE_ERROR_INCOMPATIBLE_KERNEL;
   }

   const char *path = info->device_path ? info->device_path : DEFAULT_DEVICE_PATH;
   enum mali_kbase_result r = open_device(kb, path);
   if (r == MALI_KBASE_SUCCESS)
      r = version_handshake(kb);
   if (r == MALI_KBASE_SUCCESS)
      r = set_flags(kb, info->mmu_group);
   if (r == MALI_KBASE_SUCCESS)
      r = load_gpu_props(kb);
   if (r == MALI_KBASE_SUCCESS)
      r = load_glb_iface(kb);
   if (r == MALI_KBASE_SUCCESS)
      r = map_user_reg_page(kb);
   if (r == MALI_KBASE_SUCCESS)
      r = exec_init(kb);
   if (r == MALI_KBASE_SUCCESS)
      r = jit_init(kb);

   if (r != MALI_KBASE_SUCCESS) {
      release(kb);
      return r;
   }
   *out = kb;
   return MALI_KBASE_SUCCESS;
}

void
mali_kbase_destroy(struct mali_kbase *kb)
{
   if (kb)
      release(kb);
}

uint32_t
mali_kbase_latest_flush_id(const struct mali_kbase *kb)
{
   return kb->user_reg_page[KB_USER_REG_LATEST_FLUSH / 4];
}

enum mali_kbase_result
mali_kbase_timeinfo(struct mali_kbase *kb, uint64_t *mono_ns, uint64_t *gpu_ts)
{
   union kb_ioctl_get_cpu_gpu_timeinfo t;
   memset(&t, 0, sizeof(t));
   t.in.request_flags = KB_TIMEINFO_MONOTONIC | KB_TIMEINFO_TIMESTAMP;
   long ret = kb_ioctl(kb, KB_IOCTL_GET_CPU_GPU_TIMEINFO, &t);
   if (ret < 0) {
      KB_LOG(kb, "GET_CPU_GPU_TIMEINFO failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   *mono_ns = t.out.sec * 1000000000ull + t.out.nsec;
   *gpu_ts = t.out.timestamp;
   return MALI_KBASE_SUCCESS;
}
