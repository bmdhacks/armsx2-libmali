/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Reaching the device: the OS backend (the real system calls), result
 * strings, errno mapping and logging.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* O_CLOEXEC under -std=c11 */
#endif

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kbase_priv.h"

/* ---------------------------------------------------------------------- */
/* OS backend                                                              */

static int
os_open(void *priv, const char *path)
{
   (void)priv;
   /* The blob opens with O_RDWR | O_NONBLOCK | O_CLOEXEC and insists on a
    * character device. Close-on-exec also keeps the fd out of children;
    * UK 1.20 kernels refuse file operations on an inherited kbase fd. */
   int fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
   if (fd < 0)
      return -1;

   struct stat st;
   if (fstat(fd, &st) != 0 || !S_ISCHR(st.st_mode)) {
      close(fd);
      errno = ENODEV;
      return -1;
   }
   return fd;
}

static int
os_close(void *priv, int fd)
{
   (void)priv;
   return close(fd);
}

static int
os_ioctl(void *priv, int fd, unsigned long request, void *arg)
{
   (void)priv;
   int ret;
   do {
      ret = ioctl(fd, request, arg);
   } while (ret == -1 && errno == EINTR);
   return ret;
}

static void *
os_mmap(void *priv, size_t length, int prot, int fd, uint64_t offset)
{
   (void)priv;
   return mmap(NULL, length, prot, MAP_SHARED, fd, (off_t)offset);
}

static int
os_munmap(void *priv, void *addr, size_t length)
{
   (void)priv;
   return munmap(addr, length);
}

static long
os_page_size(void *priv)
{
   (void)priv;
   return sysconf(_SC_PAGESIZE);
}

static int
os_poll(void *priv, struct pollfd *fds, unsigned nfds, int timeout_ms)
{
   (void)priv;
   return poll(fds, nfds, timeout_ms);
}

static long
os_read(void *priv, int fd, void *buf, size_t size)
{
   (void)priv;
   ssize_t n;
   do {
      n = read(fd, buf, size);
   } while (n < 0 && errno == EINTR);
   return n;
}

static const struct mali_kbase_backend os_backend = {
   .priv = NULL,
   .open = os_open,
   .close = os_close,
   .ioctl = os_ioctl,
   .mmap = os_mmap,
   .munmap = os_munmap,
   .page_size = os_page_size,
   .poll = os_poll,
   .read = os_read,
};

const struct mali_kbase_backend *
mali_kbase_os_backend(void)
{
   return &os_backend;
}

/* ---------------------------------------------------------------------- */
/* Results and logging                                                     */

const char *
mali_kbase_result_str(enum mali_kbase_result r)
{
   switch (r) {
   case MALI_KBASE_SUCCESS: return "success";
   case MALI_KBASE_ERROR_KERNEL: return "kernel call failed";
   case MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY: return "out of host memory";
   case MALI_KBASE_ERROR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
   case MALI_KBASE_ERROR_BUSY: return "busy";
   case MALI_KBASE_ERROR_DEVICE_NOT_FOUND: return "device not found";
   case MALI_KBASE_ERROR_PERMISSION_DENIED: return "permission denied";
   case MALI_KBASE_ERROR_INCOMPATIBLE_KERNEL: return "incompatible kernel";
   case MALI_KBASE_ERROR_INVALID_ARGUMENT: return "invalid argument";
   case MALI_KBASE_ERROR_MAP_FAILED: return "mmap failed";
   case MALI_KBASE_ERROR_MALFORMED_PROPERTIES: return "malformed GPU properties";
   }
   return "unknown result";
}

enum mali_kbase_result
kb_result_from_errno(int err, enum kb_errno_kind kind)
{
   switch (err) {
   case EBUSY:
      return MALI_KBASE_ERROR_BUSY;
   case ENOMEM:
      return kind == KB_ERRNO_DEVICE_MEMORY ? MALI_KBASE_ERROR_OUT_OF_DEVICE_MEMORY
                                            : MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY;
   default:
      return MALI_KBASE_ERROR_KERNEL;
   }
}

void
kb_log(mali_kbase_log_fn fn, void *user, const char *fmt, ...)
{
   char buf[512];
   int n = snprintf(buf, sizeof(buf), "malisx2/kbase: ");
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(buf + n, sizeof(buf) - (size_t)n, fmt, ap);
   va_end(ap);

   if (fn)
      fn(user, buf);
   else
      fprintf(stderr, "%s\n", buf);
}
