/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The job-manager frontend: atom-number bookkeeping, JOB_SUBMIT, the
 * non-blocking event read and poll-based wait, POST_TERM, and the
 * sync-file stream a FENCE_TRIGGER soft atom's base_fence names. This is
 * the JM counterpart of csf.c; it is compiled unconditionally (like
 * context.c and mem.c) but every entry point here is only ever called on
 * a context mali_kbase_create negotiated as MALI_KBASE_FRONTEND_JM.
 *
 * What is NOT here: anything that interprets core_req, dependencies or
 * udata, and anything that builds job chains. The queue/command-buffer
 * layer above this one owns that; this layer only moves bytes to and from
 * the kernel.
 */

#include <errno.h>
#include <poll.h>
#include <string.h>

#include "kbase_priv.h"

/* ---------------------------------------------------------------------- */
/* Atom numbers                                                            */

void
mali_kbase_jm_atom_ids_init(struct mali_kbase_jm_atom_ids *ids)
{
   memset(ids, 0, sizeof(*ids));
   /* Atom number 0 means "no dependency" in a job-chain's dependency
    * fields and is never handed out: mark it permanently in use. */
   ids->bitmap[0] |= 1u;
}

static inline bool
bit_test(const uint8_t *bitmap, unsigned n)
{
   return (bitmap[n / 8] >> (n % 8)) & 1u;
}

static inline void
bit_set(uint8_t *bitmap, unsigned n)
{
   bitmap[n / 8] |= (uint8_t)(1u << (n % 8));
}

static inline void
bit_clear(uint8_t *bitmap, unsigned n)
{
   bitmap[n / 8] &= (uint8_t)~(1u << (n % 8));
}

unsigned
mali_kbase_jm_atom_ids_alloc(struct mali_kbase_jm_atom_ids *ids, uint8_t *out, unsigned n)
{
   unsigned got = 0;
   for (unsigned id = 1; id < 256 && got < n; id++) {
      if (bit_test(ids->bitmap, id))
         continue;
      bit_set(ids->bitmap, id);
      out[got++] = (uint8_t)id;
   }
   return got;
}

void
mali_kbase_jm_atom_ids_free(struct mali_kbase_jm_atom_ids *ids, uint8_t id)
{
   if (id != 0)
      bit_clear(ids->bitmap, id);
}

bool
mali_kbase_jm_atom_ids_used(const struct mali_kbase_jm_atom_ids *ids, uint8_t id)
{
   return bit_test(ids->bitmap, id);
}

unsigned
mali_kbase_jm_atom_ids_free_count(const struct mali_kbase_jm_atom_ids *ids)
{
   unsigned used = 0;
   for (unsigned id = 0; id < 256; id++)
      used += bit_test(ids->bitmap, id);
   return 255 - (used - 1); /* id 0 is permanently "used" and not a number */
}

/* ---------------------------------------------------------------------- */
/* Submission                                                              */

/* At most 256 atoms per JOB_SUBMIT call. Stock kbase has no per-call
 * limit, but the T820's kernel (Unisoc's kbase_jd_submit) records the
 * number of every atom of one call in a fixed array of 256 entries,
 * without a bounds check: a longer call would write past the array into
 * kernel memory. Do not raise this. */
#define MALI_KBASE_JM_MAX_ATOMS_PER_SUBMIT 256

enum mali_kbase_result
mali_kbase_jm_submit(struct mali_kbase *kb, const struct kb_jm_atom *atoms, unsigned n)
{
   if (!kb || (n && !atoms))
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;

   for (unsigned off = 0; off < n; off += MALI_KBASE_JM_MAX_ATOMS_PER_SUBMIT) {
      unsigned batch = n - off;
      if (batch > MALI_KBASE_JM_MAX_ATOMS_PER_SUBMIT)
         batch = MALI_KBASE_JM_MAX_ATOMS_PER_SUBMIT;

      struct kb_jm_job_submit s = {
         .addr = (uint64_t)(uintptr_t)(atoms + off),
         .nr_atoms = batch,
         .stride = sizeof(struct kb_jm_atom),
      };
      long ret = kb_ioctl(kb, KB_JM_IOCTL_JOB_SUBMIT, &s);
      if (ret < 0) {
         KB_LOG(kb, "JOB_SUBMIT of %u atoms (offset %u of %u) failed: %s", batch, off, n,
                strerror((int)-ret));
         return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
      }
   }
   return MALI_KBASE_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* Completion                                                              */

enum mali_kbase_result
mali_kbase_jm_read_events(struct mali_kbase *kb, struct kb_jm_event *ev, unsigned max,
                          unsigned *n, bool *terminated)
{
   if (!kb || !ev || !max || !n)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   *n = 0;
   if (terminated)
      *terminated = false;

   long r = kb->backend->read(kb->backend->priv, kb->fd, ev,
                              (size_t)max * sizeof(*ev));
   if (r < 0) {
      int err = errno;
      if (err == EAGAIN || err == EWOULDBLOCK)
         return MALI_KBASE_SUCCESS;
      if (err == EPIPE) {
         /* kbase_event_close/kbase_event_pending, JM build: once
          * POST_TERM has drained the queued events, read() answers
          * EPIPE forever. */
         if (terminated)
            *terminated = true;
         return MALI_KBASE_SUCCESS;
      }
      KB_LOG(kb, "reading job-manager events failed: %s", strerror(err));
      return kb_result_from_errno(err, KB_ERRNO_GENERIC);
   }

   *n = (unsigned)(r / (long)sizeof(*ev));
   return MALI_KBASE_SUCCESS;
}

int
mali_kbase_jm_poll(struct mali_kbase *kb, int timeout_ms)
{
   if (!kb->backend->poll) {
      errno = ENOSYS;
      return -1;
   }
   struct pollfd pfd = {.fd = kb->fd, .events = POLLIN};
   return kb->backend->poll(kb->backend->priv, &pfd, 1, timeout_ms);
}

enum mali_kbase_result
mali_kbase_jm_post_term(struct mali_kbase *kb)
{
   long ret = kb_ioctl(kb, KB_JM_IOCTL_POST_TERM, NULL);
   if (ret < 0) {
      KB_LOG(kb, "POST_TERM failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   return MALI_KBASE_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* Sync-file streams                                                       */

enum mali_kbase_result
mali_kbase_stream_create(struct mali_kbase *kb, const char *name, int *out_fd)
{
   if (!kb || !out_fd)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;

   struct kb_jm_stream_create sc;
   memset(&sc, 0, sizeof(sc));
   if (name)
      strncpy(sc.name, name, sizeof(sc.name) - 1);

   long ret = kb_ioctl(kb, KB_JM_IOCTL_STREAM_CREATE, &sc);
   if (ret < 0) {
      KB_LOG(kb, "STREAM_CREATE(\"%s\") failed: %s", sc.name, strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   *out_fd = (int)ret;
   return MALI_KBASE_SUCCESS;
}
