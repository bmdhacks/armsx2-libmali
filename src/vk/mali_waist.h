/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The waist policy: what an entry point does when
 * neither the driver nor Mesa's common code implements it. The stubs are
 * generated from waist.txt (gen_waist_stubs.py) and sit under the driver's
 * and the common entry points in every dispatch table, so every name in
 * waist.txt resolves to something non-NULL:
 *
 *   void            logged once per entry point, then skipped; in a
 *                   measurement build (meson -Dmeasurement=true) the process
 *                   aborts instead
 *   VkResult        logged once, returns MALI_WAIST_ERROR
 *   anything else   logged once, returns 0
 *
 * Every call is counted, logged or not.
 */

#ifndef MALI_WAIST_H
#define MALI_WAIST_H

#include <stdbool.h>
#include <stdint.h>

#include <vulkan/vulkan_core.h>

/* What an unimplemented VkResult entry point returns. */
#define MALI_WAIST_ERROR VK_ERROR_FEATURE_NOT_PRESENT

struct mali_waist_entry {
   const char *name;     /* core name, e.g. "vkCreateEvent" */
   const char *names;    /* every waist.txt name it covers (aliases) */
   char cls;             /* class: 'a', 'b', 'c' or 'd' */
   bool nonnull;         /* one of the 49 names that must resolve */
   uint32_t calls;       /* number of calls so far */
};

/* Generated table, sorted by name. */
extern struct mali_waist_entry mali_waist_entries[];
extern const unsigned mali_waist_entry_count;

/* Called by the generated stubs. */
void mali_waist_void_call(struct mali_waist_entry *e);
VkResult mali_waist_result_call(struct mali_waist_entry *e);
void mali_waist_value_call(struct mali_waist_entry *e);

/* The entry for a name (core name or alias), or NULL. */
const struct mali_waist_entry *mali_waist_find(const char *name);

/*
 * Void stub calls that were skipped, over the whole process. A skipped
 * vkCmd* call leaves a hole in what the command buffer does; the
 * command-stream capture records this counter so a capture taken after a
 * skip is marked as incomplete.
 */
uint64_t mali_waist_skipped_calls(void);

/* True in a measurement build: void stubs abort. */
bool mali_waist_aborts(void);

#endif
