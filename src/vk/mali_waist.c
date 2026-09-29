/*
 * Copyright 2026 The libmali authors
 * SPDX-License-Identifier: MIT
 */

/*
 * What the generated waist stubs do (mali_waist.h). This file is compiled
 * twice, with and without MALI_MEASUREMENT_BUILD, so that the tests can
 * check both behaviours of a void stub; the driver links the one the meson
 * option selects.
 */

#include "mali_waist.h"

#include <stdlib.h>
#include <string.h>

#include "util/log.h"
#include "util/u_atomic.h"

#ifndef MALI_MEASUREMENT_BUILD
#define MALI_MEASUREMENT_BUILD 0
#endif

static uint64_t skipped_calls;

static const char *
class_text(char cls)
{
   switch (cls) {
   case 'a': return "loader-required, not implemented yet";
   case 'b': return "in the ARMSX2 waist, not implemented yet";
   case 'c':
   case 'd': return "outside the ARMSX2 waist";
   default:  return "unclassified";
   }
}

/* Count the call; true on the first one, which is the one we log. */
static bool
count_call(struct mali_waist_entry *e)
{
   return p_atomic_inc_return(&e->calls) == 1;
}

void
mali_waist_void_call(struct mali_waist_entry *e)
{
   bool first = count_call(e);

   if (MALI_MEASUREMENT_BUILD) {
      mesa_loge("libmali: %s called (%s); aborting (measurement build)",
                e->name, class_text(e->cls));
      abort();
   }

   p_atomic_inc(&skipped_calls);
   if (first)
      mesa_loge("libmali: %s called (%s); call skipped", e->name,
                class_text(e->cls));
}

VkResult
mali_waist_result_call(struct mali_waist_entry *e)
{
   if (count_call(e))
      mesa_loge("libmali: %s called (%s); returning VK_ERROR_FEATURE_NOT_PRESENT",
                e->name, class_text(e->cls));
   return MALI_WAIST_ERROR;
}

void
mali_waist_value_call(struct mali_waist_entry *e)
{
   if (count_call(e))
      mesa_loge("libmali: %s called (%s); returning 0", e->name,
                class_text(e->cls));
}

const struct mali_waist_entry *
mali_waist_find(const char *name)
{
   for (unsigned i = 0; i < mali_waist_entry_count; i++) {
      const struct mali_waist_entry *e = &mali_waist_entries[i];
      if (strcmp(e->name, name) == 0)
         return e;
      /* names is a space-separated list; match whole words. */
      size_t len = strlen(name);
      for (const char *p = strstr(e->names, name); p; p = strstr(p + 1, name)) {
         if ((p == e->names || p[-1] == ' ') && (p[len] == ' ' || p[len] == '\0'))
            return e;
      }
   }
   return NULL;
}

uint64_t
mali_waist_skipped_calls(void)
{
   return p_atomic_read(&skipped_calls);
}

bool
mali_waist_aborts(void)
{
   return MALI_MEASUREMENT_BUILD;
}
