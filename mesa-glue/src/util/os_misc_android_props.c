/*
 * Copyright 2026 The libmali authors
 * SPDX-License-Identifier: MIT
 */

/*
 * The Android property read os_misc.c is built with on Android (see
 * meson.build here). Mesa looks an option up with getenv() and then as three
 * properties ("debug.mesa.x", "vendor.mesa.x", "mesa.x"), and some options
 * are looked up again on every shader compile (debug_get_option() does not
 * cache). In an application's SELinux domain every vendor.* read is denied,
 * and each denial is a logcat line: about 400 per ARMSX2 corpus dump.
 *
 * So here:
 *  - vendor.* names are not read. The driver is loaded into an application;
 *    it has no vendor-partition configuration to honour, and the read is
 *    denied anyway.
 *  - every other name is read once per process and the answer kept. A debug
 *    property set after the first lookup is not seen; restart the app.
 */

#include <stdbool.h>
#include <string.h>

#include "util/simple_mtx.h"

/* This file is built with os_misc.c, whose flags rename the call; here it
 * has to reach Bionic's function. */
#undef __system_property_get

/* Bionic's, from <sys/system_properties.h> (not included, so the host test
 * can provide a fake). */
int __system_property_get(const char *name, char *value);
int mali_os_property_get(const char *name, char *value);

/* PROP_VALUE_MAX; os_misc.c uses 128 for names on API 26+. */
#define MALI_PROP_VALUE_MAX 92
#define MALI_PROP_NAME_MAX 128
/* Mesa reads a few dozen options; past this, names are read uncached. */
#define MALI_PROP_CACHE_SIZE 128

struct prop_entry {
   char name[MALI_PROP_NAME_MAX];
   char value[MALI_PROP_VALUE_MAX];
   int len;
};

static struct prop_entry prop_cache[MALI_PROP_CACHE_SIZE];
static unsigned prop_count;
static simple_mtx_t prop_mtx = SIMPLE_MTX_INITIALIZER;

int
mali_os_property_get(const char *name, char *value)
{
   value[0] = '\0';
   if (strncmp(name, "vendor.", 7) == 0)
      return 0;
   if (strlen(name) >= MALI_PROP_NAME_MAX)
      return __system_property_get(name, value);

   simple_mtx_lock(&prop_mtx);
   for (unsigned i = 0; i < prop_count; i++) {
      if (strcmp(prop_cache[i].name, name) == 0) {
         memcpy(value, prop_cache[i].value, MALI_PROP_VALUE_MAX);
         int len = prop_cache[i].len;
         simple_mtx_unlock(&prop_mtx);
         return len;
      }
   }

   char buf[MALI_PROP_VALUE_MAX] = {0};
   int len = __system_property_get(name, buf);
   if (len < 0)
      len = 0;
   if (prop_count < MALI_PROP_CACHE_SIZE) {
      struct prop_entry *e = &prop_cache[prop_count++];
      strcpy(e->name, name);
      memcpy(e->value, buf, MALI_PROP_VALUE_MAX);
      e->len = len;
   }
   simple_mtx_unlock(&prop_mtx);

   memcpy(value, buf, MALI_PROP_VALUE_MAX);
   return len;
}
