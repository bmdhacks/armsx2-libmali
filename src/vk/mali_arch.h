/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Per-arch symbol naming, so a function compiled once per GPU architecture
 * gets a distinct linker symbol instead of colliding with its other-arch
 * copy in the same .so. Mirrors panvk's panvk_per_arch()/panvk_macros.h.
 *
 * MALI_PER_ARCH() needs PAN_ARCH (the same precondition genxml/gen_macros.h
 * has); the declaration and dispatch helpers below do not, so headers that
 * code compiled once also includes can use them.
 */

/* Outside the include guard: a translation unit compiled once may get
 * PAN_ARCH only after a first include (from mali_cs.h), and a later
 * include then defines MALI_PER_ARCH. */
#if defined(PAN_ARCH) && !defined(MALI_PER_ARCH)
#if PAN_ARCH == 9
#define MALI_PER_ARCH(name) mali_v9_##name
#elif PAN_ARCH == 11
#define MALI_PER_ARCH(name) mali_v11_##name
#else
#error "no per-arch naming for this PAN_ARCH"
#endif
#endif

#ifndef MALI_ARCH_H
#define MALI_ARCH_H

/*
 * Declares both arches' variants of a per-arch function:
 *    MALI_PER_ARCH_DECL(void, pack_dummy_sampler, (void *out));
 * declares mali_v9_pack_dummy_sampler and mali_v11_pack_dummy_sampler.
 * Per-arch code calls its own with MALI_PER_ARCH(name)(...), code compiled
 * once picks one with mali_arch_dispatch.
 */
#define MALI_PER_ARCH_DECL(ret, name, params)                                  \
   ret mali_v9_##name params;                                                  \
   ret mali_v11_##name params

/*
 * For code compiled once (not per arch): call the variant that matches a
 * device found at runtime to have this arch. Only for cold paths (device
 * creation, pipeline-cache import, sync-type and image-layout choices);
 * never on the per-draw path, which reaches the right variant directly
 * through the device's dispatch table.
 */
#define mali_arch_dispatch(arch, name, ...) \
   ((arch) == 9 ? mali_v9_##name(__VA_ARGS__) : mali_v11_##name(__VA_ARGS__))

#endif
