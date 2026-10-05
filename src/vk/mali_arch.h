/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Per-arch symbol naming, so a function compiled once per GPU architecture
 * gets a distinct linker symbol instead of colliding with its other-arch
 * copy in the same .so. Mirrors panvk's panvk_per_arch()/panvk_macros.h.
 *
 * A file that uses MALI_PER_ARCH() must already have PAN_ARCH defined
 * (the same precondition genxml/gen_macros.h has).
 */

#ifndef MALI_ARCH_H
#define MALI_ARCH_H

#ifndef PAN_ARCH
#error "PAN_ARCH must be defined before including mali_arch.h"
#endif

#if PAN_ARCH == 9
#define MALI_PER_ARCH(name) mali_v9_##name
#elif PAN_ARCH == 11
#define MALI_PER_ARCH(name) mali_v11_##name
#else
#error "no per-arch naming for this PAN_ARCH"
#endif

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
