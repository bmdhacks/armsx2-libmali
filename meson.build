# SPDX-License-Identifier: MIT
#
# libmali: Vulkan driver for Arm Mali (kbase kernel driver).
#
# Layout:
#   src/           the driver
#   mesa-glue/     our meson build for the Mesa sources the driver uses; its
#                  tree mirrors Mesa's src/ so generated headers land where
#                  the Mesa sources expect them
#   mesa/          Mesa, not in this repository (see mesa_commit below)
#   subprojects/   Rust crates kraid needs (meson downloads them)
#   cross/         meson cross files

project(
  'libmali',
  ['c', 'cpp'],
  version : '0.0.1',
  license : 'MIT',
  meson_version : '>= 1.7.0',
  default_options : [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'c_std=c11',
    'cpp_std=c++17',
    'rust_std=2021',
    'build.rust_std=2021',
    'warning_level=1',
  ],
)

# The GPU architecture we build the driver's device entry points for
# (Mesa numbering). The G615 in the RG 477V reports arch 11; this is the
# only arch the vk/ sources in libmali_vk are compiled for today.
pan_arch = 11

# Every arch we generate genxml packers and a pixel-format library for.
# arch 9 (G57/G77, job manager) has no driver code yet (no vk/ source
# builds against it); this only proves the genxml side builds for it, so
# the arch's eventual driver files have packers and pan_format ready to
# link against.
pan_archs = [9, pan_arch]

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
prog_python = import('python').find_installation('python3', modules : ['mako', 'yaml'])
fs = import('fs')

# kraid (the only shader compiler backend we use) is Rust.
add_languages('rust', required : true, native : false)
add_languages('rust', required : true, native : true)
rustc = meson.get_compiler('rust')
rust = import('rust')

# Mesa: branch libmali of github.com/bmdhacks/armsx2-turnip (upstream Mesa
# plus our kraid changes) at this commit, checked out (or symlinked) at
# mesa/. Only the sources are used; Mesa's own build is not run.
mesa_commit = '730a6284cb7a88f6c19a07f1caea35c7a2ae64f2'
if not fs.is_dir('mesa' / 'src' / 'panfrost')
  error('''Mesa is missing. Fetch it with:
  git init mesa && git -C mesa fetch --depth 1 https://github.com/bmdhacks/armsx2-turnip.git @0@ && git -C mesa checkout FETCH_HEAD'''.format(mesa_commit))
endif
_mesa_head = run_command('git', '-C', meson.project_source_root() / 'mesa',
                         'rev-parse', 'HEAD', check : false)
if _mesa_head.returncode() == 0 and _mesa_head.stdout().strip() != mesa_commit
  warning('mesa/ is at @0@, not the pinned @1@'.format(
          _mesa_head.stdout().strip(), mesa_commit))
endif

subdir('mesa-glue')
subdir('src')
if fs.is_dir('tests')
  subdir('tests')
endif
