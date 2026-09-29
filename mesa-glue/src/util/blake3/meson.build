# SPDX-License-Identifier: MIT
# Derived from Mesa src/util/blake3/meson.build.

_vb3 = mesa_src / 'util' / 'blake3'
_b3_files = files(_vb3 / 'blake3.c', _vb3 / 'blake3_dispatch.c', _vb3 / 'blake3_portable.c')
_b3_defs = []
if host_machine.cpu_family() == 'x86_64'
  _b3_files += files(_vb3 / 'blake3_sse2_x86-64_unix.S', _vb3 / 'blake3_sse41_x86-64_unix.S',
                  _vb3 / 'blake3_avx2_x86-64_unix.S', _vb3 / 'blake3_avx512_x86-64_unix.S')
elif host_machine.cpu_family() == 'aarch64' and host_machine.endian() == 'little'
  _b3_files += files(_vb3 / 'blake3_neon.c')
else
  _b3_defs += ['-DBLAKE3_NO_SSE2', '-DBLAKE3_NO_SSE41', '-DBLAKE3_NO_AVX2', '-DBLAKE3_NO_AVX512']
endif

_blake3 = static_library(
  'blake3',
  _b3_files,
  c_args : _b3_defs + cc.get_supported_arguments('-Wno-error', '-Wno-unused-function'),
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

idep_blake3 = declare_dependency(link_with : _blake3)
