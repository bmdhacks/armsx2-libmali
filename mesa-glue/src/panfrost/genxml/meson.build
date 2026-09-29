# SPDX-License-Identifier: MIT
# Derived from Mesa src/panfrost/genxml/meson.build. Only the common and
# v11 packers are generated: v11 is the architecture we target.

_vg = mesa_src / 'panfrost' / 'genxml'

pan_packers = []
foreach packer : ['common', 'v@0@'.format(pan_arch)]
  pan_packers += custom_target(
    packer + '_pack.h',
    input : files(_vg / 'gen_pack.py', _vg / (packer + '.xml')),
    output : packer + '_pack.h',
    command : [prog_python, '@INPUT@'],
    capture : true,
  )
endforeach

idep_pan_packers = declare_dependency(
  sources : [pan_packers],
  include_directories : include_directories('.', '../../../../mesa/src/panfrost/genxml'),
)
