# SPDX-License-Identifier: MIT
#
# Build glue for the Mesa sources in ../mesa (see ../meson.build).
#
# Mesa's own meson.build files are not used: they hang off Mesa's top-level
# build and its option set. The files under mesa-glue/ rebuild the pieces we
# use with the same sources, generators and compile flags, and are derived
# from Mesa's meson.build files at the pinned commit. The directory tree
# mirrors Mesa's src/ so that generated headers (nir_opcodes.h, v11_pack.h,
# ...) end up at the include paths the Mesa sources expect.
#
# When the Mesa pin moves, diff Mesa's meson.build files for the directories
# built here against the previous pin and carry source-list changes over.

mesa_dir = meson.project_source_root() / 'mesa'
mesa_src = mesa_dir / 'src'

# Preprocessor configuration, mirroring the checks in Mesa's top-level
# meson.build that the Mesa sources look at.
pre_args = [
  '-D__STDC_CONSTANT_MACROS',
  '-D__STDC_FORMAT_MACROS',
  '-D__STDC_LIMIT_MACROS',
  '-DPACKAGE_VERSION="libmali-@0@"'.format(meson.project_version()),
  '-DMESA_DEBUG=@0@'.format((get_option('buildtype') == 'debug').to_int()),
  '-DHAVE_OPENGL=0',
  '-DHAVE_OPENGL_ES_1=0',
  '-DHAVE_OPENGL_ES_2=0',
  '-DTHREAD_SANITIZER=0',
]

with_platform_android = host_machine.system() == 'android'
if with_platform_android
  pre_args += ['-DHAVE_ANDROID_PLATFORM', '-D__INTRODUCED_IN(api_level)=']
  # The Android window-system platform.
  # ANDROID_API_LEVEL stays undefined: the runtime's AHardwareBuffer code
  # (API level >= 26 paths) is not built.
  pre_args += '-DVK_USE_PLATFORM_ANDROID_KHR'
endif

if ['linux', 'android'].contains(host_machine.system())
  pre_args += '-D_GNU_SOURCE'
endif

foreach b : ['bswap32', 'bswap64', 'clz', 'clzll', 'ctz', 'expect', 'ffs',
             'ffsll', 'popcount', 'popcountll', 'unreachable', 'types_compatible_p']
  if cc.has_function(b)
    pre_args += '-DHAVE___BUILTIN_@0@'.format(b.to_upper())
  endif
endforeach
foreach b : ['add_overflow', 'add_overflow_p', 'sub_overflow_p',
             'aarch64_get_fpcr', 'aarch64_set_fpcr']
  if cc.has_function('__builtin_' + b)
    pre_args += '-DHAVE___BUILTIN_@0@'.format(b.to_upper())
  endif
endforeach

_attributes = [
  'const', 'flatten', 'malloc', 'pure', 'unused', 'warn_unused_result',
  'weak', 'format', 'packed', 'returns_nonnull', 'alias', 'noreturn',
  'optimize', 'cold',
]
foreach a : cc.get_supported_function_attributes(_attributes)
  pre_args += '-DHAVE_FUNC_ATTRIBUTE_@0@'.format(a.to_upper())
endforeach
if cc.has_function_attribute('visibility:hidden')
  pre_args += '-DHAVE_FUNC_ATTRIBUTE_VISIBILITY'
endif

if cc.compiles('__uint128_t foo(void) { return 0; }', name : '__uint128_t')
  pre_args += '-DHAVE_UINT128'
endif

foreach f : ['reallocarray', 'fmemopen', 'mallinfo2', 'posix_memalign',
             'dl_iterate_phdr']
  if cc.has_function(f)
    pre_args += '-DHAVE_@0@'.format(f.to_upper())
  endif
endforeach

dep_atomic = dependency('', required : false)
if cc.compiles('''#include <stdint.h>
                  int main() {
                    struct { uint64_t *v; } x;
                    return (int)__atomic_load_n(x.v, __ATOMIC_ACQUIRE) &
                           (int)__atomic_add_fetch(x.v, (uint64_t)1, __ATOMIC_ACQ_REL);
                  }''',
               name : 'GCC atomic builtins')
  pre_args += '-DUSE_GCC_ATOMIC_BUILTINS'
endif

if (cc.has_header_symbol('sys/sysmacros.h', 'major') and
    cc.has_header_symbol('sys/sysmacros.h', 'minor') and
    cc.has_header_symbol('sys/sysmacros.h', 'makedev'))
  pre_args += '-DMAJOR_IN_SYSMACROS'
endif

if cc.check_header('sched.h')
  pre_args += '-DHAS_SCHED_H'
  if cc.has_function('sched_getaffinity')
    pre_args += '-DHAS_SCHED_GETAFFINITY'
  endif
endif

foreach h : ['xlocale.h', 'linux/futex.h', 'endian.h', 'dlfcn.h',
             'pthread_np.h', 'poll.h', 'sys/inotify.h']
  if cc.check_header(h)
    pre_args += '-DHAVE_@0@'.format(h.to_upper().underscorify())
  endif
endforeach

_functions = {
  'strtof' : '',
  'mkostemp' : '',
  'memfd_create' : '#include <sys/mman.h>',
  'random_r' : '',
  'flock' : '',
  'strtok_r' : '',
  'getrandom' : '#include <sys/random.h>',
  'posix_fallocate' : '',
  'secure_getenv' : '',
  'sysconf' : '#include <unistd.h>',
}
foreach f, prefix : _functions
  if cc.has_function(f, args : pre_args, prefix : prefix)
    pre_args += '-DHAVE_@0@'.format(f.to_upper())
  endif
endforeach

if cc.has_header_symbol('unistd.h', '_SC_PHYS_PAGES')
  pre_args += '-DHAVE_SC_PHYS_PAGES'
endif

if cpp.links('''
    #define _GNU_SOURCE
    #include <stdlib.h>
    static int dcomp(const void *l, const void *r, void *t) { return 0; }
    int main(int ac, char **av) {
      int arr[] = { 1 };
      void *t = NULL;
      qsort_r((void*)&arr[0], 1, 1, dcomp, t);
      return (0);
    }''',
    args : pre_args,
    name : 'GNU qsort_r')
  pre_args += '-DHAVE_GNU_QSORT_R'
endif

if cc.has_header_symbol('time.h', 'struct timespec')
  pre_args += '-DHAVE_STRUCT_TIMESPEC'
endif

with_c11_threads = false
if with_platform_android and cc.has_function('thrd_create', prefix : '#include <threads.h>')
  pre_args += '-DHAVE_THRD_CREATE'
  with_c11_threads = true
endif

if cc.has_header_symbol('errno.h', 'program_invocation_name', args : '-D_GNU_SOURCE')
  pre_args += '-DHAVE_PROGRAM_INVOCATION_NAME'
endif
if cc.has_header_symbol('math.h', 'issignaling', args : '-D_GNU_SOURCE')
  pre_args += '-DHAVE_ISSIGNALING'
endif
if cc.has_member('struct dirent', 'd_type',
                 prefix : '#include <sys/types.h>\n#include <dirent.h>')
  pre_args += '-DHAVE_DIRENT_D_TYPE'
endif

if cc.links('''
    #define _GNU_SOURCE
    #include <stdlib.h>
    #include <locale.h>
    int main() {
      locale_t loc = newlocale(LC_CTYPE_MASK, "C", NULL);
      char *end;
      double d = strtod_l("1.0", &end, loc);
      float f = strtof_l("1.0", &end, loc);
      freelocale(loc);
      return 0;
    }''',
    args : pre_args,
    name : 'strtod has locale support')
  pre_args += '-DHAVE_STRTOD_L'
endif

dep_dl = dependency('', required : false)
if not cc.has_function('dlopen')
  dep_dl = cc.find_library('dl', required : true)
endif
if cc.has_function('dladdr', dependencies : dep_dl)
  pre_args += '-DHAVE_DLADDR'
endif

dep_m = cc.find_library('m', required : false)
dep_thread = dependency('threads')
pre_args += '-DHAVE_PTHREAD'
if cc.has_function('pthread_setaffinity_np', dependencies : dep_thread,
                   prefix : '#include <pthread.h>', args : '-D_GNU_SOURCE')
  pre_args += '-DHAVE_PTHREAD_SETAFFINITY'
endif

# No zlib/zstd: without them Mesa's disk cache compiles to stubs. The
# Vulkan pipeline cache is the driver's own.

# Compiler flags Mesa builds its sources with. The semantic ones
# (-fno-math-errno, -fno-trapping-math, -fno-common) matter; the warning
# flags keep the Mesa code building quietly.
_trial_c = [
  '-Werror=implicit-function-declaration',
  '-Werror=return-type',
  '-Werror=incompatible-pointer-types',
  '-Werror=int-conversion',
  '-Wno-missing-field-initializers',
  '-Wno-format-truncation',
  '-Wno-nonnull-compare',
  '-Wno-unknown-pragmas',
  '-Wno-unused-function',
  '-Wno-unused-parameter',
  '-Wno-sign-compare',
  '-Wno-override-init',
  '-Wno-initializer-overrides',
  '-Wno-pointer-arith',
  '-Wno-misleading-indentation',
  '-fno-math-errno',
  '-fno-trapping-math',
  '-fno-common',
  '-ffunction-sections',
  '-fdata-sections',
]
_trial_cpp = [
  '-Werror=return-type',
  '-Wno-missing-field-initializers',
  '-Wno-non-virtual-dtor',
  '-Wno-unknown-pragmas',
  '-Wno-unused-parameter',
  '-Wno-sign-compare',
  '-fno-math-errno',
  '-fno-trapping-math',
  '-flifetime-dse=1',
  '-ffunction-sections',
  '-fdata-sections',
]
if get_option('b_ndebug') == 'true' or (get_option('b_ndebug') == 'if-release' and get_option('buildtype') == 'release')
  _trial_c += ['-Wno-unused-variable', '-Wno-unused-but-set-variable']
  _trial_cpp += ['-Wno-unused-variable', '-Wno-unused-but-set-variable']
endif

mesa_c_args = pre_args + cc.get_supported_arguments(_trial_c)
mesa_cpp_args = pre_args + cpp.get_supported_arguments(_trial_cpp)

# Rust flags from Mesa's top-level meson.build.
bindgen_version = find_program('bindgen').version()
if bindgen_version in ['unknown', '0.72.0', '0.73.0', '0.73.1']
  error('bindgen @0@ is known to be buggy (Mesa rejects it)'.format(bindgen_version))
elif bindgen_version.version_compare('< 0.71.1')
  error('bindgen 0.71.1 or newer is required')
endif
if rustc.version().version_compare('< 1.85')
  error('Rust 1.85.0 or newer is required')
endif

bindgen_output_args = [
  '--wrap-unsafe-ops',
  '--raw-line', '#![allow(clippy::all)]',
  '--raw-line', '#![allow(warnings)]',
  '--raw-line', '#![allow(unsafe_op_in_unsafe_fn)]',
]
rust_global_args = [
  '-Aclippy::collapsible_else_if',
  '-Aclippy::needless_lifetimes',
  '-Aclippy::question_mark',
]
rust_2024_lint_args = [
  '-Dboxed_slice_into_iter',
  '-Ddeprecated_safe_2024',
  '-Dimpl_trait_overcaptures',
  '-Dkeyword_idents_2024',
  '-Dmissing_unsafe_on_extern',
  '-Dnever_type_fallback_flowing_into_unsafe',
  '-Drust_2024_guarded_string_incompatible_syntax',
  '-Drust_2024_incompatible_pat',
  '-Drust_2024_prelude_collisions',
  '-Dstatic_mut_refs',
  '-Dunsafe_op_in_unsafe_fn',
]

inc_include = [include_directories('../mesa/include')]
if with_platform_android
  # Mesa's stand-ins for AOSP headers the NDK lacks.
  inc_include += include_directories('../mesa/include/android_stub')
endif

subdir('src')
