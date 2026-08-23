dnl  Copyright (C) 2022-2026 Kamila Szewczyk
dnl
dnl  This program is free software; you can redistribute it and/or modify
dnl  it under the terms of the GNU General Public License as published by
dnl  the Free Software Foundation; version 3 of the License only.
dnl
dnl  This program is distributed in the hope that it will be useful,
dnl  but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
dnl  GNU General Public License for more details.
dnl
dnl  You should have received a copy of the GNU General Public License
dnl  along with this program. If not, see <http://www.gnu.org/licenses/>.

m4_define([xpar_version_major], [2])
m4_define([xpar_version_minor], [0])
m4_define([xpar_version], [xpar_version_major.xpar_version_minor])

AC_PREREQ([2.69])
AC_INIT([xpar],[xpar_version],[https://github.com/iczelia/xpar])
AC_CONFIG_HEADERS([config.h])
AC_COPYRIGHT([Copyright (C) 2022-2026 Kamila Szewczyk])
AC_LANG([C])
AC_CONFIG_MACRO_DIR([m4])

AC_CANONICAL_TARGET
AM_INIT_AUTOMAKE([-Wall color-tests subdir-objects dist-bzip2 dist-xz])

AC_PROG_INSTALL
AC_PROG_MAKE_SET
AC_PROG_CC
AC_USE_SYSTEM_EXTENSIONS
dnl  DJGPP's ar trips AM_PROG_AR's interface probe; short-circuit it.
case $host_os in
  msdosdjgpp*|djgpp*) am_cv_ar_interface=ar ;;
esac
AM_PROG_AR
AC_PROG_RANLIB
AC_CHECK_TOOL([OBJDUMP], [objdump])

AC_DEFINE([XPAR_MAJOR], [xpar_version_major], [Major version number])
AC_DEFINE([XPAR_MINOR], [xpar_version_minor], [Minor version number])

dnl  Identify the target in crash reports.
AC_DEFINE_UNQUOTED([XPAR_HOST_TRIPLE], ["$host"],
                   [Canonical host this build targets])

AC_C_RESTRICT
AC_C_INLINE
AC_C_BIGENDIAN

AC_SYS_LARGEFILE

AC_CHECK_HEADERS([sys/mman.h sys/stat.h sys/types.h fcntl.h unistd.h \
                  immintrin.h arm_neon.h sys/auxv.h linux/io_uring.h \
                  asm/hwprobe.h sys/syscall.h riscv_vector.h arm_sve.h \
                  altivec.h \
                  sys/xattr.h attr/xattr.h pwd.h grp.h sys/time.h \
                  sys/random.h sched.h])

case "$host_os" in
  msdosdjgpp*|djgpp*) xpar_is_dos=yes ;;
  *)                  xpar_is_dos=no  ;;
esac
AM_CONDITIONAL([XPAR_DOS], [test "x$xpar_is_dos" = "xyes"])
XPAR_DOS_CPPFLAGS=""
XPAR_DOS_CFLAGS=""
if test "x$xpar_is_dos" = "xyes"; then
  AC_DEFINE([XPAR_DOS], [1], [Define on MS-DOS/DJGPP hosts])
  dnl  DJGPP hides POSIX and DOS declarations behind __STRICT_ANSI__.
  XPAR_DOS_CPPFLAGS="-U__STRICT_ANSI__"
  dnl  Crash reporting walks frame pointers on DJGPP.
  XPAR_DOS_CFLAGS="-fno-omit-frame-pointer"
fi
AC_SUBST([XPAR_DOS_CPPFLAGS])
AC_SUBST([XPAR_DOS_CFLAGS])

case "$host_os" in
  mingw*|cygwin*|msys*) xpar_is_windows=yes ;;
  *)                    xpar_is_windows=no  ;;
esac
AM_CONDITIONAL([XPAR_IS_WINDOWS], [test "x$xpar_is_windows" = "xyes"])
if test "x$xpar_is_windows" = "xyes"; then
  AC_DEFINE([XPAR_WIN32], [1], [Define on Win32 hosts])
fi

AC_ARG_WITH([windows-target],
  [AS_HELP_STRING([--with-windows-target=@<:@nt|win95@:>@],
    [select the native Windows API and loader baseline (default: nt)])],
  [xpar_win_target=$withval], [xpar_win_target=nt])
case "$xpar_win_target" in
  nt)
    xpar_win_cppflags="-D_WIN32_WINNT=0x0600 -DWINVER=0x0600"
    xpar_win_cflags=""
    xpar_win_ldflags="-Wl,--major-subsystem-version=6"
    xpar_win_ldflags="$xpar_win_ldflags,--minor-subsystem-version=0"
    xpar_win_ldflags="$xpar_win_ldflags,--major-os-version=6"
    xpar_win_ldflags="$xpar_win_ldflags,--minor-os-version=0"
    ;;
  win95)
    xpar_win_cppflags="-D_WIN32_WINNT=0x0400 -DWINVER=0x0400"
    xpar_win_cppflags="$xpar_win_cppflags -DXPAR_WIN_LEGACY=1"
    dnl  Legacy crash reporting walks frame pointers.
    xpar_win_cflags="-march=i486 -mtune=i486 -mfpmath=387"
    xpar_win_cflags="$xpar_win_cflags -fno-omit-frame-pointer"
    xpar_win_ldflags="-Wl,--major-subsystem-version=4"
    xpar_win_ldflags="$xpar_win_ldflags,--minor-subsystem-version=0"
    xpar_win_ldflags="$xpar_win_ldflags,--major-os-version=4"
    xpar_win_ldflags="$xpar_win_ldflags,--minor-os-version=0"
    xpar_win_ldflags="$xpar_win_ldflags,--disable-dynamicbase"
    xpar_win_ldflags="$xpar_win_ldflags,--disable-nxcompat"
    ;;
  *)
    AC_MSG_ERROR([--with-windows-target must be "nt" or "win95"])
    ;;
esac

XPAR_WIN_CPPFLAGS=""
XPAR_WIN_CFLAGS=""
XPAR_WIN_LDFLAGS=""
if test "x$xpar_is_windows" = "xyes"; then
  if test "x$xpar_win_target" = "xwin95"; then
    case "$host_cpu" in
      i?86) ;;
      *) AC_MSG_ERROR([the win95 target requires a 32-bit x86 host]) ;;
    esac
  fi
  XPAR_WIN_CPPFLAGS="$xpar_win_cppflags"
  XPAR_WIN_CFLAGS="$xpar_win_cflags"
  XPAR_WIN_LDFLAGS="$xpar_win_ldflags"
  AC_MSG_NOTICE([Windows target: $xpar_win_target])
elif test "x$with_windows_target" != "x"; then
  AC_MSG_ERROR([--with-windows-target is valid only for native Windows])
else
  xpar_win_target=none
fi
AC_SUBST([XPAR_WIN_CPPFLAGS])
AC_SUBST([XPAR_WIN_CFLAGS])
AC_SUBST([XPAR_WIN_LDFLAGS])
AM_CONDITIONAL([XPAR_WIN_LEGACY],
  [test "x$xpar_win_target" = "xwin95"])

AM_CONDITIONAL([XPAR_IS_POSIX],
  [test "x$xpar_is_windows" = "xno" && test "x$xpar_is_dos" = "xno"])

dnl  Detect native backtrace and signal APIs.
AC_CHECK_HEADERS([execinfo.h signal.h ucontext.h])
AC_CHECK_FUNCS([backtrace sigaction])

AC_CHECK_FUNCS([mmap madvise posix_fadvise posix_memalign aligned_alloc \
                pread pwrite ftruncate fsync fdatasync getauxval \
                openat mkdirat renameat unlinkat linkat symlinkat \
                lchown fchownat lutimes utimensat futimens fchmodat fchmod \
                lsetxattr lgetxattr llistxattr fsetxattr \
                getpwuid_r getgrgid_r getpwnam_r getgrnam_r \
                link readlink symlink realpath getrandom arc4random_buf \
                sysconf getpagesize sched_getaffinity])

AC_MSG_CHECKING([for AT_SYMLINK_NOFOLLOW])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <fcntl.h>]], [[int f = AT_SYMLINK_NOFOLLOW;
                                             return f != 0;]])],
  [AC_MSG_RESULT([yes])
   AC_DEFINE([HAVE_AT_SYMLINK_NOFOLLOW], [1],
             [Define if AT_SYMLINK_NOFOLLOW is available])],
  [AC_MSG_RESULT([no])])

AC_ARG_ENABLE([threads],
  [AS_HELP_STRING([--disable-threads],[build without thread support])],
  [enable_threads=$enableval], [enable_threads=yes])
xpar_have_threads=no
if test "x$enable_threads" = "xyes"; then
  if test "x$xpar_is_windows" = "xyes"; then
    xpar_have_threads=yes
  else
    AC_CHECK_HEADER([pthread.h],
      [save_LIBS="$LIBS"
       AC_SEARCH_LIBS([pthread_create], [pthread],
         [xpar_have_threads=yes], [LIBS="$save_LIBS"])])
  fi
fi
AM_CONDITIONAL([XPAR_THREADS], [test "x$xpar_have_threads" = "xyes"])
if test "x$xpar_have_threads" = "xyes"; then
  AC_DEFINE([HAVE_THREADS], [1], [Define if thread support is available])
fi

if test "x$ac_cv_header_linux_io_uring_h" = "xyes"; then
  AC_MSG_CHECKING([for __NR_io_uring_setup])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[
      #include <sys/syscall.h>
      #include <linux/io_uring.h>
    ]], [[
      int n = __NR_io_uring_setup + __NR_io_uring_enter;
      return n != 0;
    ]])],
    [AC_MSG_RESULT([yes])
     AC_DEFINE([HAVE_IO_URING], [1], [Define if io_uring is available])],
    [AC_MSG_RESULT([no])])
fi

xpar_skip_simd=no
test "x$xpar_is_dos" = "xyes" && xpar_skip_simd=yes
test "x$xpar_win_target" = "xwin95" && xpar_skip_simd=yes

xpar_is_x86=no
xpar_is_x86_64=no
case "$host_cpu" in
  i?86)             xpar_is_x86=yes ;;
  x86_64*|amd64*)   xpar_is_x86=yes; xpar_is_x86_64=yes ;;
esac

AC_ARG_ENABLE([simd],
  [AS_HELP_STRING([--disable-simd],[build scalar kernels only])],
  [test "x$enableval" = "xno" && xpar_skip_simd=yes])

SSSE3_CFLAGS=""
SSE42_CFLAGS=""
AVX2_CFLAGS=""
AVX512_CFLAGS=""
GFNI_CFLAGS=""
GFNI512_CFLAGS=""
VBMI_CFLAGS=""
NEON_CFLAGS=""
PMULL_CFLAGS=""
SVE2_CFLAGS=""
VSX_CFLAGS=""
RVV_CFLAGS=""
RVV_CLMUL_CFLAGS=""
VPCLMUL_CFLAGS=""
xpar_have_neon=no

save_CFLAGS="$CFLAGS"

if test "x$xpar_skip_simd" = "xno"; then

if test "x$xpar_is_x86" = "xyes"; then
CFLAGS="$save_CFLAGS -mssse3"
AC_MSG_CHECKING([for SSSE3 intrinsics])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <immintrin.h>]], [[
     __m128i a = _mm_set1_epi8(1), b = _mm_shuffle_epi8(a, a);
     return _mm_cvtsi128_si32(b);
   ]])],
  [AC_MSG_RESULT([yes])
   AC_DEFINE([HAVE_SSSE3], [1], [Define if SSSE3 is available])
   SSSE3_CFLAGS="-mssse3"],
  [AC_MSG_RESULT([no])])
CFLAGS="$save_CFLAGS"
fi

case "$host_cpu" in
  x86_64*|amd64*)
    CFLAGS="$save_CFLAGS -msse4.2 -mavx2 -mvpclmulqdq"
    AC_MSG_CHECKING([for VPCLMULQDQ intrinsics])
    AC_COMPILE_IFELSE(
      [AC_LANG_PROGRAM([[#include <immintrin.h>]], [[
         __m256i a = _mm256_set1_epi64x(1);
         __m256i b = _mm256_clmulepi64_epi128(a, a, 0);
         return (int) _mm_crc32_u64(
           0, (unsigned long long) _mm256_extract_epi64(b, 0));
       ]])],
      [AC_MSG_RESULT([yes])
       AC_DEFINE([HAVE_VPCLMUL], [1], [Define if VPCLMULQDQ is available])
       VPCLMUL_CFLAGS="-msse4.2 -mavx2 -mvpclmulqdq"],
      [AC_MSG_RESULT([no])])
    CFLAGS="$save_CFLAGS"
    ;;
esac

for xpar_rvv_try in "-march=rv64gcv" "-march=rv32gcv" \
                    "-mcpu=generic_rv64+v" "-mcpu=generic_rv32+v"; do
  CFLAGS="$save_CFLAGS $xpar_rvv_try"
  AC_MSG_CHECKING([for RISC-V V intrinsics with '$xpar_rvv_try'])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[#include <riscv_vector.h>]], [[
       size_t vl = __riscv_vsetvl_e8m1(16);
       vuint8m1_t a = __riscv_vmv_v_x_u8m1(1, vl);
       a = __riscv_vrgather_vv_u8m1(a, a, vl);
       return (int) __riscv_vmv_x_s_u8m1_u8(a);
     ]])],
    [AC_MSG_RESULT([yes])
     AC_DEFINE([HAVE_RVV], [1], [Define if RISC-V V is available])
     RVV_CFLAGS="$xpar_rvv_try"],
    [AC_MSG_RESULT([no])])
  test "x$RVV_CFLAGS" != "x" && break
done
CFLAGS="$save_CFLAGS"

for xpar_rvvc_try in "-march=rv64gcv_zvbc" \
                     "-mcpu=generic_rv64+v+zvbc"; do
  CFLAGS="$save_CFLAGS $xpar_rvvc_try"
  AC_MSG_CHECKING([for RISC-V Zvbc intrinsics with '$xpar_rvvc_try'])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[#include <riscv_vector.h>]], [[
       size_t vl = __riscv_vsetvl_e64m1(1);
       vuint64m1_t a = __riscv_vmv_v_x_u64m1(1, vl);
       a = __riscv_vclmul_vx_u64m1(a, 45, vl);
       return (int) __riscv_vmv_x_s_u64m1_u64(a);
     ]])],
    [AC_MSG_RESULT([yes])
     AC_DEFINE([HAVE_RVV_CLMUL], [1],
               [Define if RISC-V Zvbc is available])
     RVV_CLMUL_CFLAGS="$xpar_rvvc_try"],
    [AC_MSG_RESULT([no])])
  test "x$RVV_CLMUL_CFLAGS" != "x" && break
done
CFLAGS="$save_CFLAGS"

CFLAGS="$save_CFLAGS -mvsx"
AC_MSG_CHECKING([for PowerPC VSX intrinsics])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <altivec.h>]], [[
     __vector unsigned char a = vec_splats((unsigned char) 1);
     __vector unsigned char b = vec_perm(a, a, a);
     return (int) vec_extract(b, 0);
   ]])],
  [AC_MSG_RESULT([yes])
   AC_DEFINE([HAVE_VSX], [1], [Define if PowerPC VSX is available])
   VSX_CFLAGS="-mvsx"],
  [AC_MSG_RESULT([no])])
CFLAGS="$save_CFLAGS"

CFLAGS="$save_CFLAGS -march=armv8.5-a+sve2"
AC_MSG_CHECKING([for SVE2 intrinsics])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <arm_sve.h>]], [[
     svbool_t p = svptrue_b8();
     svuint8_t a = svdup_n_u8(1);
     a = sveor_u8_x(p, a, svlsl_n_u8_x(p, a, 1));
     return (int) svlastb_u8(p, a);
   ]])],
  [AC_MSG_RESULT([yes])
   AC_DEFINE([HAVE_SVE2], [1], [Define if SVE2 is available])
   SVE2_CFLAGS="-march=armv8.5-a+sve2"],
  [AC_MSG_RESULT([no])])
CFLAGS="$save_CFLAGS"

CFLAGS="$save_CFLAGS -march=armv8-a+crypto"
AC_MSG_CHECKING([for ARM PMULL intrinsics])
dnl  The PMULL kernel also requires the 128-bit table lookup.
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <arm_neon.h>]], [[
     poly8x8_t a = vdup_n_p8((poly8_t) 1);
     poly16x8_t b = vmull_p8(a, a);
     uint8x16_t t = vdupq_n_u8(1);
     t = vqtbl1q_u8(t, t);
     return (int) vgetq_lane_u16(vreinterpretq_u16_p16(b), 0)
          + (int) vgetq_lane_u8(t, 0);
   ]])],
  [AC_MSG_RESULT([yes])
   AC_DEFINE([HAVE_PMULL], [1], [Define if ARM PMULL is available])
   PMULL_CFLAGS="-march=armv8-a+crypto"],
  [AC_MSG_RESULT([no])])
CFLAGS="$save_CFLAGS"

if test "x$xpar_is_x86_64" = "xyes"; then
CFLAGS="$save_CFLAGS -mavx512f -mavx512bw -mavx512vbmi"
AC_MSG_CHECKING([for AVX-512 VBMI VPERMB intrinsics])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <immintrin.h>]], [[
     __m512i a = _mm512_set1_epi8(1);
     __m512i b = _mm512_permutexvar_epi8(a, a);
     return (int) _mm512_cvtsi512_si32(b);
   ]])],
  [AC_MSG_RESULT([yes])
   AC_DEFINE([HAVE_VBMI], [1], [Define if AVX-512 VBMI is available])
   VBMI_CFLAGS="-mavx512f -mavx512bw -mavx512vbmi"],
  [AC_MSG_RESULT([no])])
CFLAGS="$save_CFLAGS"
fi

if test "x$xpar_is_x86" = "xyes"; then
CFLAGS="$save_CFLAGS -msse4.2"
AC_MSG_CHECKING([for SSE4.2 CRC32 intrinsics])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <immintrin.h>]], [[
     unsigned c = _mm_crc32_u8(0u, 1);
     return (int) c;
   ]])],
  [AC_MSG_RESULT([yes])
   AC_DEFINE([HAVE_SSE42], [1], [Define if SSE4.2 CRC32 is available])
   SSE42_CFLAGS="-msse4.2"],
  [AC_MSG_RESULT([no])])
CFLAGS="$save_CFLAGS"

CFLAGS="$save_CFLAGS -msse4.2 -mpclmul"
AC_MSG_CHECKING([for PCLMULQDQ intrinsics])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <immintrin.h>]], [[
     __m128i a = _mm_set1_epi32(1);
     __m128i b = _mm_clmulepi64_si128(a, a, 0x00);
     return _mm_cvtsi128_si32(b);
   ]])],
  [AC_MSG_RESULT([yes])
   AC_DEFINE([HAVE_PCLMUL], [1], [Define if PCLMULQDQ is available])
   SSE42_CFLAGS="-msse4.2 -mpclmul"],
  [AC_MSG_RESULT([no])])
CFLAGS="$save_CFLAGS"
fi

if test "x$xpar_is_x86_64" = "xyes"; then
CFLAGS="$save_CFLAGS -mavx2"
AC_MSG_CHECKING([for AVX2 intrinsics])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <immintrin.h>]], [[
     __m256i a = _mm256_set1_epi8(1), b = _mm256_shuffle_epi8(a, a);
     return _mm256_extract_epi32(b, 0);
   ]])],
  [AC_MSG_RESULT([yes])
   AC_DEFINE([HAVE_AVX2], [1], [Define if AVX2 is available])
   AVX2_CFLAGS="-mavx2"],
  [AC_MSG_RESULT([no])])
CFLAGS="$save_CFLAGS"

CFLAGS="$save_CFLAGS -mgfni -mavx2"
AC_MSG_CHECKING([for GFNI intrinsics with AVX2])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <immintrin.h>]], [[
     __m256i a = _mm256_set1_epi8(1);
     __m256i b = _mm256_gf2p8affine_epi64_epi8(a, a, 0);
     return _mm256_extract_epi32(b, 0);
   ]])],
  [AC_MSG_RESULT([yes])
   AC_DEFINE([HAVE_GFNI], [1], [Define if GFNI with AVX2 is available])
   GFNI_CFLAGS="-mgfni -mavx2"],
  [AC_MSG_RESULT([no])])
CFLAGS="$save_CFLAGS"

AC_MSG_CHECKING([for GFNI intrinsics with AVX-512])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <immintrin.h>]], [[
     __m512i a = _mm512_set1_epi8(1);
     __m512i b = _mm512_gf2p8affine_epi64_epi8(a, a, 0);
     return (int) _mm512_cvtsi512_si32(b);
   ]])],
  [AC_MSG_RESULT([yes])
   AC_DEFINE([HAVE_GFNI512], [1], [Define if GFNI with AVX-512 is available])
   GFNI512_CFLAGS="-mgfni -mavx512f -mavx512bw -mavx512vl"
   AVX512_CFLAGS="-mavx512f -mavx512bw -mavx512vl"],
  [AC_MSG_RESULT([no])])
CFLAGS="$save_CFLAGS"
fi

for xpar_neon_try in "" "-mfpu=neon"; do
  CFLAGS="$save_CFLAGS $xpar_neon_try"
  AC_MSG_CHECKING([for NEON intrinsics with '$xpar_neon_try'])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[#include <arm_neon.h>]], [[
       uint8x16_t a = vdupq_n_u8(1);
       uint8x16_t b = vqtbl1q_u8(a, a);
       return vgetq_lane_u8(b, 0);
     ]])],
    [AC_MSG_RESULT([yes])
     AC_DEFINE([HAVE_NEON], [1], [Define if NEON is available])
     NEON_CFLAGS="$xpar_neon_try"
     xpar_have_neon=yes],
    [AC_MSG_RESULT([no])])
  test "x$xpar_have_neon" = "xyes" && break
done
CFLAGS="$save_CFLAGS"

CFLAGS="$save_CFLAGS -march=armv8-a+crc"
AC_MSG_CHECKING([for ARMv8 CRC32 intrinsics])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <arm_acle.h>]], [[
     return (int) __crc32cb(0u, 1);
   ]])],
  [AC_MSG_RESULT([yes])
   AC_DEFINE([HAVE_ARM_CRC32], [1], [Define if ARMv8 CRC32 is available])
   ARM_CRC_CFLAGS="-march=armv8-a+crc"],
  [AC_MSG_RESULT([no])
   ARM_CRC_CFLAGS=""])
CFLAGS="$save_CFLAGS"

fi

CFLAGS="$save_CFLAGS"

AC_SUBST([SSSE3_CFLAGS])
AC_SUBST([SSE42_CFLAGS])
AC_SUBST([AVX2_CFLAGS])
AC_SUBST([AVX512_CFLAGS])
AC_SUBST([GFNI_CFLAGS])
AC_SUBST([GFNI512_CFLAGS])
AC_SUBST([VBMI_CFLAGS])
AC_SUBST([NEON_CFLAGS])
AC_SUBST([PMULL_CFLAGS])
AC_SUBST([SVE2_CFLAGS])
AC_SUBST([VSX_CFLAGS])
AC_SUBST([RVV_CFLAGS])
AC_SUBST([RVV_CLMUL_CFLAGS])
AC_SUBST([VPCLMUL_CFLAGS])
AC_SUBST([ARM_CRC_CFLAGS])

AM_CONDITIONAL([HAVE_SSSE3],   [test "x$SSSE3_CFLAGS"   != "x"])
AM_CONDITIONAL([HAVE_SSE42],   [test "x$SSE42_CFLAGS"   != "x"])
AM_CONDITIONAL([HAVE_AVX2],    [test "x$AVX2_CFLAGS"    != "x"])
AM_CONDITIONAL([HAVE_GFNI],    [test "x$GFNI_CFLAGS"    != "x"])
AM_CONDITIONAL([HAVE_GFNI512], [test "x$GFNI512_CFLAGS" != "x"])
AM_CONDITIONAL([HAVE_VBMI],    [test "x$VBMI_CFLAGS"    != "x"])
AM_CONDITIONAL([HAVE_NEON],    [test "x$xpar_have_neon" = "xyes"])
AM_CONDITIONAL([HAVE_PMULL],   [test "x$PMULL_CFLAGS"    != "x"])
AM_CONDITIONAL([HAVE_SVE2],    [test "x$SVE2_CFLAGS"     != "x"])
AM_CONDITIONAL([HAVE_VSX],     [test "x$VSX_CFLAGS"      != "x"])
AM_CONDITIONAL([HAVE_RVV],     [test "x$RVV_CFLAGS"      != "x"])
AM_CONDITIONAL([HAVE_RVV_CLMUL],
  [test "x$RVV_CLMUL_CFLAGS" != "x" && test "x$RVV_CFLAGS" != "x"])
AM_CONDITIONAL([HAVE_VPCLMUL], [test "x$VPCLMUL_CFLAGS" != "x"])
AM_CONDITIONAL([HAVE_ARM_CRC], [test "x$ARM_CRC_CFLAGS" != "x"])

AX_APPEND_COMPILE_FLAGS([ \
  -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-align -Wwrite-strings \
  -Wmissing-prototypes -Wstrict-prototypes -Wredundant-decls \
  -Wno-unused-parameter -Wvla -Wformat=2 -Wundef], [XPAR_WARN_CFLAGS])
AC_SUBST([XPAR_WARN_CFLAGS])

AX_APPEND_COMPILE_FLAGS([-fno-strict-aliasing -fno-delete-null-pointer-checks],
                        [XPAR_HARDEN_CFLAGS])
AC_SUBST([XPAR_HARDEN_CFLAGS])

AC_ARG_ENABLE([sanitizers],
  [AS_HELP_STRING([--enable-sanitizers=LIST],
    [build with address,undefined (or yes), undefined, or thread sanitizer])],
  [enable_sanitizers=$enableval], [enable_sanitizers=no])
case "$enable_sanitizers" in
  no) ;;
  yes|address|address,undefined|undefined,address)
    enable_sanitizers=address,undefined
    AX_APPEND_COMPILE_FLAGS([-fsanitize=address -fsanitize=undefined \
                             -fno-omit-frame-pointer], [XPAR_SAN_CFLAGS])
    AX_APPEND_LINK_FLAGS([-fsanitize=address -fsanitize=undefined],
                         [XPAR_SAN_LDFLAGS])
    ;;
  undefined)
    AX_APPEND_COMPILE_FLAGS([-fsanitize=undefined -fno-omit-frame-pointer],
                            [XPAR_SAN_CFLAGS])
    AX_APPEND_LINK_FLAGS([-fsanitize=undefined], [XPAR_SAN_LDFLAGS])
    ;;
  thread)
    AX_APPEND_COMPILE_FLAGS([-fsanitize=thread -fno-omit-frame-pointer],
                            [XPAR_SAN_CFLAGS])
    AX_APPEND_LINK_FLAGS([-fsanitize=thread], [XPAR_SAN_LDFLAGS])
    ;;
  *) AC_MSG_ERROR([unknown sanitizer selection: $enable_sanitizers]) ;;
esac
AC_SUBST([XPAR_SAN_CFLAGS])
AC_SUBST([XPAR_SAN_LDFLAGS])
if test "x$enable_sanitizers" != "xno"; then
  dnl  Preserve sanitizer fault handlers.
  AC_DEFINE([XPAR_SANITIZED], [1],
            [Define when a sanitizer owns the fault handlers])
fi

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

AC_MSG_NOTICE([

  xpar xpar_version configuration
  -------------------------------
    host          : $host
    Windows target: $xpar_win_target
    threads       : $xpar_have_threads
    SSSE3         : ${SSSE3_CFLAGS:-no}
    SSE4.2        : ${SSE42_CFLAGS:-no}
    AVX2          : ${AVX2_CFLAGS:-no}
    GFNI (256)    : ${GFNI_CFLAGS:-no}
    GFNI (512)    : ${GFNI512_CFLAGS:-no}
    AVX-512 VBMI  : ${VBMI_CFLAGS:-no}
    NEON          : $xpar_have_neon
    ARM PMULL     : ${PMULL_CFLAGS:-no}
    SVE2          : ${SVE2_CFLAGS:-no}
    PowerPC VSX   : ${VSX_CFLAGS:-no}
    RISC-V V      : ${RVV_CFLAGS:-no}
    RISC-V Zvbc   : ${RVV_CLMUL_CFLAGS:-no}
    VPCLMULQDQ    : ${VPCLMUL_CFLAGS:-no}
    sanitizers    : $enable_sanitizers
])
