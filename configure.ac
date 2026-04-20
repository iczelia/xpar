m4_define([xpar_version_major], [1])
m4_define([xpar_version_minor], [1])
m4_define([xpar_version], [xpar_version_major.xpar_version_minor])

AC_PREREQ([2.69])
AC_INIT([xpar],[xpar_version],[https://github.com/iczelia/xpar])
AC_CONFIG_HEADERS([config.h])
AC_COPYRIGHT([Copyright (C) 2022-2026 Kamila Szewczyk])
AC_LANG([C])
AC_CONFIG_MACRO_DIR([m4])

AC_CANONICAL_TARGET
AM_INIT_AUTOMAKE([-Wall color-tests])

AC_PROG_INSTALL
AC_PROG_MAKE_SET
AC_PROG_CC
# AM_PROG_AR's interface probe returns "unknown" on some DJGPP cross-toolchains.
case $host_os in
  msdosdjgpp*|djgpp*) am_cv_ar_interface=ar ;;
esac
AM_PROG_AR
AC_PROG_RANLIB
AM_PROG_AS

# Probe for native Windows target (host_os is unreliable: MSYS2's msys gcc
# reports as mingw-ish but actually targets the Cygwin-style MSYS runtime).
AC_MSG_CHECKING([whether $CC targets native Windows])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[
    #if !defined(_WIN32) && !defined(__MINGW32__) && !defined(__MINGW64__)
    #error not windows
    #endif
  ]], [[]])],
  [AC_MSG_RESULT([yes]); xpar_win_host=yes],
  [AC_MSG_RESULT([no]);  xpar_win_host=no])
AM_CONDITIONAL([XPAR_WIN_HOST], [test "x$xpar_win_host" = "xyes"])

# MS-DOS / DJGPP host detection (triplet is i586-pc-msdosdjgpp).
case $host_os in
  msdosdjgpp*|djgpp*) xpar_dos_host=yes ;;
  *)                  xpar_dos_host=no  ;;
esac
if test "x$xpar_dos_host" = "xyes"; then
  AC_DEFINE([XPAR_DOS], [1], [MS-DOS / DJGPP target])
  # Self-contained DOS .exe: exe2coff strips the go32 stub, CWSDSTUB.EXE (go32 + CWSDPMI r7) is prepended so no external CWSDPMI.EXE is needed.
  AC_CHECK_PROGS([EXE2COFF], [${host_alias}-exe2coff ${host}-exe2coff exe2coff])
  if test -z "$EXE2COFF"; then
    AC_MSG_ERROR([exe2coff not found -- ships with the DJGPP binutils under <prefix>/i586-pc-msdosdjgpp/bin])
  fi
  AC_ARG_VAR([CWSDSTUB], [Path to CWSDSTUB.EXE (from csdpmi7b.zip on delorie.com)])
  if test -z "$CWSDSTUB"; then
    AC_MSG_CHECKING([for CWSDSTUB.EXE])
    for d in /usr/share/csdpmi /usr/local/share/csdpmi \
             "$HOME/csdpmi/bin" "$HOME/djgpp/bin" \
             "`dirname $EXE2COFF`"; do
      for f in CWSDSTUB.EXE cwsdstub.exe; do
        if test -f "$d/$f"; then CWSDSTUB="$d/$f"; break 2; fi
      done
    done
    if test -z "$CWSDSTUB"; then
      AC_MSG_RESULT([not found])
      AC_MSG_ERROR([CWSDSTUB.EXE not found -- fetch csdpmi7b.zip from delorie.com and pass its path via CWSDSTUB=...])
    fi
    AC_MSG_RESULT([$CWSDSTUB])
  fi
fi
AM_CONDITIONAL([XPAR_DOS_HOST], [test "x$xpar_dos_host" = "xyes"])

# Windows target selection: `vista` (default, SRWLOCK/wide API) or `win95`
# (legacy host-win32.c paths: narrow API, CRITICAL_SECTION, 32-bit offsets).
AC_ARG_WITH([windows-target],
  [AS_HELP_STRING([--with-windows-target=@<:@vista|win95@:>@],
    [target Windows version for native-Windows builds (default: vista)])],
  [xpar_win_target=$withval],
  [xpar_win_target=vista])
case $xpar_win_target in
  vista)
    xpar_win_cppflags="-D_WIN32_WINNT=0x0600 -DWINVER=0x0600"
    xpar_win_ldflags="-Wl,--major-subsystem-version=6,--minor-subsystem-version=0,--major-os-version=6,--minor-os-version=0"
    ;;
  win95)
    xpar_win_cppflags="-D_WIN32_WINNT=0x0400 -DWINVER=0x0400 -DXPAR_WIN_LEGACY=1"
    xpar_win_ldflags="-Wl,--major-subsystem-version=4,--minor-subsystem-version=0,--major-os-version=4,--minor-os-version=0"
    ;;
  *)
    AC_MSG_ERROR([--with-windows-target must be "vista" or "win95" (got: $xpar_win_target)])
    ;;
esac
XPAR_WIN_CPPFLAGS=""
XPAR_WIN_SUBSYS_LDFLAGS=""
if test "x$xpar_win_host" = "xyes"; then
  XPAR_WIN_CPPFLAGS="$xpar_win_cppflags"
  XPAR_WIN_SUBSYS_LDFLAGS="$xpar_win_ldflags"
  AC_MSG_NOTICE([Windows target: $xpar_win_target ($XPAR_WIN_CPPFLAGS)])
fi
AC_SUBST([XPAR_WIN_CPPFLAGS])
AC_SUBST([XPAR_WIN_SUBSYS_LDFLAGS])

# host-win32.c's byte-loop memcpy/memset mustn't be rewritten into calls
# to themselves -- probe for flags that disable gcc's pattern recognition.
if test "x$xpar_win_host" = "xyes"; then
  HOST_WIN32_CFLAGS=""
  save_CFLAGS="$CFLAGS"
  for flag in -fno-builtin -fno-tree-loop-distribute-patterns; do
    CFLAGS="$save_CFLAGS $HOST_WIN32_CFLAGS $flag"
    AC_MSG_CHECKING([whether $CC accepts $flag])
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]],[[]])],
      [AC_MSG_RESULT([yes]); HOST_WIN32_CFLAGS="$HOST_WIN32_CFLAGS $flag"],
      [AC_MSG_RESULT([no])])
  done
  CFLAGS="$save_CFLAGS"
fi
AC_SUBST([HOST_WIN32_CFLAGS])

AC_DEFINE([XPAR_MINOR], [xpar_version_minor], [Minor version number of xpar])
AC_DEFINE([XPAR_MAJOR], [xpar_version_major], [Major version number of xpar])

AC_C_RESTRICT
AC_CHECK_SIZEOF([size_t])

# POSIX host wraps pthreads -> need -pthread; mingw/DJGPP don't.
case $host_os in
  mingw*|msdosdjgpp*|djgpp*) ;;
  *) AX_APPEND_COMPILE_FLAGS([-pthread])
     AX_APPEND_LINK_FLAGS([-pthread]) ;;
esac

AC_ARG_ENABLE([x86_64], [AS_HELP_STRING([--enable-x86_64], [Enable x86_64 platform specific code.])], [enable_x86_64=$enableval], [enable_x86_64=no])
# DJGPP targets i386 protected mode: no SSE/AVX on the compiler, no 64-bit
# ABI. Force x86_64 off regardless of what the user passed.
if test "x$xpar_dos_host" = "xyes"; then
  if test "x$enable_x86_64" = "xyes"; then
    AC_MSG_NOTICE([x86_64 SIMD disabled on DJGPP target (forced)])
  fi
  enable_x86_64=no
fi
if test "x$enable_x86_64" = "xyes"; then
  AC_DEFINE([XPAR_X86_64], [1], [Enable x86_64 platform specific code.])
  AM_CONDITIONAL([XPAR_X86_64], [true])
  AX_PROG_NASM
  AX_GCC_FUNC_ATTRIBUTE([sysv_abi])

  # Save user flags
  save_CFLAGS="$CFLAGS"

  # ---- SSSE3 ----
  CFLAGS="$save_CFLAGS -mssse3"
  AC_MSG_CHECKING([for SSSE3 intrinsics with $CC])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[
      #include <tmmintrin.h>
    ]], [[
      __m128i a = _mm_set1_epi8(1);
      __m128i b = _mm_shuffle_epi8(a,a);
      return _mm_extract_epi16(b,0);
    ]])],
    [AC_MSG_RESULT([yes])
    AC_DEFINE([HAVE_SSSE3], [1], [Define if SSSE3 intrinsics are available])
    SSSE3_CFLAGS="-mssse3"],
    [AC_MSG_RESULT([no])
    SSSE3_CFLAGS=""]
  )

  # ---- SSE4.1 ----
  CFLAGS="$save_CFLAGS -msse4.1"
  AC_MSG_CHECKING([for SSE4.1 intrinsics with $CC])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[
      #include <smmintrin.h>
    ]], [[
      __m128i a = _mm_set1_epi32(1);
      __m128i b = _mm_insert_epi64(a, 0, 1);
      return _mm_extract_epi32(b, 0);
    ]])],
    [AC_MSG_RESULT([yes])
    AC_DEFINE([HAVE_SSE41], [1], [Define if SSE4.1 intrinsics are available])
    SSE41_CFLAGS="-msse4.1"],
    [AC_MSG_RESULT([no])
    SSE41_CFLAGS=""]
  )

  # ---- AVX2 ----
  CFLAGS="$save_CFLAGS -mavx2"
  AC_MSG_CHECKING([for AVX2 intrinsics with $CC])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[
      #include <immintrin.h>
    ]], [[
      __m256i a = _mm256_set1_epi32(1);
      __m256i b = _mm256_add_epi32(a,a);
      return _mm256_extract_epi32(b,0);
    ]])],
    [AC_MSG_RESULT([yes])
    AC_DEFINE([HAVE_AVX2], [1], [Define if AVX2 intrinsics are available])
    AVX2_CFLAGS="-mavx2"],
    [AC_MSG_RESULT([no])
    AVX2_CFLAGS=""]
  )

  CFLAGS="$save_CFLAGS"
else
  AM_CONDITIONAL([XPAR_X86_64], [false])
fi
AC_SUBST([SSSE3_CFLAGS])
AC_SUBST([SSE41_CFLAGS])
AC_SUBST([AVX2_CFLAGS])
AM_CONDITIONAL([HAVE_SSSE3], [test "x$SSSE3_CFLAGS" != "x"])
AM_CONDITIONAL([HAVE_SSE41], [test "x$SSE41_CFLAGS" != "x"])
AM_CONDITIONAL([HAVE_AVX2], [test "x$AVX2_CFLAGS" != "x"])

AC_ARG_ENABLE([aarch64], [AS_HELP_STRING([--enable-aarch64], [Enable aarch64 platform specific code.])], [enable_aarch64=$enableval], [enable_aarch64=no])
if test "x$enable_aarch64" = "xyes"; then
  AC_DEFINE([XPAR_AARCH64], [1], [Enable aarch64 platform specific code.])
  AM_CONDITIONAL([XPAR_AARCH64], [true])

  # Save user flags
  save_CFLAGS="$CFLAGS"

  # ---- NEON (mandatory on AArch64) ----
  CFLAGS="$save_CFLAGS -march=armv8-a"
  AC_MSG_CHECKING([for NEON intrinsics with $CC])
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[
      #include <arm_neon.h>
    ]], [[
      uint8x16_t a = vdupq_n_u8(1);
      uint8x16_t b = vrev64q_u8(a);
      return vgetq_lane_u8(b,0);
    ]])],
    [AC_MSG_RESULT([yes])
     AC_DEFINE([HAVE_NEON], [1], [Define if NEON intrinsics are available])
     NEON_CFLAGS="-march=armv8-a"],
    [AC_MSG_RESULT([no])
     NEON_CFLAGS=""]
  )

  CFLAGS="$save_CFLAGS $NEON_CFLAGS"
else
  AM_CONDITIONAL([XPAR_AARCH64], [false])
fi

AC_ARG_ENABLE([liburing],
  [AS_HELP_STRING([--enable-liburing@<:@=auto|yes|no@:>@],
    [Batched I/O via Linux io_uring (default: auto on Linux).])],
  [enable_liburing=$enableval], [enable_liburing=auto])
case $host_os in
  linux*) : ;;
  *) if test "x$enable_liburing" = "xyes"; then
       AC_MSG_ERROR([--enable-liburing=yes is Linux-only])
     fi
     enable_liburing=no ;;
esac
xpar_uring=no
if test "x$enable_liburing" != "xno"; then
  AC_CHECK_HEADERS([liburing.h], [xpar_has_uring_h=yes], [xpar_has_uring_h=no])
  if test "x$xpar_has_uring_h" = "xyes"; then
    AC_CHECK_LIB([uring], [io_uring_queue_init_params],
      [AC_DEFINE([XPAR_HAS_LIBURING], [1],
         [Define if io_uring batched I/O is compiled in.])
       LIBS="-luring $LIBS"
       xpar_uring=yes],
      [xpar_uring=no])
  fi
  if test "x$enable_liburing" = "xyes" -a "x$xpar_uring" != "xyes"; then
    AC_MSG_ERROR([--enable-liburing=yes requested but liburing not usable])
  fi
fi
AM_CONDITIONAL([XPAR_HAS_LIBURING], [test "x$xpar_uring" = "xyes"])

AC_ARG_ENABLE([blake2b], [AS_HELP_STRING([--disable-blake2b], [Disable BLAKE2b integrity and keyed-MAC support.])], [enable_blake2b=$enableval], [enable_blake2b=yes])
if test "x$enable_blake2b" = "xyes"; then
  AC_DEFINE([HAVE_BLAKE2B], [1], [Define if BLAKE2b integrity support is compiled in.])
  AM_CONDITIONAL([HAVE_BLAKE2B], [true])
else
  AM_CONDITIONAL([HAVE_BLAKE2B], [false])
fi

AC_ARG_ENABLE([native], [AS_HELP_STRING([--enable-native], [Enable native platform optimisations.])], [enable_native=$enableval], [enable_native=no])
if test "x$enable_native" = "xyes"; then
  AX_APPEND_COMPILE_FLAGS([-march=native -mtune=native])
fi

AC_ARG_ENABLE([static], [AS_HELP_STRING([--enable-static], [Enable static linking.])], [enable_static=$enableval], [enable_static=no])
if test "x$enable_static" = "xyes"; then
  AX_APPEND_LINK_FLAGS([-static])
fi

AC_ARG_ENABLE([lto], [AS_HELP_STRING([--enable-lto], [Enable link-time optimisation.])], [enable_lto=$enableval], [enable_lto=no])
if test "x$enable_lto" = "xyes"; then
  AX_APPEND_COMPILE_FLAGS([-flto])
  AX_APPEND_LINK_FLAGS([-flto])
fi

# Size-reduction flags. -fdata-sections omitted: on COFF/DJGPP it
# materialises .bss.* into file-backed .data and grows the binary.
AX_APPEND_COMPILE_FLAGS([-ffunction-sections -fmerge-all-constants -fno-ident])
AX_APPEND_LINK_FLAGS([-Wl,--gc-sections])

AC_CHECK_PROGS([SIZE], [${host_alias}-size ${host}-size size], [:])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
