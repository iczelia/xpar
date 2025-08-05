m4_define([xpar_version_major], [0])
m4_define([xpar_version_minor], [5])
m4_define([xpar_version], [xpar_version_major.xpar_version_minor])

AC_PREREQ([2.69])
AC_INIT([xpar],[xpar_version],[https://github.com/iczelia/xpar])
AC_CONFIG_HEADERS([config.h])
AC_COPYRIGHT([Copyright (C) 2022-2024 Kamila Szewczyk])
AC_LANG([C])
AC_CONFIG_MACRO_DIR([m4])

AC_CANONICAL_TARGET
AM_INIT_AUTOMAKE([-Wall color-tests])

AC_PROG_INSTALL
AC_PROG_MAKE_SET
AC_PROG_CC
AM_PROG_AS

AC_CHECK_HEADERS([io.h])
AC_CHECK_FUNCS([asprintf strndup stat _commit _setmode isatty fsync mmap CreateFileMappingA])
AC_DEFINE([XPAR_MINOR], [xpar_version_minor], [Minor version number of xpar])
AC_DEFINE([XPAR_MAJOR], [xpar_version_major], [Major version number of xpar])

AX_C_RESTRICT
AC_CHECK_SIZEOF([size_t])

AC_OPENMP
has_openmp=false
if test -n "$OPENMP_CFLAGS"; then
  has_openmp=true
fi
if test -n "$OPENMP_LDFLAGS"; then
  has_openmp=true
fi
if test "$has_openmp" = "true"; then
  AC_DEFINE([XPAR_OPENMP], [1], [Enable OpenMP parallelisation.])
  AM_CONDITIONAL([XPAR_OPENMP], [true])
  AX_APPEND_COMPILE_FLAGS([$OPENMP_CFLAGS])
  AX_APPEND_LINK_FLAGS([$OPENMP_LDFLAGS])
else
  AM_CONDITIONAL([XPAR_OPENMP], [false])
fi

AC_ARG_ENABLE([x86_64], [AS_HELP_STRING([--enable-x86_64], [Enable x86_64 platform specific code.])], [enable_x86_64=$enableval], [enable_x86_64=no])
if test "x$enable_x86_64" = "xyes"; then
  AC_DEFINE([XPAR_X86_64], [1], [Enable x86_64 platform specific code.])
  AM_CONDITIONAL([XPAR_X86_64], [true])
  AX_PROG_NASM
  AX_GCC_FUNC_ATTRIBUTE([sysv_abi])
else
  AM_CONDITIONAL([XPAR_X86_64], [false])
fi

AC_ARG_ENABLE([aarch64], [AS_HELP_STRING([--enable-aarch64], [Enable aarch64 platform specific code.])], [enable_aarch64=$enableval], [enable_aarch64=no])
if test "x$enable_aarch64" = "xyes"; then
  AC_DEFINE([XPAR_AARCH64], [1], [Enable aarch64 platform specific code.])
  AM_CONDITIONAL([XPAR_AARCH64], [true])
else
  AM_CONDITIONAL([XPAR_AARCH64], [false])
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

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
