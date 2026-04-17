/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.  */

/*  Host abstraction: libc-like API wired to POSIX libc or Win32 directly.  */

#ifndef _PLATFORM_H_
#define _PLATFORM_H_

#include "config.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdarg.h>

#define MiB(x) ((x) * 1024 * 1024)
#define KiB(x) ((x) * 1024)

/*  Integer aliases used across the tree.  */
typedef uint8_t  u8;   typedef int8_t  i8;
typedef uint16_t u16;  typedef int16_t i16;
typedef uint32_t u32;  typedef int32_t i32;
typedef uint64_t u64;  typedef int64_t i64;
typedef size_t   sz;

/*  -----------------------------------------------------------------------
  File handle (opaque)  */
typedef struct xpar_file xpar_file;

extern xpar_file * const xpar_stdin;
extern xpar_file * const xpar_stdout;
extern xpar_file * const xpar_stderr;

/*  Open flags. Bitmask; pass one READ or WRITE, combine with optional
    CREATE/TRUNCATE/EXCLUSIVE/APPEND.  */
enum {
  XPAR_O_READ      = 1 << 0,
  XPAR_O_WRITE     = 1 << 1,
  XPAR_O_CREATE    = 1 << 2,
  XPAR_O_TRUNCATE  = 1 << 3,
  XPAR_O_EXCLUSIVE = 1 << 4,
  XPAR_O_APPEND    = 1 << 5
};

/*  Whence constants for xpar_seek.  */
enum { XPAR_SEEK_SET = 0, XPAR_SEEK_CUR = 1, XPAR_SEEK_END = 2 };

/*  -----------------------------------------------------------------------
  Lifecycle  */
void xpar_host_init(void);
int  xpar_main(int argc, char ** argv);

/*  -----------------------------------------------------------------------
  File I/O  */
xpar_file * xpar_open  (const char * path, int flags);
int         xpar_close (xpar_file *);
sz          xpar_read  (xpar_file *, void * buf, sz n);
sz          xpar_write (xpar_file *, const void * buf, sz n);
int         xpar_seek  (xpar_file *, i64 off, int whence);
i64         xpar_tell  (xpar_file *);
int         xpar_flush (xpar_file *);
int         xpar_fsync (xpar_file *);
i64         xpar_size  (xpar_file *);
bool        xpar_is_seekable(xpar_file *);
bool        xpar_is_tty(xpar_file *);
bool        xpar_eof   (xpar_file *);
int         xpar_error (xpar_file *);

/*  "Safe" helpers: abort the process on error. Match today's xfread /
    xfwrite / xfclose semantics.  */
sz   xpar_xread (xpar_file *, void * p, sz n);
void xpar_xwrite(xpar_file *, const void * p, sz n);
void xpar_xclose(xpar_file *);
void xpar_notty (xpar_file *);

/*  -----------------------------------------------------------------------
  Filesystem queries / ops (without opening)  */
typedef struct { u64 size; bool is_dir; bool is_regular; } xpar_stat_t;
int xpar_stat_path(const char * path, xpar_stat_t * out);
int xpar_remove   (const char * path);

/*  1 if both paths resolve to the same file, 0 if not, -1 on error
    (e.g. either path does not exist).  */
int xpar_same_file(const char * a, const char * b);

/*  -----------------------------------------------------------------------
  Memory map  */
typedef struct { u8 * map; sz size; } xpar_mmap;
xpar_mmap xpar_map  (const char * path);
void      xpar_unmap(xpar_mmap *);
#define XPAR_ALLOW_MAPPING 1

/*  Memory allocation. xpar_malloc zeroes; xpar_alloc_raw does not.
    Both FATAL on OOM.  */
void * xpar_malloc   (sz n);
void * xpar_alloc_raw(sz n);
void * xpar_realloc  (void * p, sz n);
void   xpar_free     (void * p);

/*  Strings. mem... and str... route through compiler builtins when
    available; fall back to out-of-line host backend symbols.  */
#if defined(__has_builtin)
  #if __has_builtin(__builtin_memcpy)
    #define xpar_memcpy(d, s, n)  __builtin_memcpy((d), (s), (n))
  #endif
  #if __has_builtin(__builtin_memmove)
    #define xpar_memmove(d, s, n) __builtin_memmove((d), (s), (n))
  #endif
  #if __has_builtin(__builtin_memset)
    #define xpar_memset(d, c, n)  __builtin_memset((d), (c), (n))
  #endif
  #if __has_builtin(__builtin_memcmp)
    #define xpar_memcmp(a, b, n)  __builtin_memcmp((a), (b), (n))
  #endif
  #if __has_builtin(__builtin_strlen)
    #define xpar_strlen(s)        __builtin_strlen((s))
  #endif
  #if __has_builtin(__builtin_strcmp)
    #define xpar_strcmp(a, b)     __builtin_strcmp((a), (b))
  #endif
  #if __has_builtin(__builtin_strncmp)
    #define xpar_strncmp(a, b, n) __builtin_strncmp((a), (b), (n))
  #endif
#endif

#ifndef xpar_memcpy
  void * xpar_memcpy(void * d, const void * s, sz n);
#endif
#ifndef xpar_memmove
  void * xpar_memmove(void * d, const void * s, sz n);
#endif
#ifndef xpar_memset
  void * xpar_memset(void * d, int c, sz n);
#endif
#ifndef xpar_memcmp
  int    xpar_memcmp(const void * a, const void * b, sz n);
#endif
#ifndef xpar_strlen
  sz     xpar_strlen(const char * s);
#endif
#ifndef xpar_strcmp
  int    xpar_strcmp(const char * a, const char * b);
#endif
#ifndef xpar_strncmp
  int    xpar_strncmp(const char * a, const char * b, sz n);
#endif

char * xpar_strdup (const char * s);
char * xpar_strndup(const char * s, sz n);

/*  Numeric parsing: 0 on success, nonzero on malformed input.
    Optional leading '-' for signed; no hex/octal.  */
int xpar_parse_i64(const char * s, i64 * out);
int xpar_parse_u64(const char * s, u64 * out);

/*  Subset printf: %d %i %u %ld %lu %lld %llu %zd %zu %x %X %c %s %f %p %%,
    flags -,0,+,space, width/precision (decimal or *).  */
int xpar_vsnprintf(char * buf, sz cap, const char * fmt, va_list ap);
int xpar_snprintf (char * buf, sz cap, const char * fmt, ...)
    __attribute__((format(printf, 3, 4)));
int xpar_asprintf (char ** out, const char * fmt, ...)
    __attribute__((format(printf, 2, 3)));
int xpar_fprintf  (xpar_file *, const char * fmt, ...)
    __attribute__((format(printf, 2, 3)));
int xpar_vfprintf (xpar_file *, const char * fmt, va_list ap);
int xpar_fputs    (const char * s, xpar_file *);

/*  -----------------------------------------------------------------------
  Process / errors / time  */
__attribute__((noreturn)) void xpar_exit(int code);
const char * xpar_strerror(int err);
int          xpar_errno(void);
u64          xpar_usec_now(void);

/*  -----------------------------------------------------------------------
  Threading primitives (host-provided)  */
typedef struct xpar_mutex  xpar_mutex;
typedef struct xpar_cond   xpar_cond;
typedef struct xpar_thread xpar_thread;

xpar_mutex  * xpar_mutex_new   (void);
void          xpar_mutex_free  (xpar_mutex *);
void          xpar_mutex_lock  (xpar_mutex *);
void          xpar_mutex_unlock(xpar_mutex *);

xpar_cond   * xpar_cond_new      (void);
void          xpar_cond_free     (xpar_cond *);
void          xpar_cond_wait     (xpar_cond *, xpar_mutex *);
void          xpar_cond_signal   (xpar_cond *);
void          xpar_cond_broadcast(xpar_cond *);

xpar_thread * xpar_thread_start(void (*fn)(void *), void * ctx);
void          xpar_thread_join (xpar_thread *);

int           xpar_cpu_count(void);

/*  Parallel for: static partition [k*n/T, (k+1)*n/T). Lazy-init;
    serial fallback when T==1 or n<=1.  */
typedef void (*xpar_parfor_fn)(sz idx, void * ctx);
void xpar_parallel_for(sz n, xpar_parfor_fn fn, void * ctx);
void xpar_set_num_threads(int n);

/*  Atomics: C11 <stdatomic.h> when available, else gcc/clang builtins.
    Relaxed-only; counters only, barrier provided by parallel-for join.  */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L \
    && !defined(__STDC_NO_ATOMICS__)
  #include <stdatomic.h>
  typedef _Atomic int xpar_atomic_int;
  #define xpar_atomic_add_int(p, v) \
    atomic_fetch_add_explicit((p), (v), memory_order_relaxed)
  #define xpar_atomic_load_int(p) \
    atomic_load_explicit((p), memory_order_relaxed)
  #define xpar_atomic_store_int(p, v) \
    atomic_store_explicit((p), (v), memory_order_relaxed)
#elif defined(__GNUC__) || defined(__clang__)
  typedef int xpar_atomic_int;
  #define xpar_atomic_add_int(p, v) \
    __atomic_fetch_add((p), (v), __ATOMIC_RELAXED)
  #define xpar_atomic_load_int(p) \
    __atomic_load_n((p), __ATOMIC_RELAXED)
  #define xpar_atomic_store_int(p, v) \
    __atomic_store_n((p), (v), __ATOMIC_RELAXED)
#else
  #error "Need C11 atomics or GCC/Clang __atomic_* builtins."
#endif

#endif
