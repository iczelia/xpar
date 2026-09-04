/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; version 3 of the License only.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.  */

/*  Host I/O. Portable code has no system headers or host conditionals.  */

#ifndef XPAR_PORT_H
#define XPAR_PORT_H

#include "config.h"
#include <stdarg.h>
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*  Portable absence error values.  */
#ifdef ENOENT
#define XPAR_ENOENT ENOENT
#else
#define XPAR_ENOENT 2
#endif
#ifdef ENOTDIR
#define XPAR_ENOTDIR ENOTDIR
#else
#define XPAR_ENOTDIR XPAR_ENOENT
#endif

/*  Types.  */

typedef uint8_t  u8;   typedef int8_t  i8;
typedef uint16_t u16;  typedef int16_t i16;
typedef uint32_t u32;  typedef int32_t i32;
typedef uint64_t u64;  typedef int64_t i64;
typedef size_t   sz;   typedef double  f64;

#ifndef INT64_MIN
#define INT64_MIN (-INT64_MAX - 1)
#endif

/*  Validate formats against xpar's C99 formatter, not Windows msvcrt.  */
#if defined(__GNUC__)
#if defined(_WIN32) || defined(__CYGWIN__)
#define XPAR_PRINTF(fmt, args) __attribute__((format(gnu_printf, fmt, args)))
#else
#define XPAR_PRINTF(fmt, args) __attribute__((format(printf, fmt, args)))
#endif
#define XPAR_NORETURN          __attribute__((noreturn))
#else
#define XPAR_PRINTF(fmt, args)
#define XPAR_NORETURN
#endif

/*  Install the host crash reporter.  */
void xpar_crash_install(void);

/*  Allocation-free helpers for host crash reporters.  */
int  xpar_crash_wanted(void);
int  xpar_crash_entered(void);
void xpar_crash_head(const char * what, u64 code, int have_code,
                     const void * pc, const void * addr, int have_addr,
                     const void * module_base);
void xpar_crash_frame(unsigned i, const void * pc, const void * module_base);
void xpar_crash_tail(int had_frames);
unsigned xpar_crash_walk_fp(void * const * fp, void ** out, unsigned max);

/*  Process.  */

void xpar_host_init(void);
int  xpar_main(int argc, char ** argv);
void xpar_exit(int code) XPAR_NORETURN;
const char * xpar_getenv(const char * name);

/*  The host's temporary directory, used when the environment names none.  */
const char * xpar_tmpdir(void);

typedef struct xpar_file xpar_file;

extern xpar_file * const xpar_stdin;
extern xpar_file * const xpar_stdout;
extern xpar_file * const xpar_stderr;

/*  Access mode is a value, not a flag. Mask with XPAR_O_ACCMODE and compare;
    XPAR_O_RDONLY is zero. Higher flags are ordinary bits.  */
#define XPAR_O_ACCMODE 3
#define XPAR_O_RDONLY 0
#define XPAR_O_WRONLY 1
#define XPAR_O_RDWR   2
#define XPAR_O_CREAT  4
#define XPAR_O_TRUNC  8
#define XPAR_O_EXCL   16
#define XPAR_O_APPEND 32
#define XPAR_O_NOFOLLOW 64
/*  Create with owner-only permissions where the host has file modes.  */
#define XPAR_O_PRIVATE 128

#define XPAR_SEEK_SET 0
#define XPAR_SEEK_CUR 1
#define XPAR_SEEK_END 2

xpar_file * xpar_open (const char * path, int flags);
int         xpar_close(xpar_file *);
sz          xpar_read (xpar_file *, void * buf, sz n);
sz          xpar_write(xpar_file *, const void * buf, sz n);
int         xpar_seek (xpar_file *, i64 off, int whence);
i64         xpar_tell (xpar_file *);
int         xpar_flush(xpar_file *);
int         xpar_fsync(xpar_file *);
i64         xpar_size (xpar_file *);
bool        xpar_is_seekable(xpar_file *);
bool        xpar_is_tty(xpar_file *);
bool        xpar_eof  (xpar_file *);

/*  Sticky ferror-style state.  */
int         xpar_error(xpar_file *);

/*  Advisory whole-file locks.  */
int  xpar_lock       (xpar_file *, bool exclusive);
int  xpar_unlock     (xpar_file *);
bool xpar_lock_supported(void);

/*  Positional I/O, which is what in-place repair writes through.  */
sz  xpar_pread (xpar_file *, void * buf, sz n, u64 off);
sz  xpar_pwrite(xpar_file *, const void * buf, sz n, u64 off);
int xpar_ftruncate(xpar_file *, u64 length);

/*  Ordered positional reads; a NULL file returns zero without I/O.  */
typedef struct {
  xpar_file * file;
  void *      buf;
  sz          length;
  u64         offset;
  sz          result;
} xpar_read_req;

bool xpar_pread_batch(xpar_read_req *, sz count);

/*  Serial fallback; always returns false.  */
bool xpar_pread_serial(xpar_read_req *, sz count);

/*  Sync the directory containing `path`; unsupported hosts return 0.  */
int xpar_fsync_dir(const char * path);

/*  Fatal-on-failure I/O.  */
sz   xpar_xread (xpar_file *, void * p, sz n);
void xpar_xwrite(xpar_file *, const void * p, sz n);
typedef struct { const void * data;  sz length; } xpar_write_part;
void xpar_xwritev(xpar_file *, const xpar_write_part *, u32 count);
void xpar_xclose(xpar_file *);

/*  Memory mapping.  */

typedef struct { u8 * map; sz size; bool valid; } xpar_mmap;

xpar_mmap xpar_map  (const char * path);
void      xpar_unmap(xpar_mmap *);

/*  Advice for a sequential streaming pass. Best effort; never fails.  */
void xpar_advise_sequential(xpar_file *, u64 off, u64 len);
void xpar_advise_random(xpar_file *, u64 off, u64 len);

/*  Memory.  */

void * xpar_malloc   (sz n);          /*  Fatal on failure.  */
void * xpar_calloc   (sz n, sz size); /*  Fatal on failure, zeroed.  */
void * xpar_alloc_raw(sz n);          /*  Fatal on failure, uninitialised.  */
void * xpar_realloc  (void * p, sz n);
void   xpar_free     (void * p);

/*  Aligned allocation for SIMD kernels. `align` must be a power of two.
    Freed with xpar_free_aligned, never with xpar_free.  */
void * xpar_alloc_aligned(sz n, sz align);
void   xpar_free_aligned (void * p);

/*  Physical memory in bytes, or 0.  */
u64 xpar_physical_memory(void);

/*  Whether `path` is on rotating media; unknown is false.  */
bool xpar_is_rotational(const char * path);

/*  Strings and formatting.  */

void * xpar_memcpy (void * d, const void * s, sz n);
void * xpar_memmove(void * d, const void * s, sz n);
void * xpar_memset (void * d, int c, sz n);
int    xpar_memcmp (const void * a, const void * b, sz n);
sz     xpar_strlen (const char * s);
int    xpar_strcmp (const char * a, const char * b);
int    xpar_strncmp(const char * a, const char * b, sz n);
char * xpar_strdup (const char * s);
char * xpar_strndup(const char * s, sz n);

int xpar_parse_u64(const char * s, u64 * out);

/*  Backend text sink; Win32 converts console output to UTF-16.  */
void xpar_port_write_text(xpar_file *, const char * s, sz n);

int xpar_vsnprintf(char * buf, sz cap, const char * fmt, va_list ap);
int xpar_snprintf (char * buf, sz cap, const char * fmt, ...)
                   XPAR_PRINTF(3, 4);
int xpar_asprintf (char ** out, const char * fmt, ...) XPAR_PRINTF(2, 3);
int xpar_fprintf  (xpar_file *, const char * fmt, ...) XPAR_PRINTF(2, 3);
int xpar_vfprintf (xpar_file *, const char * fmt, va_list ap);
int xpar_fputs    (const char * s, xpar_file *);

/*  Errors and time.  */

const char * xpar_strerror(int err);
int          xpar_errno(void);
u64          xpar_usec_now(void);

/*  True only when a path is absent.  */
static inline bool xpar_errno_absent(int err) {
#if defined(_WIN32)
  /*  ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND, ERROR_INVALID_NAME.  */
  if (err == 2 || err == 3 || err == 123) return true;
#endif
  return err == XPAR_ENOENT || err == XPAR_ENOTDIR;
}

/*  Unix time in nanoseconds; xpar_usec_now is monotonic.  */
i64 xpar_wall_ns(void);

/*  Random bytes for staging names; not for cryptographic keys.  */
void xpar_random_bytes(void * buf, sz n);

#endif
