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

/*  DJGPP/MS-DOS host. Like host-posix.c but no mmap, no pthreads,
    32-bit off_t (<2GiB), and LFN via _use_lfn(".").  */

#if !defined(__DJGPP__)
#error "host-djgpp.c compiled on a non-DJGPP target"
#endif

#include "common.h"

#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/*  -----------------------------------------------------------------------
  xpar_file wrapping FILE *  */

struct xpar_file { FILE * fp; bool owned; };

static struct xpar_file g_stdin  = { NULL, false };
static struct xpar_file g_stdout = { NULL, false };
static struct xpar_file g_stderr = { NULL, false };

xpar_file * const xpar_stdin  = &g_stdin;
xpar_file * const xpar_stdout = &g_stdout;
xpar_file * const xpar_stderr = &g_stderr;

void xpar_host_init(void) {
  g_stdin.fp  = stdin;
  g_stdout.fp = stdout;
  g_stderr.fp = stderr;
  /*  Binary mode for non-tty std streams to pass raw bytes through.  */
  if (!isatty(fileno(stdin)))  setmode(fileno(stdin),  O_BINARY);
  if (!isatty(fileno(stdout))) setmode(fileno(stdout), O_BINARY);
  /*  Probe LFN support; DJGPP caches the answer.  */
  _use_lfn(".");
}

/*  -----------------------------------------------------------------------
  Open / close  */

xpar_file * xpar_open(const char * path, int flags) {
  const char * mode;
  if ((flags & XPAR_O_READ) && (flags & XPAR_O_WRITE)) {
    mode = (flags & XPAR_O_TRUNCATE) ? "w+b" : "r+b";
  } else if (flags & XPAR_O_WRITE) {
    mode = (flags & XPAR_O_APPEND) ? "ab" : "wb";
  } else {
    mode = "rb";
  }
  FILE * fp = fopen(path, mode);
  if (!fp) return NULL;
  struct xpar_file * f = malloc(sizeof(*f));
  if (!f) { fclose(fp); errno = ENOMEM; return NULL; }
  f->fp = fp;
  f->owned = true;
  return f;
}

int xpar_close(xpar_file * f) {
  if (!f || !f->owned) return 0;
  int r = fclose(f->fp) == EOF ? -1 : 0;
  free(f);
  return r;
}

/*  -----------------------------------------------------------------------
  Read / write / seek / tell  */

sz xpar_read(xpar_file * f, void * buf, sz n) {
  return fread(buf, 1, n, f->fp);
}
sz xpar_write(xpar_file * f, const void * buf, sz n) {
  return fwrite(buf, 1, n, f->fp);
}
int xpar_seek(xpar_file * f, i64 off, int whence) {
  int w = whence == XPAR_SEEK_SET ? SEEK_SET
        : whence == XPAR_SEEK_CUR ? SEEK_CUR : SEEK_END;
  /*  DJGPP off_t is 32-bit; reject seeks beyond 2 GiB.  */
  if (off > (i64) INT32_MAX || off < -(i64) INT32_MAX) {
    errno = EOVERFLOW; return -1;
  }
  return fseek(f->fp, (long) off, w);
}
i64 xpar_tell(xpar_file * f) { return (i64) ftell(f->fp); }
int xpar_flush(xpar_file * f) { return fflush(f->fp); }
int xpar_fsync(xpar_file * f) { return fflush(f->fp); }
i64 xpar_size(xpar_file * f) {
  struct stat st;
  if (fstat(fileno(f->fp), &st) != 0) return -1;
  return (i64) st.st_size;
}
bool xpar_is_seekable(xpar_file * f) {
  return fseek(f->fp, 0, SEEK_CUR) != -1;
}
bool xpar_is_tty(xpar_file * f) {
  return !!isatty(fileno(f->fp));
}
bool xpar_eof(xpar_file * f)  { return !!feof(f->fp); }
int  xpar_error(xpar_file * f) { return ferror(f->fp); }

/*  -----------------------------------------------------------------------
  Safe helpers (FATAL on error)  */

sz xpar_xread(xpar_file * f, void * p, sz n) {
  sz got = fread(p, 1, n, f->fp);
  if (ferror(f->fp)) FATAL_PERROR("fread");
  return got;
}
void xpar_xwrite(xpar_file * f, const void * p, sz n) {
  if (fwrite(p, 1, n, f->fp) != n) FATAL_PERROR("fwrite");
  if (ferror(f->fp)) FATAL_PERROR("fwrite");
}
void xpar_xclose(xpar_file * f) {
  if (!f) return;
  if (fflush(f->fp)) FATAL_PERROR("fflush");
  if (f->owned) {
    /*  No fsync on DOS: writes are already synchronous.  */
    if (fclose(f->fp) == EOF) FATAL_PERROR("fclose");
    free(f);
  }
}
void xpar_notty(xpar_file * f) {
  if (isatty(fileno(f->fp)))
    FATAL("Refusing to read/write binary data from/to a terminal.");
  errno = 0;
}

/*  -----------------------------------------------------------------------
  Filesystem  */

int xpar_stat_path(const char * path, xpar_stat_t * out) {
  struct stat st;
  if (stat(path, &st) != 0) return -1;
  out->size       = (u64) st.st_size;
  out->is_dir     = !!S_ISDIR(st.st_mode);
  out->is_regular = !!S_ISREG(st.st_mode);
  return 0;
}
int xpar_remove(const char * path) { return unlink(path); }
int xpar_same_file(const char * a, const char * b) {
  struct stat sa, sb;
  if (stat(a, &sa) != 0 || stat(b, &sb) != 0) return -1;
  return (sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino) ? 1 : 0;
}

/*  No mmap on DJGPP; callers treat NULL as "declined".  */

xpar_mmap xpar_map(const char * path) {
  (void) path;
  xpar_mmap m = { NULL, 0 };
  return m;
}
void xpar_unmap(xpar_mmap * m) { m->map = NULL; m->size = 0; }

/*  -----------------------------------------------------------------------
  Allocation  */

void * xpar_malloc(sz n) {
  void * p = calloc(n ? n : 1, 1);
  if (!p) FATAL("out of memory");
  return p;
}
void * xpar_alloc_raw(sz n) {
  void * p = malloc(n ? n : 1);
  if (!p) FATAL("out of memory");
  return p;
}
void * xpar_realloc(void * p, sz n) {
  void * q = realloc(p, n);
  if (!q && n) FATAL("out of memory");
  return q;
}
void xpar_free(void * p) { free(p); }

/*  Strings: strdup/strndup only; mem... and str... go via builtins.  */

char * xpar_strdup(const char * s) {
  sz n = strlen(s) + 1;
  char * c = xpar_alloc_raw(n);
  memcpy(c, s, n);
  return c;
}
char * xpar_strndup(const char * s, sz n) {
  sz len = 0;
  while (len < n && s[len]) len++;
  char * c = xpar_alloc_raw(len + 1);
  memcpy(c, s, len);
  c[len] = '\0';
  return c;
}

/*  -----------------------------------------------------------------------
  Numeric parsing  */

int xpar_parse_i64(const char * s, i64 * out) {
  char * end;
  errno = 0;
  long long v = strtoll(s, &end, 10);
  if (errno || end == s || *end) return -1;
  *out = (i64) v;
  return 0;
}
int xpar_parse_u64(const char * s, u64 * out) {
  char * end;
  errno = 0;
  unsigned long long v = strtoull(s, &end, 10);
  if (errno || end == s || *end) return -1;
  *out = (u64) v;
  return 0;
}

/*  -----------------------------------------------------------------------
  Formatted output  */

int xpar_vsnprintf(char * buf, sz cap, const char * fmt, va_list ap) {
  return vsnprintf(buf, cap, fmt, ap);
}
int xpar_snprintf(char * buf, sz cap, const char * fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = vsnprintf(buf, cap, fmt, ap);
  va_end(ap);
  return r;
}
int xpar_asprintf(char ** out, const char * fmt, ...) {
  va_list ap; va_start(ap, fmt);
  va_list ap2; va_copy(ap2, ap);
  int n = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (n < 0) { va_end(ap2); *out = NULL; return -1; }
  *out = xpar_alloc_raw((sz) n + 1);
  vsnprintf(*out, (sz) n + 1, fmt, ap2);
  va_end(ap2);
  return n;
}
int xpar_fprintf(xpar_file * f, const char * fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = vfprintf(f->fp, fmt, ap);
  va_end(ap);
  return r;
}
int xpar_vfprintf(xpar_file * f, const char * fmt, va_list ap) {
  return vfprintf(f->fp, fmt, ap);
}
int xpar_fputs(const char * s, xpar_file * f) { return fputs(s, f->fp); }

/*  -----------------------------------------------------------------------
  Process / errors / time  */

void xpar_exit(int code) { exit(code); }

const char * xpar_strerror(int err) { return strerror(err); }
int          xpar_errno(void)       { return errno; }

u64 xpar_usec_now(void) {
  /*  uclock(): monotonic ~1.19 MHz tick counter on PC.  */
  uclock_t t = uclock();
  /*  Overflow-safe (t*1e6)/UCLOCKS_PER_SEC.  */
  u64 q = (u64) t / (u64) UCLOCKS_PER_SEC;
  u64 r = (u64) t % (u64) UCLOCKS_PER_SEC;
  return q * 1000000ULL + r * 1000000ULL / (u64) UCLOCKS_PER_SEC;
}

/*  DOS is single-tasking: threading primitives are stubs.  */

struct xpar_mutex { char _unused; };
struct xpar_cond  { char _unused; };
struct xpar_thread { char _unused; };

xpar_mutex * xpar_mutex_new(void) {
  return xpar_alloc_raw(sizeof(xpar_mutex));
}
void xpar_mutex_free(xpar_mutex * m) { xpar_free(m); }
void xpar_mutex_lock  (xpar_mutex * m) { (void) m; }
void xpar_mutex_unlock(xpar_mutex * m) { (void) m; }

xpar_cond * xpar_cond_new(void) {
  return xpar_alloc_raw(sizeof(xpar_cond));
}
void xpar_cond_free(xpar_cond * c)   { xpar_free(c); }
void xpar_cond_wait(xpar_cond * c, xpar_mutex * m) {
  (void) c; (void) m;
  FATAL("xpar_cond_wait: threading disabled on DJGPP / DOS build");
}
void xpar_cond_signal   (xpar_cond * c) { (void) c; }
void xpar_cond_broadcast(xpar_cond * c) { (void) c; }

xpar_thread * xpar_thread_start(void (*fn)(void *), void * ctx) {
  (void) fn; (void) ctx;
  FATAL("xpar_thread_start: threading disabled on DJGPP / DOS build");
}
void xpar_thread_join(xpar_thread * t) {
  (void) t;
  FATAL("xpar_thread_join: threading disabled on DJGPP / DOS build");
}

int xpar_cpu_count(void) { return 1; }

/*  DOS entry: forward to xpar_main after host init.  */
int main(int argc, char ** argv) {
  xpar_host_init();
  return xpar_main(argc, argv);
}
