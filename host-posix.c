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

/*  POSIX host backend: thin adapter over libc.  */

#if defined(_WIN32) && !defined(XPAR_FORCE_POSIX_HOST)
#error "host-posix.c compiled on a Windows target; use host-win32.c"
#endif

#define _GNU_SOURCE
#include "common.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
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
}

/*  -----------------------------------------------------------------------
  Open / close  */

xpar_file * xpar_open(const char * path, int flags) {
  const char * mode;
  if ((flags & XPAR_O_READ) && (flags & XPAR_O_WRITE)) {
    mode = (flags & XPAR_O_TRUNCATE) ? "w+be" : "r+be";
  } else if (flags & XPAR_O_WRITE) {
    mode = (flags & XPAR_O_APPEND) ? "abe" : "wbe";
  } else {
    mode = "rbe";
  }
  FILE * fp = fopen(path, mode);
  if (!fp) return NULL;
  /*  Belt-and-braces: if the libc silently ignored the 'e' suffix, set
      FD_CLOEXEC explicitly so exec()'d children don't inherit the fd.  */
  int fd = fileno(fp);
  if (fd >= 0) {
    int fl = fcntl(fd, F_GETFD);
    if (fl != -1) fcntl(fd, F_SETFD, fl | FD_CLOEXEC);
  }
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
  return fseeko(f->fp, (off_t) off, w);
}
i64 xpar_tell(xpar_file * f) { return (i64) ftello(f->fp); }
int xpar_flush(xpar_file * f) { return fflush(f->fp); }
int xpar_fsync(xpar_file * f) {
  if (fflush(f->fp)) return -1;
  int r = fsync(fileno(f->fp));
  /*  EINVAL (unsyncable fd) / EACCES (read-only): no durability loss.  */
  if (r && (errno == EINVAL || errno == EACCES)) { errno = 0; return 0; }
  return r;
}
i64 xpar_size(xpar_file * f) {
  struct stat st;
  if (fstat(fileno(f->fp), &st) != 0) return -1;
  return (i64) st.st_size;
}
bool xpar_is_seekable(xpar_file * f) {
  return fseek(f->fp, 0, SEEK_CUR) != -1;
}
bool xpar_is_tty(xpar_file * f) {
  return isatty(fileno(f->fp));
}
bool xpar_eof(xpar_file * f)  { return feof(f->fp); }
int  xpar_error(xpar_file * f) { return ferror(f->fp); }

/*  -----------------------------------------------------------------------
  Safe helpers (FATAL on error)  */

sz xpar_xread(xpar_file * f, void * p, sz n) {
  sz got = 0;
  char * cp = (char *) p;
  while (got < n) {
    sz r = fread(cp + got, 1, n - got, f->fp);
    got += r;
    if (r == 0) {
      if (ferror(f->fp)) FATAL_PERROR("fread");
      break;
    }
  }
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
    int fd = fileno(f->fp);
    if (fsync(fd) && errno != EINVAL && errno != EACCES) FATAL_PERROR("fsync");
    errno = 0;
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

/*  -----------------------------------------------------------------------
  Memory map  */

xpar_mmap xpar_map(const char * path) {
  xpar_mmap m = { NULL, 0 };
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd == -1) return m;
  struct stat st;
  if (fstat(fd, &st) == -1) { close(fd); return m; }
  m.size = (sz) st.st_size;
  if (m.size == 0) { close(fd); return m; }
  void * p = mmap(NULL, m.size, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) { m.size = 0; return m; }
  m.map = (u8 *) p;
  return m;
}
void xpar_unmap(xpar_mmap * m) {
  if (m->map) munmap(m->map, m->size);
  m->map = NULL; m->size = 0;
}

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

/*  Strings: strdup/strndup only; mem.../str... go via builtins->libc.  */

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
#if defined(CLOCK_MONOTONIC)
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
    return (u64) ts.tv_sec * 1000000ULL + (u64) ts.tv_nsec / 1000ULL;
#endif
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (u64) tv.tv_sec * 1000000ULL + (u64) tv.tv_usec;
}

/*  -----------------------------------------------------------------------
  Thread / mutex / condvar primitives (pthread-backed)  */

struct xpar_mutex  { pthread_mutex_t m; };
struct xpar_cond   { pthread_cond_t  c; };
struct xpar_thread { pthread_t       t; };

xpar_mutex * xpar_mutex_new(void) {
  xpar_mutex * m = xpar_alloc_raw(sizeof(*m));
  if (pthread_mutex_init(&m->m, NULL) != 0) FATAL("pthread_mutex_init");
  return m;
}
void xpar_mutex_free(xpar_mutex * m) {
  pthread_mutex_destroy(&m->m); xpar_free(m);
}
void xpar_mutex_lock  (xpar_mutex * m) { pthread_mutex_lock  (&m->m); }
void xpar_mutex_unlock(xpar_mutex * m) { pthread_mutex_unlock(&m->m); }

xpar_cond * xpar_cond_new(void) {
  xpar_cond * c = xpar_alloc_raw(sizeof(*c));
  if (pthread_cond_init(&c->c, NULL) != 0) FATAL("pthread_cond_init");
  return c;
}
void xpar_cond_free(xpar_cond * c) {
  pthread_cond_destroy(&c->c); xpar_free(c);
}
void xpar_cond_wait(xpar_cond * c, xpar_mutex * m) {
  pthread_cond_wait(&c->c, &m->m);
}
void xpar_cond_signal   (xpar_cond * c) { pthread_cond_signal   (&c->c); }
void xpar_cond_broadcast(xpar_cond * c) { pthread_cond_broadcast(&c->c); }

struct xpar_thread_shim { void (*fn)(void *); void * ctx; };
static void * xpar_thread_trampoline(void * p) {
  struct xpar_thread_shim s = *(struct xpar_thread_shim *) p;
  xpar_free(p);
  s.fn(s.ctx);
  return NULL;
}
xpar_thread * xpar_thread_start(void (*fn)(void *), void * ctx) {
  struct xpar_thread_shim * s = xpar_alloc_raw(sizeof(*s));
  s->fn = fn; s->ctx = ctx;
  xpar_thread * t = xpar_alloc_raw(sizeof(*t));
  if (pthread_create(&t->t, NULL, xpar_thread_trampoline, s) != 0)
    FATAL("pthread_create");
  return t;
}
void xpar_thread_join(xpar_thread * t) {
  pthread_join(t->t, NULL); xpar_free(t);
}

int xpar_cpu_count(void) {
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (int) n : 1;
}

/*  POSIX entry: the usual main() just forwards.  */
int main(int argc, char ** argv) {
  xpar_host_init();
  return xpar_main(argc, argv);
}
