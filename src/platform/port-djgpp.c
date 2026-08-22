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

/*  DJGPP host I/O over DOS handles and DPMI.

    Signed 32-bit off_t bounds every libc offset.  Positional I/O moves the
    shared offset and relies on the single-threaded build.  Secure random
    bytes, and therefore authentication, are unavailable.  */

#if !defined(__DJGPP__)
#error "port-djgpp.c compiled for a non-DJGPP target"
#endif

#include "common.h"

#include <dir.h>
#include <dos.h>
#include <dpmi.h>
#include <errno.h>
#include <fcntl.h>
#include <go32.h>
#include <io.h>
#include <pc.h>
#include <stdlib.h>
#include <string.h>
#include <sys/farptr.h>
#include <sys/time.h>

const char * xpar_getenv(const char * name) { return getenv(name); }
#include <time.h>
#include <unistd.h>

struct xpar_file {
  int  fd;
  bool is_char;   /*  console, printer or another character device  */
  bool owned;
  bool at_eof;
  int  last_errno;
};

static struct xpar_file g_stdin  = { 0, false, false, false, 0 };
static struct xpar_file g_stdout = { 1, false, false, false, 0 };
static struct xpar_file g_stderr = { 2, false, false, false, 0 };

xpar_file * const xpar_stdin  = &g_stdin;
xpar_file * const xpar_stdout = &g_stdout;
xpar_file * const xpar_stderr = &g_stderr;

/*  int 21h AX=4400h: get device information for a handle. Bit 7 of DX
    distinguishes a character device from a file, which is the only part
    of the answer this program needs.  */
static bool fd_is_char(int fd) {
  __dpmi_regs r;
  memset(&r, 0, sizeof r);
  r.x.ax = 0x4400;
  r.x.bx = (unsigned short) fd;
  if (__dpmi_int(0x21, &r) < 0) return false;
  if (r.x.flags & 1) return false;
  return (r.x.dx & 0x80) != 0;
}

void xpar_host_init(void) {
  g_stdin.is_char  = fd_is_char(0);
  g_stdout.is_char = fd_is_char(1);
  g_stderr.is_char = fd_is_char(2);
  /*  A redirected standard stream carries container bytes, and DOS text
      mode would turn every 0x0A into 0x0D 0x0A on the way out and eat
      0x1A on the way in.  */
  if (!g_stdin.is_char)  setmode(0, O_BINARY);
  if (!g_stdout.is_char) setmode(1, O_BINARY);
  /*  Probe for a long-filename driver once, so open and findfirst take the
      AX=71xxh forms where the drive supports them.  */
  _use_lfn(".");
}

/*  off_t is a signed 32-bit long here, so anything past 2 GiB - 1 is
    rejected at the boundary instead of wrapping.  */
static bool off_fits(u64 v) { return v <= 0x7fffffffULL; }

xpar_file * xpar_open(const char * path, int flags) {
  int acc = flags & 3, oflag;
  int fd;
  struct xpar_file * f;
  if      (acc == XPAR_O_WRONLY) oflag = O_WRONLY;
  else if (acc == XPAR_O_RDWR)   oflag = O_RDWR;
  else                           oflag = O_RDONLY;
  oflag |= O_BINARY;
  if (flags & XPAR_O_CREAT)  oflag |= O_CREAT;
  if (flags & XPAR_O_TRUNC)  oflag |= O_TRUNC;
  if (flags & XPAR_O_EXCL)   oflag |= O_EXCL;
  if (flags & XPAR_O_APPEND) oflag |= O_APPEND;
  fd = open(path, oflag, 0666);
  if (fd < 0) return NULL;
  f = malloc(sizeof(*f));
  if (!f) { close(fd);  errno = ENOMEM;  return NULL; }
  f->fd = fd;  f->is_char = fd_is_char(fd);  f->owned = true;
  f->at_eof = false;  f->last_errno = 0;
  return f;
}

int xpar_close(xpar_file * f) {
  int r;
  if (!f || !f->owned) return 0;
  r = close(f->fd);
  free(f);
  return r;
}

sz xpar_read(xpar_file * f, void * buf, sz n) {
  sz total = 0;
  u8 * p = (u8 *) buf;
  while (total < n) {
    sz want = n - total;
    int got;
    if (want > 0x10000000u) want = 0x10000000u;
    got = read(f->fd, p + total, (unsigned) want);
    if (got < 0)  { f->last_errno = errno;  break; }
    if (got == 0) { f->at_eof = true;       break; }
    total += (sz) got;
  }
  return total;
}

sz xpar_write(xpar_file * f, const void * buf, sz n) {
  sz total = 0;
  const u8 * p = (const u8 *) buf;
  while (total < n) {
    sz want = n - total;
    int wrote;
    if (want > 0x10000000u) want = 0x10000000u;
    wrote = write(f->fd, p + total, (unsigned) want);
    if (wrote < 0)  { f->last_errno = errno;  break; }
    if (wrote == 0) break;
    total += (sz) wrote;
  }
  return total;
}

int xpar_seek(xpar_file * f, i64 off, int whence) {
  int w = whence == XPAR_SEEK_SET ? SEEK_SET
        : whence == XPAR_SEEK_CUR ? SEEK_CUR : SEEK_END;
  if (off > 0 && !off_fits((u64) off)) { errno = EOVERFLOW;  return -1; }
  if (off < -0x7fffffffLL)             { errno = EOVERFLOW;  return -1; }
  if (lseek(f->fd, (long) off, w) < 0) { f->last_errno = errno;  return -1; }
  f->at_eof = false;
  return 0;
}

i64 xpar_tell(xpar_file * f) {
  long p = lseek(f->fd, 0, SEEK_CUR);
  return p < 0 ? -1 : (i64) p;
}

/*  DOS writes go straight to the file system through int 21h; there is no
    user-space buffer to drain and no fsync to ask for.  */
int xpar_flush(xpar_file * f) { (void) f;  return 0; }
int xpar_fsync(xpar_file * f) { (void) f;  return 0; }

i64 xpar_size(xpar_file * f) {
  long len = filelength(f->fd);
  return len < 0 ? -1 : (i64) len;
}

bool xpar_is_seekable(xpar_file * f) {
  return lseek(f->fd, 0, SEEK_CUR) >= 0;
}
bool xpar_is_tty(xpar_file * f) { return f->is_char; }
bool xpar_eof   (xpar_file * f) { return f->at_eof; }
int  xpar_error (xpar_file * f) { return f->last_errno; }

/*  Advisory locks: not available.  */

int xpar_lock(xpar_file * f, bool exclusive) {
  (void) f;  (void) exclusive;
  errno = ENOSYS;
  return -1;
}

/*  Nothing was ever taken, so releasing it succeeds trivially; a caller
    unwinding an error path must not have to special-case this host.  */
int xpar_unlock(xpar_file * f) { (void) f;  return 0; }

bool xpar_lock_supported(void) { return false; }

sz xpar_pread(xpar_file * f, void * buf, sz n, u64 off) {
  if (!off_fits(off)) { f->last_errno = EOVERFLOW;  return 0; }
  if (lseek(f->fd, (long) off, SEEK_SET) < 0) {
    f->last_errno = errno;
    return 0;
  }
  return xpar_read(f, buf, n);
}

bool xpar_pread_batch(xpar_read_req * r, sz count) {
  sz i;
  for (i = 0; i < count; i++)
    r[i].result = r[i].file
                    ? xpar_pread(r[i].file, r[i].buf, r[i].length,
                                 r[i].offset) : 0;
  return false;
}

sz xpar_pwrite(xpar_file * f, const void * buf, sz n, u64 off) {
  if (!off_fits(off)) { f->last_errno = EOVERFLOW;  return 0; }
  if (lseek(f->fd, (long) off, SEEK_SET) < 0) {
    f->last_errno = errno;
    return 0;
  }
  return xpar_write(f, buf, n);
}

int xpar_ftruncate(xpar_file * f, u64 length) {
  if (!off_fits(length)) { errno = EOVERFLOW;  return -1; }
  if (ftruncate(f->fd, (long) length) != 0) {
    f->last_errno = errno;
    return -1;
  }
  return 0;
}

/*  FAT has no directory to sync: a directory entry is written by the same
    int 21h call that created the file.  */
int xpar_fsync_dir(const char * path) { (void) path;  return 0; }

sz xpar_xread(xpar_file * f, void * p, sz n) {
  sz got = xpar_read(f, p, n);
  if (f->last_errno) { errno = f->last_errno;  FATAL_PERROR("read"); }
  return got;
}

void xpar_xwrite(xpar_file * f, const void * p, sz n) {
  if (xpar_write(f, p, n) != n) {
    errno = f->last_errno ? f->last_errno : ENOSPC;
    FATAL_PERROR("write");
  }
}

void xpar_xwritev(xpar_file * f, const xpar_write_part * part, u32 count) {
  u32 i;
  for (i = 0; i < count; i++)
    xpar_xwrite(f, part[i].data, part[i].length);
}

void xpar_xclose(xpar_file * f) {
  if (!f || !f->owned) return;
  if (close(f->fd) < 0) FATAL_PERROR("close");
  free(f);
}

xpar_mmap xpar_map(const char * path) {
  xpar_mmap m;
  (void) path;
  m.map = NULL;  m.size = 0;  m.valid = false;
  return m;
}

void xpar_unmap(xpar_mmap * m) {
  m->map = NULL;  m->size = 0;  m->valid = false;
}

void xpar_advise_sequential(xpar_file * f, u64 off, u64 len) {
  (void) f;  (void) off;  (void) len;
}
void xpar_advise_random(xpar_file * f, u64 off, u64 len) {
  (void) f;  (void) off;  (void) len;
}

void * xpar_malloc(sz n) {
  void * p = calloc(n ? n : 1, 1);
  if (!p) FATAL("Out of memory.");
  return p;
}

void * xpar_calloc(sz n, sz size) {
  if (n && size && n > (sz) -1 / size) FATAL("Allocation size overflow.");
  { void * p = calloc(n ? n : 1, size ? size : 1);
    if (!p) FATAL("Out of memory.");
    return p; }
}

void * xpar_alloc_raw(sz n) {
  void * p = malloc(n ? n : 1);
  if (!p) FATAL("Out of memory.");
  return p;
}

void * xpar_realloc(void * p, sz n) {
  void * q = realloc(p, n ? n : 1);
  if (!q) FATAL("Out of memory.");
  return q;
}

void xpar_free(void * p) { free(p); }

void * xpar_alloc_aligned(sz n, sz align) {
  u8 * raw;
  uintptr_t a;
  sz pad;
  if (align < sizeof(void *)) align = sizeof(void *);
  if (!xpar_is_pow2(align)) FATAL("Alignment is not a power of two.");
  if (n == 0) n = 1;
  pad = align + sizeof(void *);
  if (n > (sz) -1 - pad) FATAL("Allocation size overflow.");
  raw = malloc(n + pad);
  if (!raw) FATAL("Out of memory.");
  a = ((uintptr_t) raw + sizeof(void *) + align - 1) &
      ~(uintptr_t) (align - 1);
  ((void **) a)[-1] = raw;
  return (void *) a;
}

void xpar_free_aligned(void * p) {
  if (p) free(((void **) p)[-1]);
}

/*  DPMI function 0500h reports the host's idea of available memory. The
    page count is what the planner wants; a DPMI host that declines the
    call leaves the planner on its 8 MiB default.  */
u64 xpar_physical_memory(void) {
  __dpmi_free_mem_info info;
  if (__dpmi_get_free_memory_information(&info) != 0) return 0;
  if (info.total_number_of_free_pages == 0) return 0;
  return (u64) info.total_number_of_free_pages * 4096ULL;
}

bool xpar_is_rotational(const char * path) {
  /*  Almost certainly true on this host, and deliberately not claimed: the
      planner only uses the answer as a hint and a wrong "true" would push
      every plan towards a layout the user did not ask for.  */
  (void) path;
  return false;
}

void * xpar_memcpy (void * d, const void * s, sz n) { return memcpy(d, s, n); }
void * xpar_memmove(void * d, const void * s, sz n) { return memmove(d, s, n); }
void * xpar_memset (void * d, int c, sz n)          { return memset(d, c, n); }
int    xpar_memcmp (const void * a, const void * b, sz n) {
  return memcmp(a, b, n);
}
sz     xpar_strlen (const char * s)                 { return strlen(s); }
int    xpar_strcmp (const char * a, const char * b) { return strcmp(a, b); }
int    xpar_strncmp(const char * a, const char * b, sz n) {
  return strncmp(a, b, n);
}

char * xpar_strdup(const char * s) {
  sz n = strlen(s) + 1;
  char * c = xpar_alloc_raw(n);
  memcpy(c, s, n);
  return c;
}

char * xpar_strndup(const char * s, sz n) {
  sz len = 0;
  char * c;
  while (len < n && s[len]) len++;
  c = xpar_alloc_raw(len + 1);
  memcpy(c, s, len);
  c[len] = '\0';
  return c;
}

int xpar_parse_u64(const char * s, u64 * out) {
  u64 v = 0;
  if (!s || !*s) return -1;
  if (*s == '+') s++;
  if (!*s) return -1;
  while (*s) {
    u64 nv;
    if (*s < '0' || *s > '9') return -1;
    nv = v * 10 + (u64) (*s - '0');
    if (nv < v) return -1;
    v = nv;  s++;
  }
  *out = v;
  return 0;
}

int xpar_parse_i64(const char * s, i64 * out) {
  int neg = 0;
  u64 v;
  if (!s || !*s) return -1;
  if (*s == '-') { neg = 1;  s++; }
  if (xpar_parse_u64(s, &v) != 0) return -1;
  if (neg) {
    if (v > (u64) INT64_MAX + 1) return -1;
    *out = v == (u64) INT64_MAX + 1 ? INT64_MIN : -(i64) v;
  } else {
    if (v > (u64) INT64_MAX) return -1;
    *out = (i64) v;
  }
  return 0;
}

typedef struct {
  char * buf;
  sz     cap;
  sz     pos;   /*  bytes that WOULD be written, for snprintf sizing  */
} fmt_ctx;

static void emit_c(fmt_ctx * c, char ch) {
  if (c->buf && c->pos + 1 < c->cap) c->buf[c->pos] = ch;
  c->pos++;
}
static void emit_str(fmt_ctx * c, const char * s, sz n) {
  for (sz i = 0; i < n; i++) emit_c(c, s[i]);
}
static void emit_pad(fmt_ctx * c, int n, char ch) {
  while (n-- > 0) emit_c(c, ch);
}

enum { F_MINUS = 1, F_PLUS = 2, F_SPACE = 4, F_ZERO = 8, F_HASH = 16 };

static void emit_uint(fmt_ctx * c, u64 v, int base, int upper,
                      int width, int prec, int flags) {
  char tmp[32];
  int n = 0, len, pad_zero, total, pad_sp;
  const char * digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
  if (v == 0 && prec == 0) n = 0;
  else do { tmp[n++] = digits[v % (u64) base];  v /= (u64) base; } while (v);
  len      = n;
  pad_zero = prec > len ? prec - len : 0;
  total    = len + pad_zero;
  pad_sp   = width > total ? width - total : 0;
  if (!(flags & F_MINUS) && !(flags & F_ZERO)) emit_pad(c, pad_sp, ' ');
  if (!(flags & F_MINUS) && (flags & F_ZERO) && prec < 0)
    emit_pad(c, pad_sp, '0');
  emit_pad(c, pad_zero, '0');
  while (n) emit_c(c, tmp[--n]);
  if (flags & F_MINUS) emit_pad(c, pad_sp, ' ');
}

static void emit_int(fmt_ctx * c, i64 v, int width, int prec, int flags) {
  char sign = 0;
  u64 uv;
  /*  (-(v+1))+1 stays in range where -v would not, at INT64_MIN.  */
  if (v < 0)                { sign = '-';  uv = (u64) (-(v + 1)) + 1; }
  else if (flags & F_PLUS)  { sign = '+';  uv = (u64) v; }
  else if (flags & F_SPACE) { sign = ' ';  uv = (u64) v; }
  else                      { uv = (u64) v; }
  if (sign) {
    if (width > 0 && !(flags & F_MINUS) && !(flags & F_ZERO)) {
      char tmp[32];
      int n = 0, len, pad_zero, total, pad_sp;
      if (uv == 0 && prec == 0) n = 0;
      else { u64 x = uv;
             do { tmp[n++] = (char) ('0' + (x % 10));  x /= 10; } while (x); }
      len      = n;
      pad_zero = prec > len ? prec - len : 0;
      total    = 1 + len + pad_zero;
      pad_sp   = width > total ? width - total : 0;
      emit_pad(c, pad_sp, ' ');
      emit_c(c, sign);
      emit_pad(c, pad_zero, '0');
      while (n) emit_c(c, tmp[--n]);
      return;
    }
    emit_c(c, sign);
    width = width > 0 ? width - 1 : 0;
  }
  emit_uint(c, uv, 10, 0, width, prec, flags);
}

static void emit_double(fmt_ctx * c, double v, int width, int prec,
                        int flags) {
  char sign = 0, ibuf[24];
  u64 ip, fp, mult = 1;
  double frac;
  int in = 0, total, pad;
  if (prec < 0) prec = 6;
  if (v < 0)                { sign = '-';  v = -v; }
  else if (flags & F_PLUS)  { sign = '+'; }
  else if (flags & F_SPACE) { sign = ' '; }
  ip   = (u64) v;
  frac = v - (double) ip;
  for (int i = 0; i < prec; i++) mult *= 10;
  fp = (u64) (frac * (double) mult + 0.5);
  if (fp >= mult) { ip++;  fp -= mult; }
  if (ip == 0) ibuf[in++] = '0';
  else { u64 x = ip;
         while (x) { ibuf[in++] = (char) ('0' + (x % 10));  x /= 10; } }
  total = in + (prec > 0 ? 1 + prec : 0) + (sign ? 1 : 0);
  pad   = width > total ? width - total : 0;
  if (!(flags & F_MINUS) && !(flags & F_ZERO)) emit_pad(c, pad, ' ');
  if (sign) emit_c(c, sign);
  if (!(flags & F_MINUS) && (flags & F_ZERO)) emit_pad(c, pad, '0');
  while (in) emit_c(c, ibuf[--in]);
  if (prec > 0) {
    char fbuf[24];
    int fn = 0;
    u64 x = fp;
    emit_c(c, '.');
    for (int i = 0; i < prec; i++) { fbuf[fn++] = (char) ('0' + (x % 10));
                                     x /= 10; }
    while (fn) emit_c(c, fbuf[--fn]);
  }
  if (flags & F_MINUS) emit_pad(c, pad, ' ');
}

int xpar_vsnprintf(char * buf, sz cap, const char * fmt, va_list ap) {
  fmt_ctx c;
  c.buf = buf;  c.cap = cap;  c.pos = 0;
  while (*fmt) {
    int flags = 0, width = 0, prec = -1, longness = 0;
    char spec;
    if (*fmt != '%') { emit_c(&c, *fmt++);  continue; }
    fmt++;
    for (;; fmt++) {
      if      (*fmt == '-') flags |= F_MINUS;
      else if (*fmt == '+') flags |= F_PLUS;
      else if (*fmt == ' ') flags |= F_SPACE;
      else if (*fmt == '0') flags |= F_ZERO;
      else if (*fmt == '#') flags |= F_HASH;
      else break;
    }
    if (*fmt == '*') { width = va_arg(ap, int);  fmt++; }
    else while (*fmt >= '0' && *fmt <= '9')
      { width = width * 10 + (*fmt - '0');  fmt++; }
    if (*fmt == '.') {
      fmt++;  prec = 0;
      if (*fmt == '*') { prec = va_arg(ap, int);  fmt++; }
      else while (*fmt >= '0' && *fmt <= '9')
        { prec = prec * 10 + (*fmt - '0');  fmt++; }
    }
    /*  0 = int, 1 = long, 2 = long long, 3 = size_t  */
    if (*fmt == 'z') { longness = 3;  fmt++; }
    else if (*fmt == 'l') {
      fmt++;
      if (*fmt == 'l') { longness = 2;  fmt++; }
      else longness = 1;
    } else if (*fmt == 'h') {
      fmt++;
      if (*fmt == 'h') fmt++;
    }
    spec = *fmt;
    if (spec) fmt++;
    switch (spec) {
      case 'd': case 'i': {
        i64 v;
        if      (longness == 2) v = va_arg(ap, long long);
        else if (longness == 1) v = va_arg(ap, long);
        else if (longness == 3) v = (i64) va_arg(ap, ptrdiff_t);
        else                    v = va_arg(ap, int);
        emit_int(&c, v, width, prec, flags);
        break;
      }
      case 'u': case 'x': case 'X': {
        u64 v;
        if      (longness == 2) v = va_arg(ap, unsigned long long);
        else if (longness == 1) v = va_arg(ap, unsigned long);
        else if (longness == 3) v = (u64) va_arg(ap, sz);
        else                    v = va_arg(ap, unsigned);
        emit_uint(&c, v, spec == 'u' ? 10 : 16, spec == 'X',
                  width, prec, flags);
        break;
      }
      case 'p': {
        void * p = va_arg(ap, void *);
        emit_str(&c, "0x", 2);
        emit_uint(&c, (u64) (uintptr_t) p, 16, 0, 8, -1, F_ZERO);
        break;
      }
      case 'c': {
        int ch  = va_arg(ap, int);
        int pad = width > 1 ? width - 1 : 0;
        if (!(flags & F_MINUS)) emit_pad(&c, pad, ' ');
        emit_c(&c, (char) ch);
        if (flags & F_MINUS) emit_pad(&c, pad, ' ');
        break;
      }
      case 's': {
        const char * s = va_arg(ap, const char *);
        sz slen = 0;
        int pad;
        if (!s) s = "(null)";
        while (s[slen] && (prec < 0 || slen < (sz) prec)) slen++;
        pad = width > (int) slen ? width - (int) slen : 0;
        if (!(flags & F_MINUS)) emit_pad(&c, pad, ' ');
        emit_str(&c, s, slen);
        if (flags & F_MINUS) emit_pad(&c, pad, ' ');
        break;
      }
      case 'f': emit_double(&c, va_arg(ap, double), width, prec, flags);
                break;
      case '%': emit_c(&c, '%');  break;
      default:  emit_c(&c, '%');  if (spec) emit_c(&c, spec);  break;
    }
  }
  if (c.buf && c.cap > 0)
    c.buf[c.pos < c.cap ? c.pos : c.cap - 1] = '\0';
  return (int) c.pos;
}

int xpar_snprintf(char * buf, sz cap, const char * fmt, ...) {
  va_list ap;  int r;
  va_start(ap, fmt);
  r = xpar_vsnprintf(buf, cap, fmt, ap);
  va_end(ap);
  return r;
}

int xpar_asprintf(char ** out, const char * fmt, ...) {
  va_list ap, ap2;  int n;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  n = xpar_vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (n < 0) { va_end(ap2);  *out = NULL;  return -1; }
  *out = xpar_alloc_raw((sz) n + 1);
  xpar_vsnprintf(*out, (sz) n + 1, fmt, ap2);
  va_end(ap2);
  return n;
}

int xpar_vfprintf(xpar_file * f, const char * fmt, va_list ap) {
  char stack[1024];
  va_list ap2;
  int n;
  va_copy(ap2, ap);
  n = xpar_vsnprintf(stack, sizeof stack, fmt, ap);
  if (n < (int) sizeof stack) {
    va_end(ap2);
    xpar_write(f, stack, (sz) n);
    return n;
  }
  { char * big = xpar_alloc_raw((sz) n + 1);
    xpar_vsnprintf(big, (sz) n + 1, fmt, ap2);
    va_end(ap2);
    xpar_write(f, big, (sz) n);
    xpar_free(big); }
  return n;
}

int xpar_fprintf(xpar_file * f, const char * fmt, ...) {
  va_list ap;  int r;
  va_start(ap, fmt);
  r = xpar_vfprintf(f, fmt, ap);
  va_end(ap);
  return r;
}

int xpar_fputs(const char * s, xpar_file * f) {
  sz n = strlen(s);
  return (int) xpar_write(f, s, n);
}

void xpar_exit(int code) {
  /*  int 21h AH=4Ch: terminate with a return code.  */
  __dpmi_regs r;
  memset(&r, 0, sizeof r);
  r.h.ah = 0x4C;
  r.h.al = (unsigned char) code;
  __dpmi_int(0x21, &r);
  for (;;) { }   /*  unreachable unless the DPMI host misbehaves  */
}

/*  The codes DJGPP's int 21h wrappers actually produce. Anything else
    falls through to a number, which is more use than a wrong name.  */
static const struct { int n; const char * s; } err_tab[] = {
  { EACCES,       "permission denied"              },
  { EEXIST,       "file exists"                    },
  { ENOENT,       "no such file or directory"      },
  { EBADF,        "bad file descriptor"            },
  { EINVAL,       "invalid argument"               },
  { EIO,          "I/O error"                      },
  { ENOMEM,       "out of memory"                  },
  { ENOSPC,       "no space left on device"        },
  { ENFILE,       "too many open files in system"  },
  { EMFILE,       "too many open files"            },
  { ENOTDIR,      "not a directory"                },
  { EISDIR,       "is a directory"                 },
  { ENAMETOOLONG, "filename too long"              },
  { EOVERFLOW,    "value too large"                },
  { EXDEV,        "cross-device link"              }
};

static char err_gen[32];

const char * xpar_strerror(int err) {
  for (sz i = 0; i < ARRAY_LEN(err_tab); i++)
    if (err_tab[i].n == err) return err_tab[i].s;
  xpar_snprintf(err_gen, sizeof err_gen, "error %d", err);
  return err_gen;
}

int xpar_errno(void) { return errno; }

/*  8254 PIT channel 0 runs at 1,193,181.8 Hz and the BIOS programs it in
    mode 3 with N = 65536, giving the 54.9254 ms tick at 0040:006C. In mode
    3 the counter descends N -> 0 twice per tick, decrementing by two per
    clock, and OUT is high for the first half and low for the second. The
    read-back command latches the count and the status byte together, and
    the OUT bit is what disambiguates the two halves into one monotonic
    timeline. Resolution is about 1.68 us; the tick counter wraps at
    midnight, which is the one discontinuity a long run can see.  */
u64 xpar_usec_now(void) {
  u32 bios0, bios1, count, out_hi, elapsed;
  u64 pit;
  do {
    u8 status;
    bios0 = _farpeekl(_dos_ds, 0x46CUL);
    outportb(0x43, 0xC2);
    status = inportb(0x40);
    count  = (u32) inportb(0x40);
    count |= (u32) inportb(0x40) << 8;
    out_hi = (status >> 7) & 1;
    bios1  = _farpeekl(_dos_ds, 0x46CUL);
  } while (bios0 != bios1);   /*  retry if IRQ0 landed inside the read  */
  elapsed = (65536u - count) >> 1;
  if (!out_hi) elapsed += 32768u;
  pit = (u64) bios1 * 65536ULL + (u64) elapsed;
  return (pit * 1000000ULL) / 1193182ULL;
}

/*  The DOS clock is the BIOS tick, so this is accurate to 55 ms and no
    better. Stored timestamps on this host are FAT timestamps anyway,
    which are quantised to two seconds.  */
i64 xpar_wall_ns(void) {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) != 0) return (i64) time(NULL) * 1000000000LL;
  return (i64) tv.tv_sec * 1000000000LL + (i64) tv.tv_usec * 1000LL;
}

/*  MS-DOS has no entropy source.  */
void xpar_random_bytes(void * buf, sz n) {
  (void) buf;  (void) n;
  FATAL("MS-DOS has no source of cryptographically strong random bytes.");
}

int main(int argc, char ** argv) {
  xpar_host_init();
  return xpar_main(argc, argv);
}
