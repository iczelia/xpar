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

/*  DJGPP/MS-DOS host: <io.h> handle I/O + int21h via DPMI.  */

#if !defined(__DJGPP__)
#error "host-djgpp.c compiled on a non-DJGPP target"
#endif

#include "common.h"

#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <stdarg.h>
#include <stdlib.h>
#include <unistd.h>
#include <dir.h>
#include <dos.h>
#include <dpmi.h>
#include <go32.h>
#include <pc.h>
#include <sys/farptr.h>

/*  -----------------------------------------------------------------------
  xpar_file: opaque wrapper around a DOS file handle  */

struct xpar_file {
  int   fd;
  u8    kind;       /*  0 = file/disk, 1 = char/pipe, 2 = unknown  */
  bool  owned;
  bool  at_eof;
  int   last_errno;
};

static struct xpar_file g_stdin  = { 0, 0, false, false, 0 };
static struct xpar_file g_stdout = { 1, 0, false, false, 0 };
static struct xpar_file g_stderr = { 2, 0, false, false, 0 };

xpar_file * const xpar_stdin  = &g_stdin;
xpar_file * const xpar_stdout = &g_stdout;
xpar_file * const xpar_stderr = &g_stderr;

/*  DOS IOCTL AX=4400h: get device info for handle. Bit 7 of DX = char
    device (terminal, pipe, device driver). Bit 14 of DX = !redirected
    for char devices; we only care whether we're talking to a tty.  */
static unsigned dos_get_dev_info(int fd) {
  __dpmi_regs r;
  xpar_memset(&r, 0, sizeof(r));
  r.x.ax = 0x4400;
  r.x.bx = fd;
  if (__dpmi_int(0x21, &r) < 0) return 0;
  if (r.x.flags & 1) return 0;
  return r.x.dx;
}

static bool dos_fd_is_tty(int fd) {
  unsigned di = dos_get_dev_info(fd);
  /*  Char device and isn't redirected (bit 7 set, bit 14 clear if char).  */
  if (!(di & 0x80)) return false;
  /*  Reserved bit clear + bit 0 (stdin) or bit 1 (stdout) indicates
      console; keep it simple -- treat any non-redirected char device as a
      terminal, which is what DJGPP's isatty does.  */
  return true;
}

void xpar_host_init(void) {
  /*  Binary mode for non-tty std streams so raw bytes pass through.  */
  if (!dos_fd_is_tty(0)) setmode(0, O_BINARY);
  if (!dos_fd_is_tty(1)) setmode(1, O_BINARY);
  /*  Prime the LFN probe; open/findfirst/unlink route through AX=71xxh
      when the drive supports long names. dos_truename() tries LFN
      TRUENAME (AX=7160h) with a fallback to the short-name AH=60h.  */
  _use_lfn(".");
}

/*  -----------------------------------------------------------------------
  Open / close  */

xpar_file * xpar_open(const char * path, int flags) {
  int oflag = 0;
  if ((flags & XPAR_O_READ) && (flags & XPAR_O_WRITE)) oflag = O_RDWR;
  else if (flags & XPAR_O_WRITE)                       oflag = O_WRONLY;
  else                                                 oflag = O_RDONLY;
  oflag |= O_BINARY;
  if (flags & XPAR_O_CREATE)    oflag |= O_CREAT;
  if (flags & XPAR_O_TRUNCATE)  oflag |= O_TRUNC;
  if (flags & XPAR_O_EXCLUSIVE) oflag |= O_EXCL;
  if (flags & XPAR_O_APPEND)    oflag |= O_APPEND;
  int fd = open(path, oflag, 0666);
  if (fd < 0) return NULL;
  struct xpar_file * f = malloc(sizeof(*f));
  if (!f) { close(fd); errno = ENOMEM; return NULL; }
  f->fd = fd;
  f->kind = dos_fd_is_tty(fd) ? 1 : 0;
  f->owned = true;
  f->at_eof = false;
  f->last_errno = 0;
  return f;
}

int xpar_close(xpar_file * f) {
  if (!f || !f->owned) return 0;
  int r = close(f->fd);
  free(f);
  return r;
}

/*  -----------------------------------------------------------------------
  Read / write / seek / tell  */

sz xpar_read(xpar_file * f, void * buf, sz n) {
  /*  Loop to deliver full-count semantics on partial reads/interrupts;
      short count only on EOF or error.  */
  sz total = 0;
  unsigned char * p = buf;
  while (total < n) {
    sz want = n - total;
    /*  _read's count is size_t; DJGPP internally chunks into int21h
        calls. Cap anyway to avoid signed overflow in ssize_t returns.  */
    if (want > 0x40000000u) want = 0x40000000u;
    int got = read(f->fd, p + total, want);
    if (got < 0) { f->last_errno = errno; return total; }
    if (got == 0) { f->at_eof = true; break; }
    total += (sz) got;
  }
  return total;
}

sz xpar_write(xpar_file * f, const void * buf, sz n) {
  sz total = 0;
  const unsigned char * p = buf;
  while (total < n) {
    sz want = n - total;
    if (want > 0x40000000u) want = 0x40000000u;
    int wrote = write(f->fd, p + total, want);
    if (wrote < 0) { f->last_errno = errno; return total; }
    if (wrote == 0) break;
    total += (sz) wrote;
  }
  return total;
}

int xpar_seek(xpar_file * f, i64 off, int whence) {
  int w = whence == XPAR_SEEK_SET ? SEEK_SET
        : whence == XPAR_SEEK_CUR ? SEEK_CUR : SEEK_END;
  /*  DJGPP off_t is 32-bit; reject seeks beyond 2 GiB.  */
  if (off > (i64) INT32_MAX || off < -(i64) INT32_MAX) {
    errno = EOVERFLOW; return -1;
  }
  return lseek(f->fd, (long) off, w) < 0 ? -1 : 0;
}

i64 xpar_tell(xpar_file * f) {
  off_t p = tell(f->fd);
  return p < 0 ? -1 : (i64) p;
}

int xpar_flush(xpar_file * f) { (void) f; return 0; }
int xpar_fsync(xpar_file * f) { (void) f; return 0; }

i64 xpar_size(xpar_file * f) {
  long len = filelength(f->fd);
  return len < 0 ? -1 : (i64) len;
}

bool xpar_is_seekable(xpar_file * f) {
  return lseek(f->fd, 0, SEEK_CUR) != (off_t) -1;
}
bool xpar_is_tty(xpar_file * f) { return f->kind == 1; }
bool xpar_eof  (xpar_file * f) { return f->at_eof; }
int  xpar_error(xpar_file * f) { return f->last_errno; }

/*  -----------------------------------------------------------------------
  Safe helpers (FATAL on error)  */

sz xpar_xread(xpar_file * f, void * p, sz n) {
  sz got = xpar_read(f, p, n);
  if (f->last_errno) FATAL_PERROR("read");
  return got;
}
void xpar_xwrite(xpar_file * f, const void * p, sz n) {
  if (xpar_write(f, p, n) != n) FATAL_PERROR("write");
}
void xpar_xclose(xpar_file * f) {
  if (!f) return;
  /*  DOS writes are synchronous; no fsync equivalent needed.  */
  if (f->owned) {
    if (close(f->fd) < 0) FATAL_PERROR("close");
    free(f);
  }
}
void xpar_notty(xpar_file * f) {
  if (f->kind == 1)
    FATAL("Refusing to read/write binary data from/to a terminal.");
}

/*  -----------------------------------------------------------------------
  Filesystem  */

int xpar_stat_path(const char * path, xpar_stat_t * out) {
  /*  _dos_findfirst via int21h AH=4Eh (or LFN AH=4Eh under Windows):
      one syscall yields size + attribute byte. Attribute bit 4 = dir.  */
  struct ffblk fb;
  if (findfirst(path, &fb, 0x37) != 0) return -1;
  out->size       = (u64) (unsigned long) fb.ff_fsize;
  out->is_dir     = (fb.ff_attrib & 0x10) != 0;
  out->is_regular = !out->is_dir
                 && !(fb.ff_attrib & 0x08)   /*  volume label  */;
  return 0;
}

int xpar_remove(const char * path) { return unlink(path); }

/*  Case-insensitive ASCII string equality for canonical DOS paths.  */
static int dos_strcasecmp(const char * a, const char * b) {
  while (*a && *b) {
    int ca = *a, cb = *b;
    if (ca >= 'a' && ca <= 'z') ca -= 32;
    if (cb >= 'a' && cb <= 'z') cb -= 32;
    if (ca != cb) return ca - cb;
    a++; b++;
  }
  return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

/*  int21h AH=60h (or LFN AX=7160h CL=2) -- TRUENAME. Returns the DOS
    canonical path (all-caps 8.3 or LFN). We use the DPMI transfer
    buffer: input path at __tb, output at __tb+128.  */
static int dos_truename(const char * path, char out[260]) {
  sz plen = xpar_strlen(path);
  if (plen >= 127) return -1;
  /*  Copy path into the transfer buffer.  */
  for (sz i = 0; i <= plen; i++)
    _farpokeb(_dos_ds, (unsigned long) __tb + i, (u8) path[i]);
  __dpmi_regs r;
  xpar_memset(&r, 0, sizeof(r));
  r.x.ax = 0x7160;  r.x.cx = 0x0002;     /*  LFN long-name canonical  */
  r.x.ds = __tb >> 4;  r.x.si = __tb & 0xF;
  r.x.es = __tb >> 4;  r.x.di = (__tb & 0xF) + 128;
  if (__dpmi_int(0x21, &r) < 0 || (r.x.flags & 1)) {
    /*  Fall back to short-name TRUENAME.  */
    xpar_memset(&r, 0, sizeof(r));
    r.h.ah = 0x60;
    r.x.ds = __tb >> 4;  r.x.si = __tb & 0xF;
    r.x.es = __tb >> 4;  r.x.di = (__tb & 0xF) + 128;
    if (__dpmi_int(0x21, &r) < 0 || (r.x.flags & 1)) return -1;
  }
  /*  Pull up to 259 bytes out of the transfer buffer.  */
  int n = 0;
  while (n < 259) {
    u8 c = _farpeekb(_dos_ds, (unsigned long) __tb + 128 + n);
    out[n++] = (char) c;
    if (!c) break;
  }
  out[259] = 0;
  return 0;
}

int xpar_same_file(const char * a, const char * b) {
  /*  DOS has no inodes; compare canonical paths (case-insensitive).  */
  char ca[260], cb[260];
  if (dos_truename(a, ca) != 0 || dos_truename(b, cb) != 0) return -1;
  return dos_strcasecmp(ca, cb) == 0 ? 1 : 0;
}

/*  No mmap on DJGPP; callers treat NULL as "declined".  */

xpar_mmap xpar_map(const char * path) {
  (void) path;
  xpar_mmap m = { NULL, 0 };
  return m;
}
void xpar_unmap(xpar_mmap * m) { m->map = NULL; m->size = 0; }

/*  -----------------------------------------------------------------------
  Allocation (libc heap: sbrk-backed, not worth rolling)  */

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

char * xpar_strdup(const char * s) {
  sz n = xpar_strlen(s) + 1;
  char * c = xpar_alloc_raw(n);
  xpar_memcpy(c, s, n);
  return c;
}
char * xpar_strndup(const char * s, sz n) {
  sz len = 0;
  while (len < n && s[len]) len++;
  char * c = xpar_alloc_raw(len + 1);
  xpar_memcpy(c, s, len);
  c[len] = '\0';
  return c;
}

/*  -----------------------------------------------------------------------
  Numeric parsing: decimal only  */

int xpar_parse_i64(const char * s, i64 * out) {
  if (!s || !*s) return -1;
  int neg = 0;
  if (*s == '-') { neg = 1; s++; } else if (*s == '+') { s++; }
  if (!*s) return -1;
  u64 v = 0;
  while (*s) {
    if (*s < '0' || *s > '9') return -1;
    u64 nv = v * 10 + (u64)(*s - '0');
    if (nv < v) return -1;
    v = nv; s++;
  }
  if (neg) {
    if (v > (u64) INT64_MAX + 1) return -1;
    *out = v == (u64) INT64_MAX + 1 ? INT64_MIN : -(i64) v;
  } else {
    if (v > (u64) INT64_MAX) return -1;
    *out = (i64) v;
  }
  return 0;
}
int xpar_parse_u64(const char * s, u64 * out) {
  if (!s || !*s) return -1;
  if (*s == '+') s++;
  if (!*s) return -1;
  u64 v = 0;
  while (*s) {
    if (*s < '0' || *s > '9') return -1;
    u64 nv = v * 10 + (u64)(*s - '0');
    if (nv < v) return -1;
    v = nv; s++;
  }
  *out = v;
  return 0;
}

/*  -----------------------------------------------------------------------
  Mini-printf (subset; mirrors host-win32.c)  */

typedef struct { char * buf; sz cap; sz pos; } fmt_ctx;

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
  char tmp[32]; int n = 0;
  const char * digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
  if (v == 0 && prec == 0) n = 0;
  else do { tmp[n++] = digits[v % base]; v /= base; } while (v);
  int len = n;
  int pad_zero = prec > len ? prec - len : 0;
  int total = len + pad_zero;
  int pad_sp = width > total ? width - total : 0;
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
  if (v < 0) { sign = '-'; uv = (u64)(-(v+1)) + 1; }
  else if (flags & F_PLUS)  { sign = '+'; uv = (u64) v; }
  else if (flags & F_SPACE) { sign = ' '; uv = (u64) v; }
  else { uv = (u64) v; }
  if (sign) {
    if (width > 0 && !(flags & F_MINUS) && !(flags & F_ZERO)) {
      char tmp[32]; int n = 0;
      if (uv == 0 && prec == 0) n = 0;
      else { u64 x = uv;
        do { tmp[n++] = '0' + (x % 10); x /= 10; } while (x); }
      int len = n;
      int pad_zero = prec > len ? prec - len : 0;
      int total = 1 + len + pad_zero;
      int pad_sp = width > total ? width - total : 0;
      emit_pad(c, pad_sp, ' ');
      emit_c(c, sign);
      emit_pad(c, pad_zero, '0');
      while (n) emit_c(c, tmp[--n]);
      return;
    } else {
      emit_c(c, sign);
      width = width > 0 ? width - 1 : 0;
    }
  }
  emit_uint(c, uv, 10, 0, width, prec, flags);
}

static void emit_double(fmt_ctx * c, double v,
                        int width, int prec, int flags) {
  if (prec < 0) prec = 6;
  char sign = 0;
  if (v < 0)                { sign = '-'; v = -v; }
  else if (flags & F_PLUS)  { sign = '+'; }
  else if (flags & F_SPACE) { sign = ' '; }
  u64 ip = (u64) v;
  double frac = v - (double) ip;
  u64 mult = 1;
  for (int i = 0; i < prec; i++) mult *= 10;
  u64 fp = (u64)(frac * (double) mult + 0.5);
  if (fp >= mult) { ip++; fp -= mult; }
  char ibuf[24]; int in = 0;
  if (ip == 0) ibuf[in++] = '0';
  else { u64 x = ip; while (x) { ibuf[in++] = '0' + (x % 10); x /= 10; } }
  int total = in + (prec > 0 ? (1 + prec) : 0) + (sign ? 1 : 0);
  int pad = width > total ? width - total : 0;
  if (!(flags & F_MINUS) && !(flags & F_ZERO)) emit_pad(c, pad, ' ');
  if (sign) emit_c(c, sign);
  if (!(flags & F_MINUS) && (flags & F_ZERO)) emit_pad(c, pad, '0');
  while (in) emit_c(c, ibuf[--in]);
  if (prec > 0) {
    emit_c(c, '.');
    char fbuf[24]; int fn = 0;
    u64 x = fp;
    for (int i = 0; i < prec; i++) { fbuf[fn++] = '0' + (x % 10); x /= 10; }
    while (fn) emit_c(c, fbuf[--fn]);
  }
  if (flags & F_MINUS) emit_pad(c, pad, ' ');
}

int xpar_vsnprintf(char * buf, sz cap, const char * fmt, va_list ap) {
  fmt_ctx c = { buf, cap, 0 };
  while (*fmt) {
    if (*fmt != '%') { emit_c(&c, *fmt++); continue; }
    fmt++;
    int flags = 0;
    for (;; fmt++) {
      if (*fmt == '-') flags |= F_MINUS;
      else if (*fmt == '+') flags |= F_PLUS;
      else if (*fmt == ' ') flags |= F_SPACE;
      else if (*fmt == '0') flags |= F_ZERO;
      else if (*fmt == '#') flags |= F_HASH;
      else break;
    }
    int width = 0;
    if (*fmt == '*') { width = va_arg(ap, int); fmt++; }
    else while (*fmt >= '0' && *fmt <= '9')
      { width = width * 10 + (*fmt - '0'); fmt++; }
    int prec = -1;
    if (*fmt == '.') {
      fmt++; prec = 0;
      if (*fmt == '*') { prec = va_arg(ap, int); fmt++; }
      else while (*fmt >= '0' && *fmt <= '9')
        { prec = prec * 10 + (*fmt - '0'); fmt++; }
    }
    int longness = 0;
    if (*fmt == 'z') { longness = 3; fmt++; }
    else if (*fmt == 'l') {
      fmt++;
      if (*fmt == 'l') { longness = 2; fmt++; }
      else longness = 1;
    } else if (*fmt == 'h') {
      fmt++;
      if (*fmt == 'h') fmt++;
    }
    char spec = *fmt; if (spec) fmt++;
    switch (spec) {
      case 'd': case 'i': {
        i64 v;
        if (longness == 2)      v = va_arg(ap, long long);
        else if (longness == 1) v = va_arg(ap, long);
        else if (longness == 3) v = (i64) va_arg(ap, ptrdiff_t);
        else                    v = va_arg(ap, int);
        emit_int(&c, v, width, prec, flags);
        break;
      }
      case 'u': {
        u64 v;
        if (longness == 2)      v = va_arg(ap, unsigned long long);
        else if (longness == 1) v = va_arg(ap, unsigned long);
        else if (longness == 3) v = (u64) va_arg(ap, sz);
        else                    v = va_arg(ap, unsigned);
        emit_uint(&c, v, 10, 0, width, prec, flags);
        break;
      }
      case 'x': case 'X': {
        u64 v;
        if (longness == 2)      v = va_arg(ap, unsigned long long);
        else if (longness == 1) v = va_arg(ap, unsigned long);
        else if (longness == 3) v = (u64) va_arg(ap, sz);
        else                    v = va_arg(ap, unsigned);
        emit_uint(&c, v, 16, spec == 'X', width, prec, flags);
        break;
      }
      case 'p': {
        void * p = va_arg(ap, void *);
        emit_str(&c, "0x", 2);
        emit_uint(&c, (u64)(uintptr_t) p, 16, 0, 8, -1, F_ZERO);
        break;
      }
      case 'c': {
        int ch = va_arg(ap, int);
        int pad = width > 1 ? width - 1 : 0;
        if (!(flags & F_MINUS)) emit_pad(&c, pad, ' ');
        emit_c(&c, (char) ch);
        if (flags & F_MINUS) emit_pad(&c, pad, ' ');
        break;
      }
      case 's': {
        const char * s = va_arg(ap, const char *);
        if (!s) s = "(null)";
        sz slen = 0; while (s[slen] && (prec < 0 || slen < (sz) prec)) slen++;
        int pad = width > (int) slen ? width - (int) slen : 0;
        if (!(flags & F_MINUS)) emit_pad(&c, pad, ' ');
        emit_str(&c, s, slen);
        if (flags & F_MINUS) emit_pad(&c, pad, ' ');
        break;
      }
      case 'f': {
        double v = va_arg(ap, double);
        emit_double(&c, v, width, prec, flags);
        break;
      }
      case '%': emit_c(&c, '%'); break;
      default: emit_c(&c, '%'); if (spec) emit_c(&c, spec); break;
    }
  }
  if (c.buf && c.cap > 0) {
    if (c.pos < c.cap) c.buf[c.pos] = '\0';
    else c.buf[c.cap - 1] = '\0';
  }
  return (int) c.pos;
}

int xpar_snprintf(char * buf, sz cap, const char * fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = xpar_vsnprintf(buf, cap, fmt, ap);
  va_end(ap);
  return r;
}

int xpar_asprintf(char ** out, const char * fmt, ...) {
  va_list ap; va_start(ap, fmt);
  va_list ap2; va_copy(ap2, ap);
  int n = xpar_vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (n < 0) { va_end(ap2); *out = NULL; return -1; }
  *out = xpar_alloc_raw((sz) n + 1);
  xpar_vsnprintf(*out, (sz) n + 1, fmt, ap2);
  va_end(ap2);
  return n;
}

static void write_raw(xpar_file * f, const char * s, sz n) {
  sz off = 0;
  while (off < n) {
    sz want = n - off;
    if (want > 0x40000000u) want = 0x40000000u;
    int w = write(f->fd, s + off, want);
    if (w <= 0) break;
    off += (sz) w;
  }
}

int xpar_vfprintf(xpar_file * f, const char * fmt, va_list ap) {
  char stack[1024];
  va_list ap2; va_copy(ap2, ap);
  int n = xpar_vsnprintf(stack, sizeof(stack), fmt, ap);
  if (n < (int) sizeof(stack)) {
    write_raw(f, stack, (sz) n);
    va_end(ap2);
    return n;
  }
  char * big = xpar_alloc_raw((sz) n + 1);
  xpar_vsnprintf(big, (sz) n + 1, fmt, ap2);
  va_end(ap2);
  write_raw(f, big, (sz) n);
  xpar_free(big);
  return n;
}

int xpar_fprintf(xpar_file * f, const char * fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = xpar_vfprintf(f, fmt, ap);
  va_end(ap);
  return r;
}

int xpar_fputs(const char * s, xpar_file * f) {
  sz n = xpar_strlen(s);
  write_raw(f, s, n);
  return (int) n;
}

/*  -----------------------------------------------------------------------
  Process, errors, time  */

__attribute__((noreturn)) void xpar_exit(int code) {
  /*  int21h AH=4Ch: terminate with return code.  */
  __dpmi_regs r;
  xpar_memset(&r, 0, sizeof(r));
  r.h.ah = 0x4C;  r.h.al = (u8) code;
  __dpmi_int(0x21, &r);
  /*  Should never return. Fall-back in case DPMI host misbehaves.  */
  for (;;);
}

/*  Compact errno->message table. Covers the codes DJGPP's int21h wrappers
    actually produce; anything else falls through to a generic string.  */
static const struct { int n; const char * s; } err_tab[] = {
  { EACCES,       "permission denied" },
  { EEXIST,       "file exists" },
  { ENOENT,       "no such file or directory" },
  { EBADF,        "bad file descriptor" },
  { EINVAL,       "invalid argument" },
  { EIO,          "I/O error" },
  { ENOMEM,       "out of memory" },
  { ENOSPC,       "no space left on device" },
  { ENFILE,       "too many open files in system" },
  { EMFILE,       "too many open files" },
  { ENOTDIR,      "not a directory" },
  { EISDIR,       "is a directory" },
  { ENAMETOOLONG, "filename too long" },
  { EOVERFLOW,    "value too large" },
  { EXDEV,        "cross-device link" },
};

static char err_gen[32];

const char * xpar_strerror(int err) {
  for (sz i = 0; i < sizeof(err_tab)/sizeof(err_tab[0]); i++)
    if (err_tab[i].n == err) return err_tab[i].s;
  xpar_snprintf(err_gen, sizeof(err_gen), "error %d", err);
  return err_gen;
}
int xpar_errno(void) { return errno; }

/*  8254 PIT channel 0 runs at 1,193,181.818... Hz in mode 3 with N=65536,
    giving a 54.9254 ms BIOS-tick period. In mode 3 the counter decrements
    by 2 per CLK and descends N -> 0 TWICE per BIOS tick; OUT is high on
    the first half and low on the second. The read-back command latches
    the counter and the status byte (bit 7 = OUT) together, which is what
    disambiguates the two halves into a single monotonic timeline.
    Resolution ~1.68 us; monotonic until midnight wrap on 0040:006C.  */
u64 xpar_usec_now(void) {
  u32 bios0, bios1, count, out_hi, elapsed;
  do {
    bios0 = _farpeekl(_dos_ds, 0x46CUL);
    outportb(0x43, 0xC2);                  /*  read-back ch0: status+count  */
    u8 status = inportb(0x40);
    count  = (u32) inportb(0x40);
    count |= (u32) inportb(0x40) << 8;
    out_hi = (status >> 7) & 1;
    bios1 = _farpeekl(_dos_ds, 0x46CUL);
  } while (bios0 != bios1);                /*  retry on IRQ0 during read  */
  /*  Within a period: first half (OUT=high) elapsed = (N-count)/2 in
      [0,32768]; second half (OUT=low) elapsed = 32768 + (N-count)/2.  */
  elapsed = (65536u - count) >> 1;
  if (!out_hi) elapsed += 32768u;
  u64 pit = (u64) bios1 * 65536ULL + (u64) elapsed;
  return (pit * 1000000ULL) / 1193182ULL;
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
