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

/*  A freestanding vsnprintf for the hosts whose C library has none worth
    trusting: the Windows CRT truncates without saying how much it wanted
    and predates %zu, and the DOS one is no better. Only the conversions
    xpar itself emits are implemented.  */

#include "common.h"

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
  len = n;
  /*  POSIX: an explicit precision on an integer conversion voids '0'.  */
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
  /*  Negating INT64_MIN is undefined, so the magnitude is built as
      (-(v+1))+1, which stays in range at every step.  */
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
  total = in + (prec > 0 ? 1 + prec : 0) + (sign != 0);
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
      if (*fmt == 'h') fmt++;   /*  default promotions make both int  */
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
        emit_uint(&c, (u64) (uintptr_t) p, 16, 0,
                  (int) (sizeof(void *) * 2), -1, F_ZERO);
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
