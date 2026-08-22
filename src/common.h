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

/*  Shared tools for every translation unit: short type names, the
    fatal-error macros, the loop macros, and the small helpers that would
    otherwise be rewritten per module.  */

#ifndef XPAR_COMMON_H
#define XPAR_COMMON_H

#include "platform/port.h"

/*  Fatal errors.  */
#define FATAL_CODE(code, fmt, ...)                                            \
  do {                                                                        \
    xpar_fprintf(xpar_stderr, "xpar: " fmt "\n", ##__VA_ARGS__);              \
    xpar_exit(code);                                                          \
  } while (0)

#define FATAL(fmt, ...)        FATAL_CODE(XPAR_EXIT_USAGE, fmt, ##__VA_ARGS__)
#define FATAL_IO(fmt, ...)     FATAL_CODE(XPAR_EXIT_IO, fmt, ##__VA_ARGS__)
#define FATAL_FORMAT(fmt, ...) FATAL_CODE(XPAR_EXIT_NOTFOUND, fmt,            \
                                          ##__VA_ARGS__)

#define FATAL_UNLESS(fmt, cond, ...)                                          \
  do {                                                                        \
    if (!(cond)) FATAL(fmt, ##__VA_ARGS__);                                   \
  } while (0)

#define FATAL_PERROR(who)                                                     \
  do {                                                                        \
    xpar_fprintf(xpar_stderr, "xpar: %s: %s\n", (who),                        \
                 xpar_strerror(xpar_errno()));                                \
    xpar_exit(XPAR_EXIT_IO);                                                  \
  } while (0)

#define xpar_assert(x)                                                        \
  do {                                                                        \
    if (!(x))                                                                 \
      FATAL_CODE(XPAR_EXIT_INTERNAL,                                          \
                 "internal: assertion failed: %s (%s:%d)", #x, __FILE__,      \
                 __LINE__);                                                   \
  } while (0)

/*  Loop macros.  */
#define For(t, v, n, ...)  for (t v = 0; v < (n); v++) { __VA_ARGS__; }

#define Fi(n, ...)  For(int, i, n, __VA_ARGS__)
#define Fj(n, ...)  For(int, j, n, __VA_ARGS__)
#define Fk(n, ...)  For(int, k, n, __VA_ARGS__)
#define Fi0(n, s, ...)  for (int i = (s); i < (n); i++) { __VA_ARGS__; }
#define Fj0(n, s, ...)  for (int j = (s); j < (n); j++) { __VA_ARGS__; }
#define Fk0(n, s, ...)  for (int k = (s); k < (n); k++) { __VA_ARGS__; }

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/*  Element count of a true array. Never pass a pointer.  */
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/*  Message pluralisation: the suffix a regular noun takes for a count.  */
#define PLURAL(n) ((n) == 1 ? "" : "s")

#define XPAR_EXIT_OK           0  /*  Success; for verify, no damage.  */
#define XPAR_EXIT_REPAIRABLE   1  /*  verify: damage found, repairable.
                                      repair --exit-on-change: changed.  */
#define XPAR_EXIT_UNREPAIRABLE 2  /*  Damage beyond the available recovery.  */
#define XPAR_EXIT_NOTFOUND     3  /*  Not found, not an xpar set, or an
                                      unsupported format version.  */
#define XPAR_EXIT_USAGE        4  /*  Usage error.  */
#define XPAR_EXIT_IO           5  /*  I/O error.  */
#define XPAR_EXIT_AUTH         6  /*  Missing key, wrong key, MAC mismatch.  */
#define XPAR_EXIT_NOPLAN       7  /*  No plan fits -m, or out of memory.  */
#define XPAR_EXIT_INTERNAL     8  /*  A bug; should never occur.  */

/*  Round up to a multiple of a power of two. `a` must be a power of two.  */
static inline u64 xpar_align_up(u64 v, u64 a) { return (v + a - 1) & ~(a - 1); }

/*  ceil(a / b) without overflowing on a + b.  */
static inline u64 xpar_ceil_div(u64 a, u64 b) {
  return a / b + (a % b != 0);
}

/*  Smallest power of two >= v, for v <= 2^63. Returns 1 for v == 0, which is
    what NextPow2 means everywhere it is used here.  */
static inline u64 xpar_next_pow2(u64 v) {
  if (v <= 1) return 1;
  v--;
  v |= v >> 1;   v |= v >> 2;   v |= v >> 4;
  v |= v >> 8;   v |= v >> 16;  v |= v >> 32;
  return v + 1;
}

static inline bool xpar_is_pow2(u64 v) { return v && !(v & (v - 1)); }

static inline int xpar_log2_floor(u64 v) {
  int r = 0;
  while (v >>= 1) r++;
  return r;
}

static inline u16 xpar_rd16(const u8 * p) {
  return (u16) ((u16) p[0] | ((u16) p[1] << 8));
}

static inline u32 xpar_rd32(const u8 * p) {
  return (u32) p[0] | ((u32) p[1] << 8) | ((u32) p[2] << 16) |
         ((u32) p[3] << 24);
}

static inline u64 xpar_rd64(const u8 * p) {
  return (u64) xpar_rd32(p) | ((u64) xpar_rd32(p + 4) << 32);
}

static inline void xpar_wr16(u8 * p, u16 v) {
  p[0] = (u8) v;  p[1] = (u8) (v >> 8);
}

static inline void xpar_wr32(u8 * p, u32 v) {
  p[0] = (u8) v;         p[1] = (u8) (v >> 8);
  p[2] = (u8) (v >> 16); p[3] = (u8) (v >> 24);
}

static inline void xpar_wr64(u8 * p, u64 v) {
  xpar_wr32(p, (u32) v);  xpar_wr32(p + 4, (u32) (v >> 32));
}

/*  Lower-case hexadecimal text for `n` bytes, NUL-terminated; the buffer
    holds 2n + 1.  */
static inline void xpar_hex(char * out, const u8 * p, sz n) {
  static const char d[] = "0123456789abcdef";
  for (sz i = 0; i < n; i++) {
    out[2 * i]     = d[p[i] >> 4];
    out[2 * i + 1] = d[p[i] & 15];
  }
  out[2 * n] = 0;
}

/*  Whether `pfx`, folded to lower case, is a hexadecimal prefix of the
    `n` bytes at `id`. An empty prefix matches; a longer one cannot.  */
static inline bool xpar_hex_prefix(const u8 * id, sz n, const char * pfx) {
  static const char d[] = "0123456789abcdef";
  for (sz i = 0; pfx[i]; i++) {
    char c = pfx[i];
    if (i >= 2 * n) return false;
    if (c >= 'A' && c <= 'F') c = (char) (c - 'A' + 'a');
    if (c != d[i & 1 ? id[i / 2] & 15 : id[i / 2] >> 4]) return false;
  }
  return true;
}

/*  Decimal digits in `v`, and so the field width that prints it: 1 for
    zero, which is what every name-padding caller wants.  */
static inline int xpar_digits10(u64 v) {
  int d = 1;
  while (v >= 10) { v /= 10;  d++; }
  return d;
}

/*  Constant-time comparison.  */
static inline bool xpar_ct_equal(const void * a, const void * b, sz n) {
  const u8 * x = (const u8 *) a, * y = (const u8 *) b;
  u8 acc = 0;
  for (sz i = 0; i < n; i++) acc |= (u8) (x[i] ^ y[i]);
  return acc == 0;
}

/*  Keep key erasure observable to the abstract machine.  */
static inline void xpar_secure_zero(void * p, sz n) {
  volatile u8 * q = (volatile u8 *) p;
  while (n--) *q++ = 0;
}

/*  Progress reporting.  */

typedef struct {
  bool enabled;
  u64 total_bytes;     /*  0 when the total is unknown, e.g. a pipe.  */
  u64 bytes_done;
  u64 bytes_at_emit;   /*  bytes_done at the last emit; suppresses a dup.  */
  u64 start_usec;
  u64 last_usec;
  u64 since_check;     /*  Bytes since the clock was last read.  */
  const char * op;     /*  "Creating", "Verifying", "Repairing", ...  */
} xpar_progress_t;

void xpar_progress_init(xpar_progress_t *, bool on, u64 total,
                        const char * op);
void xpar_progress_tick(xpar_progress_t *, u64 bytes);
void xpar_progress_end (xpar_progress_t *);

#endif
