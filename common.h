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

#ifndef _COMMON_H_
#define _COMMON_H_

#include "platform.h"

#define FATAL(fmt, ...)                                                       \
  do {                                                                        \
    xpar_fprintf(xpar_stderr, fmt "\n", ##__VA_ARGS__);                       \
    xpar_exit(1);                                                             \
  } while (0)

#define FATAL_UNLESS(fmt, cond, ...)                                          \
  do {                                                                        \
    if (!(cond)) {                                                            \
      xpar_fprintf(xpar_stderr, fmt "\n", ##__VA_ARGS__);                     \
      xpar_exit(1);                                                           \
    }                                                                         \
  } while (0)

#define FATAL_PERROR(who)                                                     \
  do {                                                                        \
    xpar_fprintf(xpar_stderr, "%s: %s\n", (who), xpar_strerror(xpar_errno()));\
    xpar_exit(1);                                                             \
  } while (0)

#define xpar_assert(x)                                                        \
  do {                                                                        \
    if (!(x))                                                                 \
      FATAL("assertion failed: %s", #x);                                      \
  } while (0)

#define Fi(n, ...)                                                            \
  for (int i = 0; i < (n); i++) {                                             \
    __VA_ARGS__;                                                              \
  }
#define Fj(n, ...)                                                            \
  for (int j = 0; j < (n); j++) {                                             \
    __VA_ARGS__;                                                              \
  }
#define Fk(n, ...)                                                            \
  for (int k = 0; k < (n); k++) {                                             \
    __VA_ARGS__;                                                              \
  }
#define Fi0(n, s, ...)                                                        \
  for (int i = s; i < (n); i++) {                                             \
    __VA_ARGS__;                                                              \
  }
#define Fj0(n, s, ...)                                                        \
  for (int j = s; j < (n); j++) {                                             \
    __VA_ARGS__;                                                              \
  }
#define Fk0(n, s, ...)                                                        \
  for (int k = s; k < (n); k++) {                                             \
    __VA_ARGS__;                                                              \
  }

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*  Integrity algorithm identifiers stored verbatim in joint-mode headers.  */
#define INTEGRITY_CRC32C 0
#define INTEGRITY_BLAKE2B 1

/*  Throttled progress reporter writing to stderr. Callers bump byte
    counts on every loop iteration; the emitter fires at most once per
    second and skips the xpar_usec_now() syscall until >=1 MiB has been
    added, so tight per-lace loops don't pay for it.  */
typedef struct {
  bool enabled;
  u64 total_bytes;       /*  0 when the total is unknown (e.g. stdin).  */
  u64 bytes_done;
  u64 bytes_at_emit;     /*  bytes_done snapshot at last emit; dedup guard.  */
  u64 start_usec;
  u64 last_usec;
  u64 since_check;       /*  Bytes accumulated since last time query.  */
  const char * op;       /*  "Encoding", "Decoding", "Testing", ...  */
} xpar_progress_t;

static inline void xpar_progress_emit(const xpar_progress_t * p) {
  u64 elapsed_us = p->last_usec - p->start_usec;
  if (elapsed_us == 0) elapsed_us = 1;
  u64 done_mib = p->bytes_done >> 20;
  u64 rate_mbs = p->bytes_done / elapsed_us; /*  bytes/usec == MB/s  */
  if (p->total_bytes) {
    u64 tot_mib = p->total_bytes >> 20;
    u64 num = p->bytes_done * 100;
    unsigned pct = (unsigned)(num / p->total_bytes);
    if (pct > 100) pct = 100;
    xpar_fprintf(xpar_stderr,
      "%s: %u%% (%llu / %llu MiB) @ %llu MB/s\n",
      p->op, pct,
      (unsigned long long) done_mib,
      (unsigned long long) tot_mib,
      (unsigned long long) rate_mbs);
  } else {
    xpar_fprintf(xpar_stderr,
      "%s: %llu MiB @ %llu MB/s\n",
      p->op,
      (unsigned long long) done_mib,
      (unsigned long long) rate_mbs);
  }
}

static inline void xpar_progress_init(xpar_progress_t * p, bool enabled,
                                      u64 total, const char * op) {
  p->enabled = enabled;
  p->total_bytes = total;
  p->bytes_done = 0;
  p->bytes_at_emit = 0;
  p->since_check = 0;
  p->op = op;
  p->start_usec = enabled ? xpar_usec_now() : 0;
  p->last_usec = p->start_usec;
}

static inline void xpar_progress_tick(xpar_progress_t * p, u64 bytes) {
  if (!p->enabled) return;
  p->bytes_done += bytes;
  p->since_check += bytes;
  if (p->since_check < ((u64) 1 << 20)) return;
  p->since_check = 0;
  u64 now = xpar_usec_now();
  if (now - p->last_usec < 1000000) return;
  p->last_usec = now;
  p->bytes_at_emit = p->bytes_done;
  xpar_progress_emit(p);
}

static inline void xpar_progress_end(xpar_progress_t * p) {
  if (!p->enabled || !p->bytes_done) return;
  /*  Skip the final line if a tick already emitted at this byte count.  */
  if (p->bytes_at_emit == p->bytes_done) return;
  p->last_usec = xpar_usec_now();
  p->bytes_at_emit = p->bytes_done;
  xpar_progress_emit(p);
}

/*  Derive output name: POSIX appends suffix; DOS (8.3) replaces extension.  */
static inline char * xpar_derive_name(const char * input,
    const char * suffix) {
  sz ilen = xpar_strlen(input), slen = xpar_strlen(suffix);
#if defined(XPAR_DOS)
  sz cut = ilen;
  for (sz i = ilen; i > 0; i--) {
    char c = input[i - 1];
    if (c == '/' || c == '\\' || c == ':')
      break;
    if (c == '.') {
      cut = i - 1;
      break;
    }
  }
#else
  sz cut = ilen;
#endif
  char * r = xpar_alloc_raw(cut + slen + 1);
  xpar_memcpy(r, input, cut);
  xpar_memcpy(r + cut, suffix, slen);
  r[cut + slen] = '\0';
  return r;
}

#endif
