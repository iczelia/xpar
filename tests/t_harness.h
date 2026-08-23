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

/* Shared assertion harness. Tests define xpar_main; failures accumulate. */

#ifndef XPAR_T_HARNESS_H
#define XPAR_T_HARNESS_H

#include "common.h"

/* Scale iteration counts with XPAR_TEST_LEVEL. */
static int xt_level = 1;
static u64 xt_checks, xt_failures;
static const char * xt_section = "(none)";

static inline void xt_level_from_env(const char * v) {
  if (!v) return;
  if (!xpar_strcmp(v, "quick")) xt_level = 1;
  else if (!xpar_strcmp(v, "full")) xt_level = 4;
  else if (!xpar_strcmp(v, "torture")) xt_level = 16;
}

static inline u32 xt_scale(u32 quick) {
  u64 n = (u64) quick * (u64) xt_level;
  return n > 0xFFFFFFFFu ? 0xFFFFFFFFu : (u32) n;
}

/* Optional tracing helps diagnose hangs without cluttering normal logs. */
static int xt_tracing;

static inline void xt_trace_from_env(const char * v) {
  xt_tracing = v && *v && xpar_strcmp(v, "0");
}

static inline void xt_section_begin(const char * name) {
  xt_section = name;
  if (xt_tracing)
    xpar_fprintf(xpar_stderr, "  == %s (%" PRIu64 " us)\n", name,
                 xpar_usec_now());
}

static inline void xt_trace(const char * fmt, ...) XPAR_PRINTF(1, 2);

static inline void xt_trace(const char * fmt, ...) {
  va_list ap;
  if (!xt_tracing) return;
  xpar_fprintf(xpar_stderr, "     ");
  va_start(ap, fmt);
  xpar_vfprintf(xpar_stderr, fmt, ap);
  va_end(ap);
  xpar_fprintf(xpar_stderr, " (%" PRIu64 " us)\n", xpar_usec_now());
}

static inline void xt_report(bool ok, const char * fmt, ...) XPAR_PRINTF(2, 3);

static inline void xt_report(bool ok, const char * fmt, ...) {
  va_list ap;
  xt_checks++;
  if (ok) return;
  xt_failures++;
  xpar_fprintf(xpar_stderr, "FAIL [%s] ", xt_section);
  va_start(ap, fmt);
  xpar_vfprintf(xpar_stderr, fmt, ap);
  va_end(ap);
  xpar_fprintf(xpar_stderr, "\n");
}

#define CHECK(cond, ...)  xt_report((cond), __VA_ARGS__)

#define CHECK_U64(got, want, ...)                                             \
  do {                                                                        \
    u64 xt_g = (u64) (got), xt_w = (u64) (want);                              \
    if (xt_g != xt_w) {                                                       \
      xt_report(false, __VA_ARGS__);                                          \
      xpar_fprintf(xpar_stderr, "       got %" PRIu64 ", want %" PRIu64 "\n", \
                   xt_g, xt_w);                                               \
    } else xt_checks++;                                                       \
  } while (0)

/* Report the first differing byte. */
static inline bool xt_bytes_equal(const char * what, const u8 * got,
                                  const u8 * want, sz n) {
  sz i;
  for (i = 0; i < n; i++) {
    if (got[i] == want[i]) continue;
    xt_checks++;  xt_failures++;
    xpar_fprintf(xpar_stderr,
                 "FAIL [%s] %s differs at byte %" PRIu64 " of %" PRIu64
                 ": got %02X, want %02X\n",
                 xt_section, what, (u64) i, (u64) n, got[i], want[i]);
    return false;
  }
  xt_checks++;
  return true;
}

static inline int xt_finish(const char * program) {
  xpar_fprintf(xpar_stderr, "%s: %" PRIu64 " checks, %" PRIu64 " failed\n",
               program, xt_checks, xt_failures);
  return xt_failures ? 1 : 0;
}

/* Match the benchmark KAT generator for reproducible inputs. */

typedef struct { u64 s; } xt_rng;

static inline void xt_seed(xt_rng * r, u64 seed) {
  r->s = seed ? seed : 0x9E3779B97F4A7C15ull;
}

static inline u32 xt_next(xt_rng * r) {
  r->s ^= r->s << 13;  r->s ^= r->s >> 7;  r->s ^= r->s << 17;
  return (u32) (r->s >> 32);
}

/*  Uniform on [0, n) by rejection: a modulo would skew the small ranges
    that erasure patterns are drawn from.  */
static inline u32 xt_below(xt_rng * r, u32 n) {
  u32 limit, v;
  if (n <= 1) return 0;
  limit = 0xFFFFFFFFu - (0xFFFFFFFFu % n) - 1;
  do v = xt_next(r); while (v > limit);
  return v % n;
}

static inline void xt_fill(xt_rng * r, u8 * p, sz n) {
  For(sz, i, n, p[i] = (u8) xt_next(r))
}

#endif
