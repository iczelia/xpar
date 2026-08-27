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

/*  The throttled progress line.  */

#include "common.h"

/*  Bytes that must accumulate before the clock is consulted at all.  */
#define PROGRESS_CHECK_BYTES  ((u64) 1 << 20)
#define PROGRESS_EMIT_USEC    1000000ULL

static void progress_emit(const xpar_progress_t * p) {
  u64 elapsed = p->last_usec - p->start_usec;
  u64 done_mib = p->bytes_done >> 20;
  u64 rate;
  /*  Avoid division by zero on the first update.  */
  if (elapsed == 0) elapsed = 1;
  rate = p->bytes_done / elapsed;
  if (p->sink) {
    /*  Compute exact bytes/s without overflowing on large totals.  */
    u64 bps = rate * 1000000u +
              (p->bytes_done % elapsed) * 1000000u / elapsed;
    p->sink(p->sink_user, p->bytes_done, p->total_bytes, bps);
    return;
  }
  if (p->total_bytes) {
    unsigned pct = (unsigned) (p->bytes_done * 100 / p->total_bytes);
    /*  The caller may overshoot its estimate.  */
    if (pct > 100) pct = 100;
    xpar_fprintf(xpar_stderr, "%s: %u%% (%" PRIu64 " / %" PRIu64 " MiB) @ %"
                 PRIu64
                 " MB/s\n",
                 p->op, pct,
                 done_mib,
                 (p->total_bytes >> 20),
                 rate);
  } else {
    xpar_fprintf(xpar_stderr, "%s: %" PRIu64 " MiB @ %" PRIu64 " MB/s\n",
                 p->op,
                 done_mib,
                 rate);
  }
}

void xpar_progress_init(xpar_progress_t * p, bool on, u64 total,
                        const char * op) {
  p->enabled       = on;
  p->total_bytes   = total;
  p->bytes_done    = 0;
  p->bytes_at_emit = 0;
  p->since_check   = 0;
  p->op            = op;
  p->sink       = NULL;
  p->sink_user  = NULL;
  p->start_usec = on ? xpar_usec_now() : 0;
  p->last_usec  = p->start_usec;
}

void xpar_progress_sink(xpar_progress_t * p, xpar_progress_fn fn,
                        void * user) {
  p->sink = fn;  p->sink_user = user;
}

void xpar_progress_tick(xpar_progress_t * p, u64 bytes) {
  u64 now;
  if (!p->enabled) return;
  p->bytes_done  += bytes;
  p->since_check += bytes;
  if (p->since_check < PROGRESS_CHECK_BYTES) return;
  p->since_check = 0;
  now = xpar_usec_now();
  if (now - p->last_usec < PROGRESS_EMIT_USEC) return;
  p->last_usec     = now;
  p->bytes_at_emit = p->bytes_done;
  progress_emit(p);
}

void xpar_progress_end(xpar_progress_t * p) {
  if (!p->enabled || p->bytes_done == 0) return;
  /*  Do not repeat the last update.  */
  if (p->bytes_at_emit == p->bytes_done) return;
  p->last_usec     = xpar_usec_now();
  p->bytes_at_emit = p->bytes_done;
  progress_emit(p);
}
