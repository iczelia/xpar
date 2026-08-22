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
  /*  A first emit inside the first microsecond would divide by zero; one
      microsecond of elapsed time is close enough for a rate that is about
      to be replaced a second later.  */
  if (elapsed == 0) elapsed = 1;
  rate = p->bytes_done / elapsed;
  if (p->total_bytes) {
    unsigned pct = (unsigned) (p->bytes_done * 100 / p->total_bytes);
    /*  The caller may overshoot its own estimate (a growing file, a
        re-read after a repair), and a percentage above 100 reads as a
        bug in xpar rather than as a moved target.  */
    if (pct > 100) pct = 100;
    xpar_fprintf(xpar_stderr, "%s: %u%% (%llu / %llu MiB) @ %llu MB/s\n",
                 p->op, pct,
                 (unsigned long long) done_mib,
                 (unsigned long long) (p->total_bytes >> 20),
                 (unsigned long long) rate);
  } else {
    xpar_fprintf(xpar_stderr, "%s: %llu MiB @ %llu MB/s\n",
                 p->op,
                 (unsigned long long) done_mib,
                 (unsigned long long) rate);
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
  /*  The clock is read once here even when disabled would allow skipping
      it, so that an enabled reporter always has a start point and the
      rate of the first line is not measured from zero.  */
  p->start_usec = on ? xpar_usec_now() : 0;
  p->last_usec  = p->start_usec;
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
  /*  Suppress the final line when a tick has already reported this exact
      byte count: a run shorter than the emit interval prints one line and
      a longer one does not repeat its last.  */
  if (p->bytes_at_emit == p->bytes_done) return;
  p->last_usec     = xpar_usec_now();
  p->bytes_at_emit = p->bytes_done;
  progress_emit(p);
}
