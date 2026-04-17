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

/*  Threadpool + parallel-for: static [k*n/T,(k+1)*n/T) partition.
    g_gen generation counter prevents workers re-running the same job.  */

#include "common.h"

static int            g_n_threads = 0; /*  0 before init; live count after  */
/*  set via xpar_set_num_threads before first use  */
static int            g_pending_threads = 0;
static xpar_thread ** g_workers = NULL;
static xpar_mutex *   g_m = NULL;
static xpar_cond *    g_work_ready = NULL;
static xpar_cond *    g_work_done  = NULL;

static xpar_parfor_fn g_fn  = NULL;
static void *         g_ctx = NULL;
static sz             g_n   = 0;
static int            g_busy = 0;
static u64            g_gen  = 0;   /*  job-generation counter (monotonic)  */

struct worker_args { int k; };

static void worker_main(void * arg) {
  int k = ((struct worker_args *) arg)->k;
  xpar_free(arg);
  u64 last_gen = 0;
  for (;;) {
    xpar_mutex_lock(g_m);
    while (g_gen == last_gen) xpar_cond_wait(g_work_ready, g_m);
    last_gen = g_gen;
    xpar_parfor_fn fn = g_fn;
    void *         ctx = g_ctx;
    sz             n   = g_n;
    int            T   = g_n_threads;
    xpar_mutex_unlock(g_m);

    sz start = (sz) k * n / (sz) T;
    sz end   = (sz)(k + 1) * n / (sz) T;
    for (sz i = start; i < end; i++) fn(i, ctx);

    xpar_mutex_lock(g_m);
    if (--g_busy == 0) xpar_cond_broadcast(g_work_done);
    xpar_mutex_unlock(g_m);
  }
}

static void lazy_init(void) {
  if (g_n_threads != 0) return;
  int T = g_pending_threads > 0 ? g_pending_threads : xpar_cpu_count();
  if (T < 1) T = 1;
  g_m          = xpar_mutex_new();
  g_work_ready = xpar_cond_new();
  g_work_done  = xpar_cond_new();
  if (T > 1) {
    g_workers = xpar_alloc_raw((sz) T * sizeof(xpar_thread *));
    for (int k = 0; k < T; k++) {
      struct worker_args * a = xpar_alloc_raw(sizeof(*a));
      a->k = k;
      g_workers[k] = xpar_thread_start(worker_main, a);
    }
  }
  g_n_threads = T;
}

void xpar_set_num_threads(int n) {
  if (g_n_threads != 0) {
    /*  Already initialised; we don't currently support dynamic resize.  */
    return;
  }
  g_pending_threads = n > 0 ? n : 0;
}

void xpar_parallel_for(sz n, xpar_parfor_fn fn, void * ctx) {
  if (n == 0) return;
  if (g_n_threads == 0) lazy_init();
  if (n == 1 || g_n_threads == 1) {
    for (sz i = 0; i < n; i++) fn(i, ctx);
    return;
  }

  xpar_mutex_lock(g_m);
  /*  Drain any prior job before reusing the slots (reentrancy guard).  */
  while (g_busy > 0) xpar_cond_wait(g_work_done, g_m);

  g_fn  = fn;
  g_ctx = ctx;
  g_n   = n;
  g_busy = g_n_threads;
  g_gen++;
  xpar_cond_broadcast(g_work_ready);

  while (g_busy > 0) xpar_cond_wait(g_work_done, g_m);
  xpar_mutex_unlock(g_m);
}
