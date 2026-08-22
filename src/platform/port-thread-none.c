/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License only.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.  */

/*  Serial worker-pool implementation for DJGPP and --disable-threads.  */

#include "common.h"
#include "port-thread.h"

int xpar_cpu_count(void) { return 1; }
int xpar_core_count(void) { return 1; }

struct xpar_pool { int nthreads; };

xpar_pool * xpar_pool_create(int threads) {
  struct xpar_pool * p = xpar_alloc_raw(sizeof(*p));
  (void) threads;
  p->nthreads = 1;
  return p;
}

int xpar_pool_threads(const xpar_pool * p) { return p ? p->nthreads : 1; }

void xpar_pool_run(xpar_pool * p, sz n, xpar_work_fn fn, void * ctx) {
  sz i;
  (void) p;
  for (i = 0; i < n; i++) fn(i, ctx);
}

void xpar_pool_destroy(xpar_pool * p) { xpar_free(p); }
