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

/*  Dynamic index-range worker pool.  */

#ifndef XPAR_PORT_THREAD_H
#define XPAR_PORT_THREAD_H

#include "port.h"

/*  Online processors, or 1 where the host will not say or has no threads.
    The planner uses it for the -j default.  */
int xpar_cpu_count(void);
int xpar_core_count(void);

typedef struct xpar_pool xpar_pool;

/*  Called once per index. `ctx` is shared across every call in the batch,
    so anything it points at that a callback writes must either be indexed
    by `index` or be synchronised by the callback itself.  */
typedef void (*xpar_work_fn)(sz index, void * ctx);

/*  A pool of `threads` workers; <= 0 means xpar_cpu_count(). A pool of one
    creates no threads at all and runs every batch on the calling thread,
    which is the only path a single-threaded host has.  */
xpar_pool * xpar_pool_create(int threads);
int         xpar_pool_threads(const xpar_pool * p);
void        xpar_pool_destroy(xpar_pool * p);

/*  Run fn(0..n-1) and return when all n calls have completed. The calling
    thread is one of the workers, so a pool of T threads runs T ways and not
    T+1. Not reentrant: a callback must not call xpar_pool_run on the pool
    it is running under, which would deadlock waiting for itself.  */
void xpar_pool_run(xpar_pool * p, sz n, xpar_work_fn fn, void * ctx);

#endif
