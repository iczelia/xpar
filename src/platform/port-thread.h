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

/*  Online processors and physical cores, or 1.  */
int xpar_cpu_count(void);
int xpar_core_count(void);

#ifndef XPAR_POOL_TYPEDEF
#define XPAR_POOL_TYPEDEF
typedef struct xpar_pool xpar_pool;
#endif

/*  Called once per index; `ctx` is shared.  */
typedef void (*xpar_work_fn)(sz index, void * ctx);

/*  Worker pool; <= 0 uses xpar_cpu_count(), one runs inline.  */
xpar_pool * xpar_pool_create(int threads);
int         xpar_pool_threads(const xpar_pool * p);
void        xpar_pool_destroy(xpar_pool * p);

/*  Run fn(0..n-1). The caller participates. Not reentrant.  */
void xpar_pool_run(xpar_pool * p, sz n, xpar_work_fn fn, void * ctx);

#endif
