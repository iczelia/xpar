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

/*  io_uring-backed batched write pipeline.

    The entire API is hidden behind XPAR_HAS_LIBURING so call sites for
    non-Linux / pre-5.7 builds never see it. Every caller guards its use
    with `#ifdef XPAR_HAS_LIBURING` blocks that fall through to the
    classic synchronous xpar_xwrite path when io_uring is absent.  */

#ifndef XPAR_IO_URING_HOST_H
#define XPAR_IO_URING_HOST_H

#include "common.h"

#ifdef XPAR_HAS_LIBURING

typedef struct xpar_iogroup xpar_iogroup;

/*  Request `depth` concurrent SQEs. Returns NULL if the running kernel
    refuses io_uring_setup or lacks FAST_POLL (5.7+), or when env
    XPAR_URING=0 forces the sync path.  */
xpar_iogroup * xpar_iogroup_new(unsigned depth);

/*  Register an xpar_file's fd with the group. Flushes any pending
    stdio buffer so the fd is uring-owned until _free. Returns the
    fid used by subsequent enqueue / fsync calls; < 0 on error.  */
int xpar_iogroup_register_file(xpar_iogroup *, xpar_file *);

/*  Enqueue a write at `off` of `len` bytes from `buf` to the file
    identified by `fid`. `tag` is an opaque identifier surfaced in
    FATAL messages on completion error (e.g. shard index or block
    seq). The buffer must stay live until drain() returns.  */
void xpar_iogroup_enqueue_write(xpar_iogroup *,
                                int fid, const void * buf,
                                u64 off, sz len, u64 tag);

/*  Kick pending SQEs without waiting. Optional -- drain() submits too.  */
void xpar_iogroup_submit(xpar_iogroup *);

/*  Wait for every outstanding CQE. FATALs on partial or error
    completion, naming the first offending tag.  */
void xpar_iogroup_drain(xpar_iogroup *);

/*  Enqueue IORING_OP_FSYNC (DATASYNC) for a registered fd. Drains
    pending writes first so the fsync follows them.  */
void xpar_iogroup_fsync(xpar_iogroup *, int fid);

/*  Drain then destroy. Callers must close the registered files
    themselves after _free.  */
void xpar_iogroup_free(xpar_iogroup *);

/*  Batch arena size for streaming encoders (in per-block records).
    Defaults to 4096, overridable via env XPAR_URING_BATCH.  */
unsigned xpar_iogroup_batch_records(void);

#endif  /*  XPAR_HAS_LIBURING  */
#endif
