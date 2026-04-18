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

#include "io_uring_host.h"

/*  This translation unit is empty when XPAR_HAS_LIBURING is undefined;
    no symbols leak into non-Linux / liburing-less builds. Every caller
    gates its uring path with the same macro.  */

#ifdef XPAR_HAS_LIBURING

#define _GNU_SOURCE
#include <errno.h>
#include <liburing.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*  Per-SQE bookkeeping lives inside a pool referenced by SQE user_data.
    We need the expected completion length (to detect short writes) and
    the caller's tag (for the FATAL message).  */
typedef struct {
  sz   expect_len;
  u64  tag;
  int  is_fsync;
} uring_cookie;

struct xpar_iogroup {
  struct io_uring ring;
  unsigned depth;

  int *    fds;                /*  dynamic; grown via register_file  */
  int      n_files;
  int      cap_files;

  uring_cookie * cookies;      /*  depth entries, indexed by SQE user_data  */
  int          * free_ck;      /*  stack of free cookie indices  */
  int            n_free;

  unsigned in_flight;
  int      had_error;
  int      saved_errno;
  u64      saved_tag;
};

/*  Returns 1 if the environment override disables io_uring.  */
static int env_disabled(void) {
  const char * e = getenv("XPAR_URING");
  return e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N');
}

xpar_iogroup * xpar_iogroup_new(unsigned depth) {
  if (env_disabled()) return NULL;
  if (depth < 8)  depth = 8;
  if (depth > 1024) depth = 1024;

  xpar_iogroup * g = calloc(1, sizeof *g);
  if (!g) return NULL;

  struct io_uring_params params;
  memset(&params, 0, sizeof params);
  /*  IORING_SETUP_CLAMP silently caps depth to the kernel's max rather
      than failing with -EINVAL on the very large values some configs
      forbid.  */
  params.flags = IORING_SETUP_CLAMP;
  int rc = io_uring_queue_init_params(depth, &g->ring, &params);
  if (rc < 0) { free(g); return NULL; }

  /*  Require FAST_POLL (Linux 5.7+): without it, uring write latency on
      buffered file I/O regresses compared to vanilla write(2).  */
  if (!(params.features & IORING_FEAT_FAST_POLL)) {
    io_uring_queue_exit(&g->ring);
    free(g);
    return NULL;
  }

  g->depth   = depth;
  g->n_files = 0;
  g->cookies = calloc(depth, sizeof *g->cookies);
  g->free_ck = calloc(depth, sizeof *g->free_ck);
  if (!g->cookies || !g->free_ck) {
    free(g->cookies); free(g->free_ck);
    io_uring_queue_exit(&g->ring);
    free(g);
    return NULL;
  }
  for (unsigned i = 0; i < depth; i++)
    g->free_ck[i] = (int)(depth - 1 - i);
  g->n_free = (int) depth;
  return g;
}

int xpar_iogroup_register_file(xpar_iogroup * g, xpar_file * f) {
  if (!g || !f) return -1;
  if (g->n_files == g->cap_files) {
    int new_cap = g->cap_files ? g->cap_files * 2 : 16;
    int * p = realloc(g->fds, (size_t) new_cap * sizeof *p);
    if (!p) return -1;
    g->fds = p; g->cap_files = new_cap;
  }
  /*  Flush so no stdio buffer sits between us and the fd; after this
      point the fd is uring-owned until _free.  */
  if (xpar_file_flush_stdio(f) != 0) return -1;
  int fd = xpar_file_fd(f);
  if (fd < 0) return -1;
  g->fds[g->n_files] = fd;
  return g->n_files++;
}

static void reap_one(xpar_iogroup * g, struct io_uring_cqe * cqe) {
  uring_cookie * ck = &g->cookies[cqe->user_data];
  if (!g->had_error) {
    int res = cqe->res;
    if (res < 0) {
      g->had_error   = 1;
      g->saved_errno = -res;
      g->saved_tag   = ck->tag;
    } else if (!ck->is_fsync && (sz) res != ck->expect_len) {
      g->had_error   = 1;
      g->saved_errno = EIO;
      g->saved_tag   = ck->tag;
    }
  }
  /*  Return the cookie slot to the free stack.  */
  g->free_ck[g->n_free++] = (int) cqe->user_data;
  io_uring_cqe_seen(&g->ring, cqe);
  g->in_flight--;
}

/*  Drain until at least `reserve` cookie slots are free.  */
static void drain_until_reserve(xpar_iogroup * g, int reserve) {
  while (g->n_free < reserve) {
    struct io_uring_cqe * cqe;
    int rc = io_uring_submit_and_wait(&g->ring, 1);
    if (rc < 0 && rc != -EAGAIN && rc != -EINTR) {
      g->had_error = 1; g->saved_errno = -rc; break;
    }
    while (io_uring_peek_cqe(&g->ring, &cqe) == 0) reap_one(g, cqe);
  }
}

void xpar_iogroup_submit(xpar_iogroup * g) {
  if (!g) return;
  int rc = io_uring_submit(&g->ring);
  if (rc < 0 && !g->had_error) {
    g->had_error = 1;
    g->saved_errno = -rc;
  }
}

void xpar_iogroup_enqueue_write(xpar_iogroup * g,
                                int fid, const void * buf,
                                u64 off, sz len, u64 tag) {
  if (!g) return;
  if (fid < 0 || fid >= g->n_files) {
    g->had_error = 1; g->saved_errno = EINVAL; g->saved_tag = tag; return;
  }
  if (g->n_free == 0) drain_until_reserve(g, 1);
  if (g->had_error) return;

  struct io_uring_sqe * sqe = io_uring_get_sqe(&g->ring);
  if (!sqe) {
    /*  SQ is full; force submit and retry once.  */
    io_uring_submit(&g->ring);
    drain_until_reserve(g, 1);
    sqe = io_uring_get_sqe(&g->ring);
    if (!sqe) {
      g->had_error = 1; g->saved_errno = EAGAIN; g->saved_tag = tag; return;
    }
  }

  int idx = g->free_ck[--g->n_free];
  g->cookies[idx] = (uring_cookie){ .expect_len = len,
                                    .tag = tag, .is_fsync = 0 };
  io_uring_prep_write(sqe, g->fds[fid], buf, (unsigned) len, off);
  io_uring_sqe_set_data64(sqe, (u64) idx);
  g->in_flight++;
}

void xpar_iogroup_fsync(xpar_iogroup * g, int fid) {
  if (!g) return;
  if (fid < 0 || fid >= g->n_files) {
    g->had_error = 1; g->saved_errno = EINVAL; return;
  }
  /*  Drain pending writes first so the fsync follows them.  */
  xpar_iogroup_submit(g);
  drain_until_reserve(g, (int) g->depth);
  if (g->had_error) return;

  struct io_uring_sqe * sqe = io_uring_get_sqe(&g->ring);
  if (!sqe) {
    g->had_error = 1; g->saved_errno = EAGAIN; return;
  }
  int idx = g->free_ck[--g->n_free];
  g->cookies[idx] = (uring_cookie){ .expect_len = 0,
                                    .tag = 0, .is_fsync = 1 };
  io_uring_prep_fsync(sqe, g->fds[fid], IORING_FSYNC_DATASYNC);
  io_uring_sqe_set_data64(sqe, (u64) idx);
  g->in_flight++;
}

void xpar_iogroup_drain(xpar_iogroup * g) {
  if (!g) return;
  xpar_iogroup_submit(g);
  while (g->in_flight) {
    struct io_uring_cqe * cqe;
    int rc = io_uring_wait_cqe(&g->ring, &cqe);
    if (rc < 0) {
      if (rc == -EINTR) continue;
      g->had_error = 1; g->saved_errno = -rc; break;
    }
    reap_one(g, cqe);
    while (io_uring_peek_cqe(&g->ring, &cqe) == 0) reap_one(g, cqe);
  }
  if (g->had_error) {
    FATAL("io_uring write failed (tag=%llu): %s",
          (unsigned long long) g->saved_tag,
          strerror(g->saved_errno));
  }
}

void xpar_iogroup_free(xpar_iogroup * g) {
  if (!g) return;
  xpar_iogroup_drain(g);   /*  may FATAL, which is what we want  */
  io_uring_queue_exit(&g->ring);
  free(g->cookies);
  free(g->free_ck);
  free(g->fds);
  free(g);
}

unsigned xpar_iogroup_batch_records(void) {
  const char * e = getenv("XPAR_URING_BATCH");
  if (e && *e) {
    long v = strtol(e, NULL, 10);
    if (v >= 64 && v <= 65536) return (unsigned) v;
  }
  return 4096;
}

#endif  /*  XPAR_HAS_LIBURING  */
