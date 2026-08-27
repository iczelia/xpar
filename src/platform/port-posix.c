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

/*  POSIX host I/O over unbuffered descriptors, which permits positional
    writes without a competing stdio offset.  Loops retry EINTR and join
    partial transfers.  */

#if defined(_WIN32) && !defined(XPAR_FORCE_POSIX_HOST)
#error "port-posix.c compiled for a Windows target; use port-win32.c"
#endif

/*  common.h pulls in config.h, which carries the feature-test macros
    AC_USE_SYSTEM_EXTENSIONS defines. It must precede every system header
    or pread, fsync and clock_gettime go missing under -std=c99.  */
#include "common.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <sys/uio.h>
#include <signal.h>

#if defined(HAVE_UCONTEXT_H)
  #include <ucontext.h>
#endif

#if defined(HAVE_SYS_MMAN_H)
  #include <sys/mman.h>
#endif
#if defined(HAVE_SYS_TIME_H)
  #include <sys/time.h>
#endif
#if defined(HAVE_GETRANDOM) && defined(HAVE_SYS_RANDOM_H)
  #include <sys/random.h>
#endif
#if defined(HAVE_THREADS)
  #include <pthread.h>
#endif

const char * xpar_getenv(const char * name) { return getenv(name); }
#if defined(__linux__)
  #include <sys/sysmacros.h>
#endif
#if defined(HAVE_IO_URING)
  #include <linux/io_uring.h>
  #include <sys/syscall.h>
#endif

/*  flock where the host has it and fcntl F_SETLK where it does not.
    configure probes neither, so the header decides; __has_include is
    itself guarded, because a compiler without it must still reach the
    fcntl path rather than fail to build.  */
#if defined(HAVE_FLOCK)
  #define XPAR_TRY_FLOCK 1
#elif defined(__has_include)
  #if __has_include(<sys/file.h>)
    #define XPAR_TRY_FLOCK 1
  #endif
#endif
#if defined(XPAR_TRY_FLOCK)
  #include <sys/file.h>
  #if defined(LOCK_EX) && defined(LOCK_SH) && defined(LOCK_NB)
    #define XPAR_LOCK_FLOCK 1
  #endif
#endif

struct xpar_file {
  int  fd;
  bool owned;
  bool at_eof;
  int  last_errno;
};

static struct xpar_file g_stdin  = { 0, false, false, 0 };
static struct xpar_file g_stdout = { 1, false, false, 0 };
static struct xpar_file g_stderr = { 2, false, false, 0 };

xpar_file * const xpar_stdin  = &g_stdin;
xpar_file * const xpar_stdout = &g_stdout;
xpar_file * const xpar_stderr = &g_stderr;

void xpar_host_init(void) { }

/*  off_t is 32 bits on a 32-bit host built without _FILE_OFFSET_BITS=64,
    and a silent truncation there would read or write the wrong place in a
    large file. Range-check every offset that crosses into libc.  */
static bool off_fits(u64 v) {
  return sizeof(off_t) >= 8 || v <= 0x7fffffffULL;
}

#if defined(HAVE_OPENAT) && defined(O_DIRECTORY) && defined(O_NOFOLLOW)
static int open_nofollow(const char * path, int flags) {
  char * work = strdup(path), * p, * slash;
  int dfd, fd = -1, saved;
  if (!work) { errno = ENOMEM;  return -1; }
  dfd = open(path[0] == '/' ? "/" : ".",
             O_RDONLY | O_DIRECTORY | O_NOFOLLOW
#if defined(O_CLOEXEC)
             | O_CLOEXEC
#endif
             );
  if (dfd < 0) { free(work);  return -1; }
  p = work;
  while (*p == '/') p++;
  for (;;) {
    int next;
    slash = strchr(p, '/');
    if (slash) *slash = '\0';
    if (!*p || strcmp(p, ".") == 0) {
      if (!slash) { fd = dup(dfd);  break; }
    } else if (!slash) {
      fd = openat(dfd, p, flags | O_NOFOLLOW, 0666);
      break;
    } else {
      next = openat(dfd, p, O_RDONLY | O_DIRECTORY | O_NOFOLLOW
#if defined(O_CLOEXEC)
                    | O_CLOEXEC
#endif
                    );
      if (next < 0) break;
      close(dfd);  dfd = next;
    }
    p = slash + 1;
    while (*p == '/') p++;
  }
  saved = errno;
  close(dfd);  free(work);  errno = saved;
  return fd;
}
#endif

xpar_file * xpar_open(const char * path, int flags) {
  int acc = flags & 3, of;
  if      (acc == XPAR_O_WRONLY) of = O_WRONLY;
  else if (acc == XPAR_O_RDWR)   of = O_RDWR;
  else                           of = O_RDONLY;
  if (flags & XPAR_O_CREAT)  of |= O_CREAT;
  if (flags & XPAR_O_TRUNC)  of |= O_TRUNC;
  if (flags & XPAR_O_EXCL)   of |= O_EXCL;
  if (flags & XPAR_O_APPEND) of |= O_APPEND;
#if defined(O_NOFOLLOW)
  if (flags & XPAR_O_NOFOLLOW) of |= O_NOFOLLOW;
#endif
#if defined(O_CLOEXEC)
  of |= O_CLOEXEC;
#endif
  int fd;
  if (flags & XPAR_O_NOFOLLOW) {
#if defined(HAVE_OPENAT) && defined(O_DIRECTORY) && defined(O_NOFOLLOW)
    fd = open_nofollow(path, of);
#else
    errno = ENOTSUP;  fd = -1;
#endif
  } else {
    fd = open(path, of, 0666);
  }
  if (fd < 0) return NULL;
#if !defined(O_CLOEXEC)
  /*  A descriptor on a half-written volume must not leak into a child.  */
  { int fl = fcntl(fd, F_GETFD);
    if (fl != -1) fcntl(fd, F_SETFD, fl | FD_CLOEXEC); }
#endif
  struct xpar_file * f = malloc(sizeof(*f));
  if (!f) { close(fd);  errno = ENOMEM;  return NULL; }
  f->fd = fd;  f->owned = true;  f->at_eof = false;  f->last_errno = 0;
  return f;
}

int xpar_close(xpar_file * f) {
  if (!f) return 0;
  if (!f->owned) return 0;
  int r = close(f->fd) == 0 ? 0 : -1;
  free(f);
  return r;
}

sz xpar_read(xpar_file * f, void * buf, sz n) {
  sz got = 0;
  u8 * p = (u8 *) buf;
  while (got < n) {
    sz want = n - got;
    if (want > 0x10000000u) want = 0x10000000u;
    ssize_t r = read(f->fd, p + got, want);
    if (r < 0) {
      if (errno == EINTR) continue;
      f->last_errno = errno;
      break;
    }
    if (r == 0) { f->at_eof = true;  break; }
    got += (sz) r;
  }
  return got;
}

sz xpar_write(xpar_file * f, const void * buf, sz n) {
  sz done = 0;
  const u8 * p = (const u8 *) buf;
  while (done < n) {
    sz want = n - done;
    if (want > 0x10000000u) want = 0x10000000u;
    ssize_t r = write(f->fd, p + done, want);
    if (r < 0) {
      if (errno == EINTR) continue;
      f->last_errno = errno;
      break;
    }
    if (r == 0) break;
    done += (sz) r;
  }
  return done;
}

int xpar_seek(xpar_file * f, i64 off, int whence) {
  int w = whence == XPAR_SEEK_SET ? SEEK_SET
        : whence == XPAR_SEEK_CUR ? SEEK_CUR : SEEK_END;
  if (off > 0 && !off_fits((u64) off)) { errno = EOVERFLOW;  return -1; }
  if (lseek(f->fd, (off_t) off, w) == (off_t) -1) {
    f->last_errno = errno;
    return -1;
  }
  f->at_eof = false;
  return 0;
}

i64 xpar_tell(xpar_file * f) {
  off_t p = lseek(f->fd, 0, SEEK_CUR);
  return p == (off_t) -1 ? -1 : (i64) p;
}

int xpar_flush(xpar_file * f) { (void) f;  return 0; }

int xpar_fsync(xpar_file * f) {
  if (fsync(f->fd) == 0) return 0;
  /*  A pipe, a terminal or a read-only descriptor cannot lose data that a
      later fsync would have saved, so refusing to sync one is not a
      durability failure and must not abort a run.  */
  if (errno == EINVAL || errno == EACCES || errno == ENOTSUP ||
      errno == EBADF   || errno == EROFS) { errno = 0;  return 0; }
  f->last_errno = errno;
  return -1;
}

i64 xpar_size(xpar_file * f) {
  struct stat st;
  if (fstat(f->fd, &st) != 0) return -1;
  return (i64) st.st_size;
}

bool xpar_is_seekable(xpar_file * f) {
  return lseek(f->fd, 0, SEEK_CUR) != (off_t) -1;
}
bool xpar_is_tty(xpar_file * f) { return isatty(f->fd) == 1; }
bool xpar_eof   (xpar_file * f) { return f->at_eof; }
int  xpar_error (xpar_file * f) { return f->last_errno; }

int xpar_lock(xpar_file * f, bool exclusive) {
#if defined(XPAR_LOCK_FLOCK)
  int op = (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB;
  while (flock(f->fd, op) != 0) {
    if (errno == EINTR) continue;
    return -1;
  }
  return 0;
#else
  /* Use nonblocking whole-file locks with matching descriptor access. */
  struct flock fl;
  fl.l_type   = exclusive ? F_WRLCK : F_RDLCK;
  fl.l_whence = SEEK_SET;
  fl.l_start  = 0;
  fl.l_len    = 0;
  while (fcntl(f->fd, F_SETLK, &fl) != 0) {
    if (errno == EINTR) continue;
    return -1;
  }
  return 0;
#endif
}

int xpar_unlock(xpar_file * f) {
#if defined(XPAR_LOCK_FLOCK)
  return flock(f->fd, LOCK_UN) == 0 ? 0 : -1;
#else
  struct flock fl;
  fl.l_type   = F_UNLCK;
  fl.l_whence = SEEK_SET;
  fl.l_start  = 0;
  fl.l_len    = 0;
  return fcntl(f->fd, F_SETLK, &fl) == 0 ? 0 : -1;
#endif
}

/* Every supported POSIX backend provides locking. */
bool xpar_lock_supported(void) { return true; }

#if !defined(HAVE_PREAD) || !defined(HAVE_PWRITE)
  #if defined(HAVE_THREADS)
static pthread_mutex_t g_ofs_lock = PTHREAD_MUTEX_INITIALIZER;
    #define OFS_LOCK()   pthread_mutex_lock(&g_ofs_lock)
    #define OFS_UNLOCK() pthread_mutex_unlock(&g_ofs_lock)
  #else
    #define OFS_LOCK()   do { } while (0)
    #define OFS_UNLOCK() do { } while (0)
  #endif
#endif

sz xpar_pread(xpar_file * f, void * buf, sz n, u64 off) {
  sz got = 0;
  u8 * p = (u8 *) buf;
  if (!off_fits(off)) { f->last_errno = EOVERFLOW;  return 0; }
  while (got < n) {
    sz want = n - got;
    ssize_t r;
    if (want > 0x10000000u) want = 0x10000000u;
#if defined(HAVE_PREAD)
    r = pread(f->fd, p + got, want, (off_t) (off + got));
#else
    OFS_LOCK();
    if (lseek(f->fd, (off_t) (off + got), SEEK_SET) == (off_t) -1) r = -1;
    else r = read(f->fd, p + got, want);
    OFS_UNLOCK();
#endif
    if (r < 0) {
      if (errno == EINTR) continue;
      f->last_errno = errno;
      break;
    }
    /*  A short positional read is self-describing through the count, and
        the EOF flag belongs to the sequential cursor: setting it here
        would tell a streaming reader on the same handle that its own
        position had reached the end.  */
    if (r == 0) break;
    got += (sz) r;
  }
  return got;
}

#if defined(HAVE_IO_URING) && defined(HAVE_MMAP)
/*  Keep one small ring for the lifetime of the process. Creating, mapping,
    unmapping and closing a ring for every eight reads costs more than the
    reads themselves once the protected files are in the page cache. The
    public operation is still synchronous, and the lock makes the singleton
    safe for callers which issue batches from several worker threads.  */
typedef struct {
  struct io_uring_params p;
  struct io_uring_sqe * sqe;
  struct io_uring_cqe * cqe;
  void * sqmap, * cqmap, * smap;
  sz sqsz, cqsz, smapsz;
  u32 * sq_head, * sq_tail, * sq_mask, * sq_array;
  u32 * cq_head, * cq_tail, * cq_mask;
  int fd, state;                 /*  0 untried, 1 ready, -1 unavailable.  */
} xpar_uring;

static xpar_uring g_uring = {
  { 0 }, NULL, NULL, MAP_FAILED, MAP_FAILED, MAP_FAILED,
  0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, -1, 0
};
#if defined(HAVE_THREADS)
static pthread_mutex_t g_uring_lock = PTHREAD_MUTEX_INITIALIZER;
  #define URING_LOCK()   pthread_mutex_lock(&g_uring_lock)
  #define URING_UNLOCK() pthread_mutex_unlock(&g_uring_lock)
#else
  #define URING_LOCK()   do { } while (0)
  #define URING_UNLOCK() do { } while (0)
#endif

static void uring_close(xpar_uring * u) {
  if (u->smap != MAP_FAILED) munmap(u->smap, u->smapsz);
  if (u->sqmap != MAP_FAILED) munmap(u->sqmap, u->sqsz);
  if (!(u->p.features & IORING_FEAT_SINGLE_MMAP) &&
      u->cqmap != MAP_FAILED) munmap(u->cqmap, u->cqsz);
  if (u->fd >= 0) close(u->fd);
  u->sqmap = u->cqmap = u->smap = MAP_FAILED;
  u->fd = -1;  u->state = -1;
}

static void uring_close_global(void) {
  uring_close(&g_uring);
}

static bool uring_open(xpar_uring * u, sz need) {
  u32 entries = (u32) MAX(need, (sz) 64);
  xpar_memset(&u->p, 0, sizeof u->p);
  u->fd = (int) syscall(__NR_io_uring_setup, entries, &u->p);
  if (u->fd < 0) { u->state = -1;  return false; }
  u->sqsz = u->p.sq_off.array + (sz) u->p.sq_entries * sizeof(u32);
  u->cqsz = u->p.cq_off.cqes +
            (sz) u->p.cq_entries * sizeof(struct io_uring_cqe);
  if (u->p.features & IORING_FEAT_SINGLE_MMAP) {
    u->sqsz = u->cqsz = MAX(u->sqsz, u->cqsz);
    u->sqmap = mmap(NULL, u->sqsz, PROT_READ | PROT_WRITE, MAP_SHARED,
                    u->fd, IORING_OFF_SQ_RING);
    u->cqmap = u->sqmap;
  } else {
    u->sqmap = mmap(NULL, u->sqsz, PROT_READ | PROT_WRITE, MAP_SHARED,
                    u->fd, IORING_OFF_SQ_RING);
    u->cqmap = mmap(NULL, u->cqsz, PROT_READ | PROT_WRITE, MAP_SHARED,
                    u->fd, IORING_OFF_CQ_RING);
  }
  u->smapsz = (sz) u->p.sq_entries * sizeof(struct io_uring_sqe);
  u->smap = mmap(NULL, u->smapsz, PROT_READ | PROT_WRITE, MAP_SHARED,
                 u->fd, IORING_OFF_SQES);
  if (u->sqmap == MAP_FAILED || u->cqmap == MAP_FAILED ||
      u->smap == MAP_FAILED) {
    uring_close(u);
    return false;
  }
  u->sq_head  = (u32 *) ((u8 *) u->sqmap + u->p.sq_off.head);
  u->sq_tail  = (u32 *) ((u8 *) u->sqmap + u->p.sq_off.tail);
  u->sq_mask  = (u32 *) ((u8 *) u->sqmap + u->p.sq_off.ring_mask);
  u->sq_array = (u32 *) ((u8 *) u->sqmap + u->p.sq_off.array);
  u->cq_head  = (u32 *) ((u8 *) u->cqmap + u->p.cq_off.head);
  u->cq_tail  = (u32 *) ((u8 *) u->cqmap + u->p.cq_off.tail);
  u->cq_mask  = (u32 *) ((u8 *) u->cqmap + u->p.cq_off.ring_mask);
  u->sqe = (struct io_uring_sqe *) u->smap;
  u->cqe = (struct io_uring_cqe *)
             ((u8 *) u->cqmap + u->p.cq_off.cqes);
  u->state = 1;
  atexit(uring_close_global);
  return true;
}
#endif

bool xpar_pread_batch(xpar_read_req * r, sz count) {
#if defined(HAVE_IO_URING) && defined(HAVE_MMAP)
  xpar_uring * u = &g_uring;
  sz submitted = 0, completed = 0, i;
  bool used = false;
  if (!count) return false;
  if (count > 1024 || count > (sz) UINT32_MAX)
    return xpar_pread_serial(r, count);
  for (i = 0; i < count; i++)
    if (r[i].length > (sz) UINT32_MAX)
      return xpar_pread_serial(r, count);
  URING_LOCK();
  if (u->state == 0 && !uring_open(u, count)) goto fallback;
  if (u->state != 1 || count > u->p.sq_entries) goto fallback;

  for (i = 0; i < count; i++) {
    u32 tail, slot;
    r[i].result = 0;
    if (!r[i].file || !r[i].length) continue;
    tail = *u->sq_tail + (u32) submitted;
    slot = tail & *u->sq_mask;
    xpar_memset(&u->sqe[slot], 0, sizeof(u->sqe[slot]));
    u->sqe[slot].opcode = IORING_OP_READ;
    u->sqe[slot].fd = r[i].file->fd;
    u->sqe[slot].off = r[i].offset;
    u->sqe[slot].addr = (u64) (uintptr_t) r[i].buf;
    u->sqe[slot].len = (u32) r[i].length;
    u->sqe[slot].user_data = (u64) i + 1;
    u->sq_array[slot] = slot;
    submitted++;
  }
  if (!submitted) { used = true;  goto done; }
  __atomic_store_n(u->sq_tail, *u->sq_tail + (u32) submitted,
                   __ATOMIC_RELEASE);
  while (completed < submitted) {
    int entered = (int) syscall(__NR_io_uring_enter, u->fd,
                                 completed ? 0 : (u32) submitted,
                                 (u32) (submitted - completed),
                                 IORING_ENTER_GETEVENTS, NULL, 0);
    u32 head, tail;
    if (entered < 0 && errno == EINTR) continue;
    if (entered < 0) { uring_close(u);  goto fallback; }
    head = *u->cq_head;
    tail = __atomic_load_n(u->cq_tail, __ATOMIC_ACQUIRE);
    while (head != tail) {
      struct io_uring_cqe * q = &u->cqe[head & *u->cq_mask];
      if (q->user_data && q->user_data <= count) {
        i = (sz) q->user_data - 1;
        if (q->res > 0) r[i].result = (sz) q->res;
      }
      head++;  completed++;
    }
    __atomic_store_n(u->cq_head, head, __ATOMIC_RELEASE);
  }
  used = true;
done:
  URING_UNLOCK();
  for (i = 0; i < count; i++)
    if (r[i].file && r[i].result < r[i].length) {
      sz n = xpar_pread(r[i].file, (u8 *) r[i].buf + r[i].result,
                        r[i].length - r[i].result,
                        r[i].offset + r[i].result);
      r[i].result += n;
    }
  return used;

fallback:
  URING_UNLOCK();
  return xpar_pread_serial(r, count);
#else
  return xpar_pread_serial(r, count);
#endif
}

sz xpar_pwrite(xpar_file * f, const void * buf, sz n, u64 off) {
  sz done = 0;
  const u8 * p = (const u8 *) buf;
  if (!off_fits(off)) { f->last_errno = EOVERFLOW;  return 0; }
  while (done < n) {
    sz want = n - done;
    ssize_t r;
    if (want > 0x10000000u) want = 0x10000000u;
#if defined(HAVE_PWRITE)
    r = pwrite(f->fd, p + done, want, (off_t) (off + done));
#else
    OFS_LOCK();
    if (lseek(f->fd, (off_t) (off + done), SEEK_SET) == (off_t) -1) r = -1;
    else r = write(f->fd, p + done, want);
    OFS_UNLOCK();
#endif
    if (r < 0) {
      if (errno == EINTR) continue;
      f->last_errno = errno;
      break;
    }
    if (r == 0) break;
    done += (sz) r;
  }
  return done;
}

int xpar_ftruncate(xpar_file * f, u64 length) {
  if (!off_fits(length)) { errno = EOVERFLOW;  return -1; }
#if defined(HAVE_FTRUNCATE)
  while (ftruncate(f->fd, (off_t) length) != 0) {
    if (errno == EINTR) continue;
    f->last_errno = errno;
    return -1;
  }
  return 0;
#else
  (void) f;  errno = ENOSYS;  return -1;
#endif
}

/*  A rename or a create is not on disk until the *directory* is synced;
    syncing the file only covers its contents.  */
int xpar_fsync_dir(const char * path) {
  sz len = xpar_strlen(path), cut = 0;
  char * dir;
  int fd, r;
  for (sz i = 0; i < len; i++) if (path[i] == '/') cut = i;
  dir = cut ? xpar_strndup(path, cut) : xpar_strdup(len && path[0] == '/'
                                                    ? "/" : ".");
#if defined(HAVE_OPENAT) && defined(O_DIRECTORY) && defined(O_NOFOLLOW)
  fd = open_nofollow(dir, O_RDONLY | O_DIRECTORY);
#else
  fd = open(dir, O_RDONLY);
#endif
  xpar_free(dir);
  if (fd < 0) return -1;
  r = fsync(fd);
  /*  Some filesystems refuse to sync a directory descriptor. Nothing was
      lost that a different call could have saved, so this is not a
      failure the caller can act on.  */
  if (r != 0 && (errno == EINVAL || errno == ENOTSUP)) r = 0;
  close(fd);
  return r;
}

sz xpar_xread(xpar_file * f, void * p, sz n) {
  sz got = xpar_read(f, p, n);
  if (f->last_errno) { errno = f->last_errno;  FATAL_PERROR("read"); }
  return got;
}

void xpar_xwrite(xpar_file * f, const void * p, sz n) {
  if (xpar_write(f, p, n) != n) {
    errno = f->last_errno ? f->last_errno : ENOSPC;
    FATAL_PERROR("write");
  }
}

void xpar_xwritev(xpar_file * f, const xpar_write_part * part, u32 count) {
  struct iovec vec[16];
  u32 at = 0, used = 0;
  if (count > ARRAY_LEN(vec)) {
    for (at = 0; at < count; at++)
      xpar_xwrite(f, part[at].data, part[at].length);
    return;
  }
  for (at = 0; at < count; at++) {
    if (!part[at].length) continue;
    if (part[at].length > 0x10000000u) {
      for (at = 0; at < count; at++)
        xpar_xwrite(f, part[at].data, part[at].length);
      return;
    }
    /* writev declares iov_base as non-const. */
    vec[used].iov_base = (void *) part[at].data;
    vec[used].iov_len = part[at].length;
    used++;
  }
  if (!used) return;
  at = 0;
  while (at < used) {
    ssize_t n = writev(f->fd, vec + at, (int) (used - at));
    if (n < 0) {
      if (errno == EINTR) continue;
      f->last_errno = errno;
      break;
    }
    if (n == 0) break;
    while (at < used && (sz) n >= vec[at].iov_len) {
      n -= (ssize_t) vec[at].iov_len;
      at++;
    }
    if (at < used && n > 0) {
      vec[at].iov_base = (u8 *) vec[at].iov_base + n;
      vec[at].iov_len -= (sz) n;
    }
  }
  if (at != used) {
    errno = f->last_errno ? f->last_errno : ENOSPC;
    FATAL_PERROR("writev");
  }
}

void xpar_xclose(xpar_file * f) {
  if (!f) return;
  if (!f->owned) return;
  if (xpar_fsync(f) != 0) { errno = f->last_errno;  FATAL_PERROR("fsync"); }
  /*  close() is where a delayed write error on NFS surfaces, so it is
      checked rather than assumed to succeed.  */
  if (close(f->fd) != 0) FATAL_PERROR("close");
  free(f);
}

xpar_mmap xpar_map(const char * path) {
  xpar_mmap m;
  m.map = NULL;  m.size = 0;  m.valid = false;
#if defined(HAVE_MMAP)
  {
    struct stat st;
    void * p;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return m;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd);  return m; }
    /*  sz is 32-bit on some hosts and st_size is not: a file larger than
        the address space cannot be mapped and the caller streams it.  */
    if ((u64) st.st_size > (u64) (sz) -1) { close(fd);  return m; }
    m.size = (sz) st.st_size;
    p = mmap(NULL, m.size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { m.size = 0;  return m; }
    m.map = (u8 *) p;  m.valid = true;
  }
#else
  (void) path;
#endif
  return m;
}

void xpar_unmap(xpar_mmap * m) {
#if defined(HAVE_MMAP)
  if (m->map) munmap(m->map, m->size);
#endif
  m->map = NULL;  m->size = 0;  m->valid = false;
}

static void advise(xpar_file * f, u64 off, u64 len, int advice) {
#if defined(HAVE_POSIX_FADVISE)
  if (!off_fits(off) || !off_fits(len)) return;
  /*  Best effort by contract: the return value is deliberately dropped,
      since a host that declines the hint still reads the same bytes.  */
  (void) posix_fadvise(f->fd, (off_t) off, (off_t) len, advice);
#else
  (void) f;  (void) off;  (void) len;  (void) advice;
#endif
}

void xpar_advise_sequential(xpar_file * f, u64 off, u64 len) {
#if defined(HAVE_POSIX_FADVISE)
  advise(f, off, len, POSIX_FADV_SEQUENTIAL);
#else
  advise(f, off, len, 0);
#endif
}

void xpar_advise_random(xpar_file * f, u64 off, u64 len) {
#if defined(HAVE_POSIX_FADVISE)
  advise(f, off, len, POSIX_FADV_RANDOM);
#else
  advise(f, off, len, 0);
#endif
}

void * xpar_malloc(sz n) {
  void * p = calloc(n ? n : 1, 1);
  if (!p) FATAL_CODE(XPAR_EXIT_NOPLAN, "Out of memory.");
  return p;
}

void * xpar_calloc(sz n, sz size) {
  if (n && size && n > (sz) -1 / size) FATAL_CODE(XPAR_EXIT_NOPLAN,
                              "Allocation size overflow.");
  { void * p = calloc(n ? n : 1, size ? size : 1);
    if (!p) FATAL_CODE(XPAR_EXIT_NOPLAN, "Out of memory.");
    return p; }
}

void * xpar_alloc_raw(sz n) {
  void * p = malloc(n ? n : 1);
  if (!p) FATAL_CODE(XPAR_EXIT_NOPLAN, "Out of memory.");
  return p;
}

void * xpar_realloc(void * p, sz n) {
  void * q = realloc(p, n ? n : 1);
  if (!q) FATAL_CODE(XPAR_EXIT_NOPLAN, "Out of memory.");
  return q;
}

void xpar_free(void * p) { free(p); }

void * xpar_alloc_aligned(sz n, sz align) {
  if (align < sizeof(void *)) align = sizeof(void *);
  if (!xpar_is_pow2(align)) FATAL("Alignment is not a power of two.");
  if (n == 0) n = 1;
#if defined(HAVE_POSIX_MEMALIGN)
  { void * p = NULL;
    if (posix_memalign(&p, align, n) != 0 || !p)
      FATAL_CODE(XPAR_EXIT_NOPLAN, "Out of memory.");
    return p; }
#else
  { sz pad = align + sizeof(void *);
    u8 * raw;
    uintptr_t a;
    if (n > (sz) -1 - pad) FATAL_CODE(XPAR_EXIT_NOPLAN,
                              "Allocation size overflow.");
    raw = (u8 *) malloc(n + pad);
    if (!raw) FATAL_CODE(XPAR_EXIT_NOPLAN, "Out of memory.");
    a = ((uintptr_t) raw + sizeof(void *) + align - 1) &
        ~(uintptr_t) (align - 1);
    ((void **) a)[-1] = raw;
    return (void *) a; }
#endif
}

void xpar_free_aligned(void * p) {
  if (!p) return;
#if defined(HAVE_POSIX_MEMALIGN)
  free(p);
#else
  free(((void **) p)[-1]);
#endif
}

u64 xpar_physical_memory(void) {
#if defined(HAVE_SYSCONF) && defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
  long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGESIZE);
  if (pages > 0 && psz > 0) return (u64) pages * (u64) psz;
#endif
  return 0;
}

bool xpar_is_rotational(const char * path) {
#if defined(__linux__)
  struct stat st;
  char name[64];
  unsigned maj, min;
  if (stat(path, &st) != 0) return false;
  maj = (unsigned) major(st.st_dev);
  min = (unsigned) minor(st.st_dev);
  for (int i = 0; i < 2; i++) {
    char c;
    int fd;
    if (i == 0)
      xpar_snprintf(name, sizeof name,
                    "/sys/dev/block/%u:%u/queue/rotational", maj, min);
    else
      xpar_snprintf(name, sizeof name,
                    "/sys/dev/block/%u:%u/../queue/rotational", maj, min);
    fd = open(name, O_RDONLY);
    if (fd < 0) continue;
    { ssize_t r = read(fd, &c, 1);
      close(fd);
      if (r == 1) return c == '1'; }
  }
#else
  (void) path;
#endif
  return false;
}

void * xpar_memcpy (void * d, const void * s, sz n) { return memcpy(d, s, n); }
void * xpar_memmove(void * d, const void * s, sz n) { return memmove(d, s, n); }
void * xpar_memset (void * d, int c, sz n)          { return memset(d, c, n); }
int    xpar_memcmp (const void * a, const void * b, sz n) {
  return memcmp(a, b, n);
}
sz     xpar_strlen (const char * s)                 { return strlen(s); }
int    xpar_strcmp (const char * a, const char * b) { return strcmp(a, b); }
int    xpar_strncmp(const char * a, const char * b, sz n) {
  return strncmp(a, b, n);
}

int xpar_vsnprintf(char * buf, sz cap, const char * fmt, va_list ap) {
  return vsnprintf(buf, cap, fmt, ap);
}

int xpar_vfprintf(xpar_file * f, const char * fmt, va_list ap) {
  char stack[1024];
  va_list ap2;
  int n;
  va_copy(ap2, ap);
  n = vsnprintf(stack, sizeof stack, fmt, ap);
  if (n < 0) { va_end(ap2);  return -1; }
  if ((sz) n < sizeof stack) {
    va_end(ap2);
    xpar_write(f, stack, (sz) n);
    return n;
  }
  { char * big = xpar_alloc_raw((sz) n + 1);
    vsnprintf(big, (sz) n + 1, fmt, ap2);
    va_end(ap2);
    xpar_write(f, big, (sz) n);
    xpar_free(big); }
  return n;
}

int xpar_fputs(const char * s, xpar_file * f) {
  sz n = strlen(s);
  return (int) xpar_write(f, s, n);
}

void xpar_exit(int code) { exit(code); }

const char * xpar_strerror(int err) { return strerror(err); }
int          xpar_errno(void)       { return errno; }

u64 xpar_usec_now(void) {
#if defined(CLOCK_MONOTONIC)
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
    return (u64) ts.tv_sec * 1000000ULL + (u64) ts.tv_nsec / 1000ULL;
#endif
#if defined(HAVE_SYS_TIME_H)
  { struct timeval tv;
    if (gettimeofday(&tv, NULL) == 0)
      return (u64) tv.tv_sec * 1000000ULL + (u64) tv.tv_usec; }
#endif
  return (u64) time(NULL) * 1000000ULL;
}

i64 xpar_wall_ns(void) {
#if defined(CLOCK_REALTIME)
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) == 0)
    return (i64) ts.tv_sec * 1000000000LL + (i64) ts.tv_nsec;
#endif
#if defined(HAVE_SYS_TIME_H)
  { struct timeval tv;
    if (gettimeofday(&tv, NULL) == 0)
      return (i64) tv.tv_sec * 1000000000LL + (i64) tv.tv_usec * 1000LL; }
#endif
  return (i64) time(NULL) * 1000000000LL;
}

void xpar_random_bytes(void * buf, sz n) {
  u8 * p = (u8 *) buf;
  sz got = 0;
#if defined(HAVE_GETRANDOM)
  while (got < n) {
    ssize_t r = getrandom(p + got, n - got, 0);
    if (r < 0) {
      if (errno == EINTR) continue;
      break;   /*  ENOSYS on a pre-3.17 kernel; try the other sources.  */
    }
    got += (sz) r;
  }
  if (got == n) return;
#endif
#if defined(HAVE_ARC4RANDOM_BUF)
  arc4random_buf(p + got, n - got);
  return;
#else
  { int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
      while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r < 0) { if (errno == EINTR) continue;  break; }
        if (r == 0) break;
        got += (sz) r;
      }
      close(fd);
    }
  }
  if (got == n) return;
  FATAL("No source of cryptographically strong random bytes.");
#endif
}

/*  The process entry point lives in the host file so that the Win32 build
    can replace it wholesale; xpar_main is the portable one.  */
int main(int argc, char ** argv) {
  xpar_host_init();
  xpar_crash_install();
  return xpar_main(argc, argv);
}

/*  Report the fault, then re-raise it with the default disposition.  */
#if defined(HAVE_SIGACTION)

#if defined(HAVE_EXECINFO_H)
#include <execinfo.h>
#endif

#define XPAR_CRASH_FRAMES 48

static const char * crash_name(int sig) {
  switch (sig) {
    case SIGSEGV: return "invalid memory reference (SIGSEGV)";
    case SIGBUS:  return "bus error (SIGBUS)";
    case SIGILL:  return "illegal instruction (SIGILL)";
    case SIGFPE:  return "arithmetic exception (SIGFPE)";
    case SIGABRT: return "aborted (SIGABRT)";
    default:      return "fatal signal";
  }
}

/*  Extract the faulting instruction address when available.  */
static const void * crash_pc(void * uc) {
#if defined(__linux__) && defined(HAVE_UCONTEXT_H)
  ucontext_t * u = (ucontext_t *) uc;
  if (!u) return NULL;
#if defined(__x86_64__)
  return (const void *) (uintptr_t) u->uc_mcontext.gregs[REG_RIP];
#elif defined(__i386__)
  return (const void *) (uintptr_t) u->uc_mcontext.gregs[REG_EIP];
#elif defined(__aarch64__)
  return (const void *) (uintptr_t) u->uc_mcontext.pc;
#elif defined(__arm__)
  return (const void *) (uintptr_t) u->uc_mcontext.arm_pc;
#else
  return NULL;
#endif
#else
  (void) uc;
  return NULL;
#endif
}

static void crash_handler(int sig, siginfo_t * si, void * uc) {
  void * frames[XPAR_CRASH_FRAMES];
  unsigned n = 0;
  int have_addr = si && si->si_code > 0 &&
                  (sig == SIGSEGV || sig == SIGBUS || sig == SIGILL ||
                   sig == SIGFPE);
  if (xpar_crash_entered()) _exit(XPAR_EXIT_INTERNAL);
#if defined(HAVE_BACKTRACE)
  { int got = backtrace(frames, XPAR_CRASH_FRAMES);
    if (got > 0) n = (unsigned) got; }
#else
  n = xpar_crash_walk_fp((void * const *) __builtin_frame_address(0),
                         frames, XPAR_CRASH_FRAMES);
#endif
  xpar_crash_head(crash_name(sig), (u64) sig, 1, crash_pc(uc),
                  have_addr ? si->si_addr : NULL, have_addr, NULL);
#if defined(HAVE_BACKTRACE)
  /*  Avoid backtrace_symbols(), which allocates.  */
  if (n) backtrace_symbols_fd(frames, (int) n, 2);
#else
  { unsigned i;
    for (i = 0; i < n; i++) xpar_crash_frame(i, frames[i], NULL); }
#endif
  xpar_crash_tail(n != 0);
  /*  Preserve normal core-dump behavior.  */
  { struct sigaction sa;
    xpar_memset(&sa, 0, sizeof sa);
    sa.sa_handler = SIG_DFL;
    sigaction(sig, &sa, NULL); }
  raise(sig);
  _exit(XPAR_EXIT_INTERNAL);
}

void xpar_crash_install(void) {
  if (!xpar_crash_wanted()) return;
  static const int sigs[] = { SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT };
  struct sigaction sa;
  sz i;
  xpar_memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = crash_handler;
  /* Preserve SA_RESETHAND's sign-bit pattern when narrowing. */
  sa.sa_flags = (int) (unsigned) (SA_SIGINFO | SA_NODEFER |
                                  SA_RESETHAND);
  sigemptyset(&sa.sa_mask);
  for (i = 0; i < sizeof sigs / sizeof *sigs; i++)
    (void) sigaction(sigs[i], &sa, NULL);
}

#else
void xpar_crash_install(void) { }
#endif
