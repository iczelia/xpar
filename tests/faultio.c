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

/*  LD_PRELOAD fault injector for hostfaults.sh. XPAR_FI_<OP> fails calls;
    XPAR_FI_CRASH_<OP> exits after success. XPAR_FI_PATH filters paths;
    XPAR_FI_STICKY and XPAR_FI_TRACE alter behavior. Linux/glibc only.  */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

enum { OPENW, OPENR, UNLINK, RMDIR, RENAME, LINK, SYMLINK, MKDIR, FSYNC,
       FSYNCDIR, FTRUNCATE, PWRITE, WRITE, PREAD, READ, CHMOD, UTIMES, CHOWN,
       CLOSE, NOPS };
static const char * names[NOPS] = { "OPENW", "OPENR", "UNLINK", "RMDIR",
  "RENAME", "LINK", "SYMLINK", "MKDIR", "FSYNC", "FSYNCDIR", "FTRUNCATE",
  "PWRITE", "WRITE", "PREAD", "READ", "CHMOD", "UTIMES", "CHOWN", "CLOSE" };
static long fail_at[NOPS], crash_at[NOPS], count[NOPS];
static int inited, trace, sticky, err_no = EIO;
static const char * path_filter;
static ssize_t (*real_write)(int, const void *, size_t);

static void init(void) {
  char key[64];
  int i;
  if (inited) return;
  inited = 1;
  real_write = dlsym(RTLD_NEXT, "write");
  for (i = 0; i < NOPS; i++) {
    const char * v;
    snprintf(key, sizeof key, "XPAR_FI_%s", names[i]);
    v = getenv(key);  fail_at[i] = v ? atol(v) : 0;
    snprintf(key, sizeof key, "XPAR_FI_CRASH_%s", names[i]);
    v = getenv(key);  crash_at[i] = v ? atol(v) : 0;
  }
  path_filter = getenv("XPAR_FI_PATH");
  if (path_filter && !*path_filter) path_filter = NULL;
  trace = getenv("XPAR_FI_TRACE") != NULL;
  sticky = getenv("XPAR_FI_STICKY") != NULL;
  if (getenv("XPAR_FI_ERRNO")) err_no = atoi(getenv("XPAR_FI_ERRNO"));
}

/*  Apply the filter only to absolute paths.  */
static int matches(const char * path) {
  if (!path) return path_filter == NULL;
  if (path[0] != '/') return 1;
  if (!path_filter) return 1;
  return strstr(path, path_filter) != NULL;
}

static const char * fd_path(int fd, char * buf, size_t n) {
  char link[64];
  ssize_t got;
  snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
  got = readlink(link, buf, n - 1);
  if (got < 0) return NULL;
  buf[got] = 0;
  return buf;
}

/*  Fail before the call; after() handles crash injection.  */
static int before(int op, const char * path) {
  long k;
  init();
  if (!matches(path)) return 0;
  k = __atomic_add_fetch(&count[op], 1, __ATOMIC_SEQ_CST);
  if (trace) {
    char m[512];
    int n = snprintf(m, sizeof m, "FI %s #%ld %s\n", names[op], k,
                     path ? path : "-");
    real_write(2, m, (size_t) n);
  }
  if (fail_at[op] && (k == fail_at[op] || (sticky && k > fail_at[op]))) {
    char m[512];
    int n = snprintf(m, sizeof m, "FI %s #%ld FAILED %s\n", names[op], k,
                     path ? path : "-");
    real_write(2, m, (size_t) n);
    return 1;
  }
  return 0;
}

static void after(int op, const char * path) {
  long k = count[op];
  if (!matches(path)) return;
  if (crash_at[op] && k == crash_at[op]) {
    char m[512];
    int n = snprintf(m, sizeof m, "FI %s #%ld CRASH %s\n", names[op], k,
                     path ? path : "-");
    real_write(2, m, (size_t) n);
    _exit(97);
  }
}

#define REAL(name) static typeof(name) * r_##name; \
  if (!r_##name) r_##name = dlsym(RTLD_NEXT, #name)

static int open_common(int op, const char * path) {
  if (before(op, path)) { errno = err_no;  return -1; }
  return 0;
}

int open(const char * path, int flags, ...) {
  va_list ap;  mode_t mode = 0;  int r;
  int op = (flags & (O_WRONLY | O_RDWR | O_CREAT)) ? OPENW : OPENR;
  REAL(open);
  va_start(ap, flags);
  if (flags & (O_CREAT | O_TMPFILE)) mode = va_arg(ap, mode_t);
  va_end(ap);
  if (open_common(op, path)) return -1;
  r = r_open(path, flags, mode);
  if (r >= 0) after(op, path);
  return r;
}
int open64(const char * path, int flags, ...) {
  va_list ap;  mode_t mode = 0;  int r;
  int op = (flags & (O_WRONLY | O_RDWR | O_CREAT)) ? OPENW : OPENR;
  REAL(open64);
  va_start(ap, flags);
  if (flags & (O_CREAT | O_TMPFILE)) mode = va_arg(ap, mode_t);
  va_end(ap);
  if (open_common(op, path)) return -1;
  r = r_open64(path, flags, mode);
  if (r >= 0) after(op, path);
  return r;
}
int openat(int dirfd, const char * path, int flags, ...) {
  va_list ap;  mode_t mode = 0;  int r;
  int op = (flags & (O_WRONLY | O_RDWR | O_CREAT)) ? OPENW : OPENR;
  REAL(openat);
  va_start(ap, flags);
  if (flags & (O_CREAT | O_TMPFILE)) mode = va_arg(ap, mode_t);
  va_end(ap);
  if (open_common(op, path)) return -1;
  r = r_openat(dirfd, path, flags, mode);
  if (r >= 0) after(op, path);
  return r;
}
int openat64(int dirfd, const char * path, int flags, ...) {
  va_list ap;  mode_t mode = 0;  int r;
  int op = (flags & (O_WRONLY | O_RDWR | O_CREAT)) ? OPENW : OPENR;
  REAL(openat64);
  va_start(ap, flags);
  if (flags & (O_CREAT | O_TMPFILE)) mode = va_arg(ap, mode_t);
  va_end(ap);
  if (open_common(op, path)) return -1;
  r = r_openat64(dirfd, path, flags, mode);
  if (r >= 0) after(op, path);
  return r;
}

int unlink(const char * path) {
  int r;  REAL(unlink);
  if (before(UNLINK, path)) { errno = err_no;  return -1; }
  r = r_unlink(path);  if (r == 0) after(UNLINK, path);  return r;
}
int unlinkat(int dirfd, const char * path, int flags) {
  int r, op = (flags & AT_REMOVEDIR) ? RMDIR : UNLINK;  REAL(unlinkat);
  if (before(op, path)) { errno = err_no;  return -1; }
  r = r_unlinkat(dirfd, path, flags);  if (r == 0) after(op, path);  return r;
}
int rmdir(const char * path) {
  int r;  REAL(rmdir);
  if (before(RMDIR, path)) { errno = err_no;  return -1; }
  r = r_rmdir(path);  if (r == 0) after(RMDIR, path);  return r;
}
int rename(const char * a, const char * b) {
  int r;  REAL(rename);
  if (before(RENAME, b)) { errno = err_no;  return -1; }
  r = r_rename(a, b);  if (r == 0) after(RENAME, b);  return r;
}
int renameat(int da, const char * a, int db, const char * b) {
  int r;  REAL(renameat);
  if (before(RENAME, b)) { errno = err_no;  return -1; }
  r = r_renameat(da, a, db, b);  if (r == 0) after(RENAME, b);  return r;
}
int link(const char * a, const char * b) {
  int r;  REAL(link);
  if (before(LINK, b)) { errno = err_no;  return -1; }
  r = r_link(a, b);  if (r == 0) after(LINK, b);  return r;
}
int linkat(int da, const char * a, int db, const char * b, int flags) {
  int r;  REAL(linkat);
  if (before(LINK, b)) { errno = err_no;  return -1; }
  r = r_linkat(da, a, db, b, flags);  if (r == 0) after(LINK, b);  return r;
}
int symlink(const char * t, const char * p) {
  int r;  REAL(symlink);
  if (before(SYMLINK, p)) { errno = err_no;  return -1; }
  r = r_symlink(t, p);  if (r == 0) after(SYMLINK, p);  return r;
}
int symlinkat(const char * t, int d, const char * p) {
  int r;  REAL(symlinkat);
  if (before(SYMLINK, p)) { errno = err_no;  return -1; }
  r = r_symlinkat(t, d, p);  if (r == 0) after(SYMLINK, p);  return r;
}
int mkdir(const char * p, mode_t m) {
  int r;  REAL(mkdir);
  if (before(MKDIR, p)) { errno = err_no;  return -1; }
  r = r_mkdir(p, m);  if (r == 0) after(MKDIR, p);  return r;
}
int mkdirat(int d, const char * p, mode_t m) {
  int r;  REAL(mkdirat);
  if (before(MKDIR, p)) { errno = err_no;  return -1; }
  r = r_mkdirat(d, p, m);  if (r == 0) after(MKDIR, p);  return r;
}
static int fsync_common(int fd, int (*real)(int)) {
  struct stat st;  char buf[512];  const char * p;  int op, r;
  if (fstat(fd, &st) != 0) return real(fd);
  op = S_ISDIR(st.st_mode) ? FSYNCDIR : FSYNC;
  p = fd_path(fd, buf, sizeof buf);
  if (before(op, p)) { errno = err_no;  return -1; }
  r = real(fd);  if (r == 0) after(op, p);  return r;
}
int fsync(int fd) { REAL(fsync);  return fsync_common(fd, r_fsync); }
int fdatasync(int fd) { REAL(fdatasync);  return fsync_common(fd, r_fdatasync); }
int ftruncate(int fd, off_t len) {
  char buf[512];  const char * p = fd_path(fd, buf, sizeof buf);  int r;
  REAL(ftruncate);
  if (before(FTRUNCATE, p)) { errno = err_no;  return -1; }
  r = r_ftruncate(fd, len);  if (r == 0) after(FTRUNCATE, p);  return r;
}
int ftruncate64(int fd, off64_t len) {
  char buf[512];  const char * p = fd_path(fd, buf, sizeof buf);  int r;
  REAL(ftruncate64);
  if (before(FTRUNCATE, p)) { errno = err_no;  return -1; }
  r = r_ftruncate64(fd, len);  if (r == 0) after(FTRUNCATE, p);  return r;
}
ssize_t pwrite(int fd, const void * b, size_t n, off_t off) {
  char buf[512];  const char * p;  ssize_t r;  REAL(pwrite);
  if (fd <= 2) return r_pwrite(fd, b, n, off);
  p = fd_path(fd, buf, sizeof buf);
  if (before(PWRITE, p)) { errno = err_no;  return -1; }
  r = r_pwrite(fd, b, n, off);  if (r >= 0) after(PWRITE, p);  return r;
}
ssize_t pwrite64(int fd, const void * b, size_t n, off64_t off) {
  char buf[512];  const char * p;  ssize_t r;  REAL(pwrite64);
  if (fd <= 2) return r_pwrite64(fd, b, n, off);
  p = fd_path(fd, buf, sizeof buf);
  if (before(PWRITE, p)) { errno = err_no;  return -1; }
  r = r_pwrite64(fd, b, n, off);  if (r >= 0) after(PWRITE, p);  return r;
}
ssize_t write(int fd, const void * b, size_t n) {
  char buf[512];  const char * p;  ssize_t r;  REAL(write);
  if (fd <= 2) return r_write(fd, b, n);
  p = fd_path(fd, buf, sizeof buf);
  if (before(WRITE, p)) { errno = err_no;  return -1; }
  r = r_write(fd, b, n);  if (r >= 0) after(WRITE, p);  return r;
}
ssize_t pread(int fd, void * b, size_t n, off_t off) {
  char buf[512];  const char * p;  ssize_t r;  REAL(pread);
  p = fd_path(fd, buf, sizeof buf);
  if (before(PREAD, p)) { errno = err_no;  return -1; }
  r = r_pread(fd, b, n, off);  if (r >= 0) after(PREAD, p);  return r;
}
ssize_t pread64(int fd, void * b, size_t n, off64_t off) {
  char buf[512];  const char * p;  ssize_t r;  REAL(pread64);
  p = fd_path(fd, buf, sizeof buf);
  if (before(PREAD, p)) { errno = err_no;  return -1; }
  r = r_pread64(fd, b, n, off);  if (r >= 0) after(PREAD, p);  return r;
}
ssize_t read(int fd, void * b, size_t n) {
  char buf[512];  const char * p;  ssize_t r;  REAL(read);
  if (fd <= 2) return r_read(fd, b, n);
  p = fd_path(fd, buf, sizeof buf);
  if (before(READ, p)) { errno = err_no;  return -1; }
  r = r_read(fd, b, n);  if (r >= 0) after(READ, p);  return r;
}
int chmod(const char * p, mode_t m) {
  int r;  REAL(chmod);
  if (before(CHMOD, p)) { errno = err_no;  return -1; }
  r = r_chmod(p, m);  if (r == 0) after(CHMOD, p);  return r;
}
int fchmod(int fd, mode_t m) {
  char buf[512];  const char * p = fd_path(fd, buf, sizeof buf);  int r;
  REAL(fchmod);
  if (before(CHMOD, p)) { errno = err_no;  return -1; }
  r = r_fchmod(fd, m);  if (r == 0) after(CHMOD, p);  return r;
}
int fchmodat(int d, const char * p, mode_t m, int f) {
  int r;  REAL(fchmodat);
  if (before(CHMOD, p)) { errno = err_no;  return -1; }
  r = r_fchmodat(d, p, m, f);  if (r == 0) after(CHMOD, p);  return r;
}
int utimensat(int d, const char * p, const struct timespec t[2], int f) {
  int r;  REAL(utimensat);
  if (before(UTIMES, p)) { errno = err_no;  return -1; }
  r = r_utimensat(d, p, t, f);  if (r == 0) after(UTIMES, p);  return r;
}
int chown(const char * p, uid_t u, gid_t g) {
  int r;  REAL(chown);
  if (before(CHOWN, p)) { errno = err_no;  return -1; }
  r = r_chown(p, u, g);  if (r == 0) after(CHOWN, p);  return r;
}
int fchownat(int d, const char * p, uid_t u, gid_t g, int f) {
  int r;  REAL(fchownat);
  if (before(CHOWN, p)) { errno = err_no;  return -1; }
  r = r_fchownat(d, p, u, g, f);  if (r == 0) after(CHOWN, p);  return r;
}
/*  Truncate matching files after mapping to trigger SIGBUS.  */
static void short_map(void * r, int fd) {
  const char * want;
  init();
  want = getenv("XPAR_FI_SHORT_MAP");
  want = getenv("XPAR_FI_SHORT_MAP");
  if (r != MAP_FAILED && fd >= 0 && want && *want) {
    char buf[512];
    const char * p = fd_path(fd, buf, sizeof buf);
    if (p && matches(p) && strstr(p, want)) {
      static const char m[] = "FI SHORT_MAP truncated\n";
      if (truncate(p, 0) == 0) real_write(2, m, sizeof m - 1);
    }
  }
}

void * mmap(void * addr, size_t len, int prot, int flags, int fd,
            off_t off) {
  void * r;
  REAL(mmap);
  r = r_mmap(addr, len, prot, flags, fd, off);
  short_map(r, fd);
  return r;
}

void * mmap64(void * addr, size_t len, int prot, int flags, int fd,
              off64_t off) {
  void * r;
  REAL(mmap64);
  r = r_mmap64(addr, len, prot, flags, fd, off);
  short_map(r, fd);
  return r;
}

int close(int fd) {
  char buf[512];  const char * p;  int r;  REAL(close);
  if (fd <= 2) return r_close(fd);
  p = fd_path(fd, buf, sizeof buf);
  if (!p || !matches(p)) return r_close(fd);
  if (before(CLOSE, p)) { r_close(fd);  errno = err_no;  return -1; }
  r = r_close(fd);  if (r == 0) after(CLOSE, p);  return r;
}
