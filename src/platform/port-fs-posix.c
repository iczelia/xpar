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

/*  POSIX filesystem metadata using only no-follow operations.  A missing
    safe primitive disables that capability; setters never fall back to a
    symlink-following variant.  */

#include "common.h"
#include "port-fs.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#if defined(HAVE_SYS_TIME_H)
  #include <sys/time.h>
#endif
#if defined(HAVE_PWD_H)
  #include <pwd.h>
#endif
#if defined(HAVE_GRP_H)
  #include <grp.h>
#endif
#if defined(HAVE_SYS_XATTR_H)
  #include <sys/xattr.h>
#endif
#if defined(__linux__)
  #include <sys/vfs.h>
#endif

/*  Apple and POSIX use different names for nanosecond timestamps.  */
#if defined(__APPLE__)
  #define XPAR_MTIM(st) ((st).st_mtimespec)
  #define XPAR_ATIM(st) ((st).st_atimespec)
  #define XPAR_CTIM(st) ((st).st_ctimespec)
  #define XPAR_NSEC_STAT 1
#elif defined(st_mtime)
  #define XPAR_MTIM(st) ((st).st_mtim)
  #define XPAR_ATIM(st) ((st).st_atim)
  #define XPAR_CTIM(st) ((st).st_ctim)
  #define XPAR_NSEC_STAT 1
#endif

/*  Enable no-follow metadata setters only when every primitive is safe.  */
#if defined(HAVE_UTIMENSAT) && defined(HAVE_AT_SYMLINK_NOFOLLOW) &&          \
    defined(HAVE_OPENAT) && defined(O_DIRECTORY) && defined(O_NOFOLLOW)
  #define XPAR_NOFOLLOW_TIMES 1
#endif
#if defined(HAVE_FCHOWNAT) && defined(HAVE_AT_SYMLINK_NOFOLLOW) &&           \
    defined(HAVE_OPENAT) && defined(O_DIRECTORY) && defined(O_NOFOLLOW)
  #define XPAR_NOFOLLOW_OWNER 1
#endif
#if defined(XPAR_NOFOLLOW_TIMES) && defined(XPAR_NOFOLLOW_OWNER) &&           \
    defined(HAVE_FCHMOD)
  #define XPAR_NOFOLLOW_ALL 1
#endif

#if defined(HAVE_LSETXATTR) && defined(HAVE_LGETXATTR) &&                     \
    defined(HAVE_LLISTXATTR)
  #define XPAR_XATTR_L 1
#elif defined(__APPLE__) && defined(HAVE_SYS_XATTR_H)
  #define XPAR_XATTR_APPLE 1
#endif

static i64 sec_ns(time_t s, long ns) {
  return (i64) s * 1000000000LL + (i64) ns;
}

int xpar_lstat(const char * path, xpar_stat_t * out) {
  struct stat st;
  if (lstat(path, &st) != 0) return -1;
  out->size  = (u64) st.st_size;
  out->mode  = (u32) (st.st_mode & XPAR_MODE_PERM);
  out->uid   = (u32) st.st_uid;
  out->gid   = (u32) st.st_gid;
  out->dev   = (u64) st.st_dev;
  out->ino   = (u64) st.st_ino;
  out->nlink = (u64) st.st_nlink;
#if defined(XPAR_NSEC_STAT)
  out->mtime_ns = sec_ns(st.st_mtime, XPAR_MTIM(st).tv_nsec);
  out->atime_ns = sec_ns(st.st_atime, XPAR_ATIM(st).tv_nsec);
  out->ctime_ns = sec_ns(st.st_ctime, XPAR_CTIM(st).tv_nsec);
#else
  out->mtime_ns = sec_ns(st.st_mtime, 0);
  out->atime_ns = sec_ns(st.st_atime, 0);
  out->ctime_ns = sec_ns(st.st_ctime, 0);
#endif
  out->btime_ns = XPAR_TIME_NONE;
#if defined(__APPLE__)
  out->btime_ns = sec_ns(st.st_birthtimespec.tv_sec,
                         st.st_birthtimespec.tv_nsec);
#elif defined(__linux__) && defined(STATX_BTIME)
  { struct statx stx;
    if (statx(AT_FDCWD, path, AT_SYMLINK_NOFOLLOW | AT_STATX_SYNC_AS_STAT,
              STATX_BTIME, &stx) == 0 && (stx.stx_mask & STATX_BTIME))
      out->btime_ns = sec_ns((time_t) stx.stx_btime.tv_sec,
                             (long) stx.stx_btime.tv_nsec); }
#endif
  out->attrs      = 0;
  out->is_symlink = S_ISLNK (st.st_mode) != 0;
  out->is_dir     = S_ISDIR (st.st_mode) != 0;
  out->is_regular = S_ISREG (st.st_mode) != 0;
  return 0;
}

/*  FAT filesystem magics from Linux UAPI.  */
#define XPAR_MSDOS_MAGIC  0x4d44UL
#define XPAR_EXFAT_MAGIC  0x2011BAB0UL

u32 xpar_fs_caps(const char * path) {
  struct stat st;
  u32 c;
  if (lstat(path, &st) != 0) return 0;

  c = XPAR_FS_LINKID | XPAR_FS_HARDLINK | XPAR_FS_OWNER;
#if defined(XPAR_NSEC_STAT)
  c |= XPAR_FS_NSEC_TIME;
#endif
#if defined(XPAR_NOFOLLOW_ALL)
  c |= XPAR_FS_NOFOLLOW;
#endif

#if defined(__linux__)
  { struct statfs sfs;
    if (statfs(path, &sfs) == 0) {
      /* Accommodate signed and unsigned f_type declarations. */
      unsigned long t = (unsigned long) sfs.f_type;
      if (t == XPAR_MSDOS_MAGIC || t == XPAR_EXFAT_MAGIC)
        c &= ~(u32) (XPAR_FS_LINKID | XPAR_FS_HARDLINK | XPAR_FS_OWNER |
                     XPAR_FS_NSEC_TIME);
    }
  }
#endif

#if defined(XPAR_XATTR_L)
  if (llistxattr(path, NULL, 0) >= 0) c |= XPAR_FS_XATTR;
#elif defined(XPAR_XATTR_APPLE)
  if (listxattr(path, NULL, 0, XATTR_NOFOLLOW) >= 0) c |= XPAR_FS_XATTR;
#endif

  return c;
}

struct xpar_dir {
  DIR *       d;
  char *      path;
  xpar_dirent ent;
};

xpar_dir * xpar_opendir(const char * path) {
  DIR * d = opendir(path);
  struct xpar_dir * h;
  if (!d) return NULL;
  h = xpar_alloc_raw(sizeof(*h));
  h->d = d;
  h->path = xpar_strdup(path);
  h->ent.name = NULL;
  h->ent.is_dir = h->ent.is_symlink = h->ent.is_regular = false;
  return h;
}

const xpar_dirent * xpar_readdir(xpar_dir * h) {
  for (;;) {
    struct dirent * de = readdir(h->d);
    if (!de) return NULL;
    if (de->d_name[0] == '.' &&
        (de->d_name[1] == '\0' ||
         (de->d_name[1] == '.' && de->d_name[2] == '\0'))) continue;
    h->ent.name = de->d_name;
    h->ent.is_dir = h->ent.is_symlink = h->ent.is_regular = false;
#if defined(DT_DIR)
    if (de->d_type != DT_UNKNOWN) {
      h->ent.is_dir     = de->d_type == DT_DIR;
      h->ent.is_symlink = de->d_type == DT_LNK;
      h->ent.is_regular = de->d_type == DT_REG;
      return &h->ent;
    }
#endif
    /*  No d_type, or the filesystem declines to fill it in. lstat and not
        stat: the caller is building a manifest entry for the name that is
        here, not for whatever it points at.  */
    { char * full = NULL;
      xpar_stat_t sb;
      xpar_asprintf(&full, "%s/%s", h->path, de->d_name);
      if (full && xpar_lstat(full, &sb) == 0) {
        h->ent.is_dir     = sb.is_dir;
        h->ent.is_symlink = sb.is_symlink;
        h->ent.is_regular = sb.is_regular;
      }
      xpar_free(full); }
    return &h->ent;
  }
}

void xpar_closedir(xpar_dir * h) {
  if (!h) return;
  closedir(h->d);
  xpar_free(h->path);
  xpar_free(h);
}

#if defined(HAVE_OPENAT) && defined(O_DIRECTORY) && defined(O_NOFOLLOW)
static int parent_open(const char * path, char ** storage,
                       const char ** leaf) {
  char * work = xpar_strdup(path), * p = work, * slash;
  int dfd = open(path[0] == '/' ? "/" : ".",
                 O_RDONLY | O_DIRECTORY | O_NOFOLLOW
#if defined(O_CLOEXEC)
                 | O_CLOEXEC
#endif
                 );
  if (dfd < 0) { xpar_free(work);  return -1; }
  while (*p == '/') p++;
  if (!*p) { close(dfd);  xpar_free(work);  errno = EINVAL;  return -1; }
  while ((slash = strchr(p, '/')) != NULL) {
    int next;
    *slash = '\0';
    if (*p && strcmp(p, ".") != 0) {
      next = openat(dfd, p, O_RDONLY | O_DIRECTORY | O_NOFOLLOW
#if defined(O_CLOEXEC)
                    | O_CLOEXEC
#endif
                    );
      if (next < 0) {
        int saved = errno;
        close(dfd);  xpar_free(work);  errno = saved;
        return -1;
      }
      close(dfd);  dfd = next;
    }
    p = slash + 1;
    while (*p == '/') p++;
    if (!*p) { close(dfd);  xpar_free(work);  errno = EINVAL;  return -1; }
  }
  *storage = work;  *leaf = p;
  return dfd;
}

static int at_done(int dfd, char * storage, int rc) {
  int saved = errno;
  close(dfd);  xpar_free(storage);  errno = saved;
  return rc;
}
#endif

i64 xpar_readlink(const char * path, char * buf, sz n) {
#if defined(HAVE_READLINK)
  ssize_t r;
  if (n == 0) { errno = EINVAL;  return -1; }
  r = readlink(path, buf, n - 1);
  if (r < 0) return -1;
  /*  A filled buffer is indistinguishable from a truncated target, and a
      truncated target is a wrong path rather than a short one.  */
  if ((sz) r == n - 1) { errno = ENAMETOOLONG;  return -1; }
  buf[r] = '\0';
  return (i64) r;
#else
  (void) path;  (void) buf;  (void) n;
  errno = ENOSYS;  return -1;
#endif
}

int xpar_symlink(const char * target, const char * path) {
#if defined(HAVE_SYMLINKAT) && defined(HAVE_OPENAT) && \
    defined(O_DIRECTORY) && defined(O_NOFOLLOW)
  char * storage;
  const char * leaf;
  int dfd = parent_open(path, &storage, &leaf);
  if (dfd < 0) return -1;
  return at_done(dfd, storage, symlinkat(target, dfd, leaf));
#elif defined(HAVE_SYMLINK)
  return symlink(target, path);
#else
  (void) target;  (void) path;
  errno = ENOSYS;  return -1;
#endif
}

int xpar_link(const char * existing, const char * newpath) {
#if defined(HAVE_LINKAT) && defined(HAVE_OPENAT) && defined(O_DIRECTORY) && \
    defined(O_NOFOLLOW)
  char * old_storage, * new_storage;
  const char * old_leaf, * new_leaf;
  int oldfd = parent_open(existing, &old_storage, &old_leaf), newfd, rc;
  if (oldfd < 0) return -1;
  newfd = parent_open(newpath, &new_storage, &new_leaf);
  if (newfd < 0) return at_done(oldfd, old_storage, -1);
  rc = linkat(oldfd, old_leaf, newfd, new_leaf, 0);
  at_done(newfd, new_storage, rc);
  return at_done(oldfd, old_storage, rc);
#elif defined(HAVE_LINK)
  return link(existing, newpath);
#else
  (void) existing;  (void) newpath;
  errno = ENOSYS;  return -1;
#endif
}

int xpar_mkdir(const char * path, u32 mode) {
#if defined(HAVE_MKDIRAT) && defined(HAVE_OPENAT) && defined(O_DIRECTORY) && \
    defined(O_NOFOLLOW)
  char * storage;
  const char * leaf;
  int dfd = parent_open(path, &storage, &leaf);
  if (dfd < 0) return -1;
  return at_done(dfd, storage,
                 mkdirat(dfd, leaf, (mode_t) (mode & XPAR_MODE_PERM)));
#else
  return mkdir(path, (mode_t) (mode & XPAR_MODE_PERM));
#endif
}

int xpar_rmdir(const char * path) {
#if defined(HAVE_UNLINKAT) && defined(HAVE_OPENAT) && defined(O_DIRECTORY) && \
    defined(O_NOFOLLOW)
  char * storage;
  const char * leaf;
  int dfd = parent_open(path, &storage, &leaf);
  if (dfd < 0) return -1;
  return at_done(dfd, storage, unlinkat(dfd, leaf, AT_REMOVEDIR));
#else
  return rmdir(path);
#endif
}

int xpar_remove(const char * path) {
#if defined(HAVE_UNLINKAT) && defined(HAVE_OPENAT) && defined(O_DIRECTORY) && \
    defined(O_NOFOLLOW)
  char * storage;
  const char * leaf;
  int dfd = parent_open(path, &storage, &leaf);
  if (dfd < 0) return -1;
  return at_done(dfd, storage, unlinkat(dfd, leaf, 0));
#else
  return unlink(path);
#endif
}

int xpar_rename(const char * from, const char * to) {
#if defined(HAVE_RENAMEAT) && defined(HAVE_OPENAT) && \
    defined(O_DIRECTORY) && defined(O_NOFOLLOW)
  char * from_storage, * to_storage;
  const char * from_leaf, * to_leaf;
  int fromfd = parent_open(from, &from_storage, &from_leaf), tofd, rc;
  if (fromfd < 0) return -1;
  tofd = parent_open(to, &to_storage, &to_leaf);
  if (tofd < 0) return at_done(fromfd, from_storage, -1);
  rc = renameat(fromfd, from_leaf, tofd, to_leaf);
  at_done(tofd, to_storage, rc);
  return at_done(fromfd, from_storage, rc);
#else
  return rename(from, to);
#endif
}

int xpar_mkdir_p(const char * path, u32 mode) {
#if defined(HAVE_OPENAT) && defined(HAVE_MKDIRAT) && defined(O_DIRECTORY) && \
    defined(O_NOFOLLOW)
  char * work = xpar_strdup(path), * p = work, * slash;
  int dfd, rc = 0, saved = 0;
  dfd = open(path[0] == '/' ? "/" : ".",
             O_RDONLY | O_DIRECTORY | O_NOFOLLOW
#if defined(O_CLOEXEC)
             | O_CLOEXEC
#endif
             );
  if (dfd < 0) { xpar_free(work);  return -1; }
  while (*p == '/') p++;
  while (*p && rc == 0) {
    int next;
    slash = strchr(p, '/');
    if (slash) *slash = '\0';
    if (*p && strcmp(p, ".") != 0) {
      next = openat(dfd, p, O_RDONLY | O_DIRECTORY | O_NOFOLLOW
#if defined(O_CLOEXEC)
                    | O_CLOEXEC
#endif
                    );
      if (next < 0 && errno == ENOENT &&
          mkdirat(dfd, p, (mode_t) (mode & XPAR_MODE_PERM)) == 0)
        next = openat(dfd, p, O_RDONLY | O_DIRECTORY | O_NOFOLLOW
#if defined(O_CLOEXEC)
                      | O_CLOEXEC
#endif
                      );
      if (next < 0) { rc = -1;  saved = errno; }
      else { close(dfd);  dfd = next; }
    }
    if (!slash) break;
    p = slash + 1;
    while (*p == '/') p++;
  }
  close(dfd);  xpar_free(work);
  if (rc) errno = saved;
  return rc;
#else
  char * work = xpar_strdup(path);
  sz i, n = xpar_strlen(work);
  int rc = 0;
  for (i = 1; i <= n && rc == 0; i++) {
    if (work[i] != '/' && work[i] != '\0') continue;
    { char save = work[i];
      struct stat st;
      work[i] = '\0';
      if (lstat(work, &st) == 0) {
        if (!S_ISDIR(st.st_mode) || S_ISLNK(st.st_mode)) { errno = ENOTDIR;  rc = -1; }
      } else if (mkdir(work, (mode_t) (mode & XPAR_MODE_PERM)) != 0 &&
                 errno != EEXIST) {
        rc = -1;
      }
      work[i] = save; }
  }
  xpar_free(work);
  return rc;
#endif
}

int xpar_set_times(const char * path, int nofollow,
                   i64 atime_ns, i64 mtime_ns, i64 btime_ns) {
  (void) btime_ns;
  if (atime_ns == XPAR_TIME_NONE && mtime_ns == XPAR_TIME_NONE) return 0;

#if defined(HAVE_UTIMENSAT)
  { struct timespec ts[2];
    if (atime_ns == XPAR_TIME_NONE) {
      ts[0].tv_sec = 0;  ts[0].tv_nsec = UTIME_OMIT;
    } else {
      ts[0].tv_sec  = (time_t) (atime_ns / 1000000000LL);
      ts[0].tv_nsec = (long)   (atime_ns % 1000000000LL);
      if (ts[0].tv_nsec < 0) { ts[0].tv_nsec += 1000000000L; ts[0].tv_sec--; }
    }
    if (mtime_ns == XPAR_TIME_NONE) {
      ts[1].tv_sec = 0;  ts[1].tv_nsec = UTIME_OMIT;
    } else {
      ts[1].tv_sec  = (time_t) (mtime_ns / 1000000000LL);
      ts[1].tv_nsec = (long)   (mtime_ns % 1000000000LL);
      if (ts[1].tv_nsec < 0) { ts[1].tv_nsec += 1000000000L; ts[1].tv_sec--; }
    }
  #if defined(HAVE_AT_SYMLINK_NOFOLLOW)
    if (nofollow) {
  #if defined(HAVE_OPENAT) && defined(O_DIRECTORY) && defined(O_NOFOLLOW)
      char * storage;
      const char * leaf;
      int dfd = parent_open(path, &storage, &leaf);
      if (dfd < 0) return -1;
      return at_done(dfd, storage,
                     utimensat(dfd, leaf, ts, AT_SYMLINK_NOFOLLOW));
  #else
      errno = ENOTSUP;  return -1;
  #endif
    }
    return utimensat(AT_FDCWD, path, ts, 0);
  #else
    if (nofollow) { errno = ENOSYS;  return -1; }
    return utimensat(AT_FDCWD, path, ts, 0);
  #endif
  }
#elif defined(HAVE_LUTIMES)
  { struct timeval tv[2];
    i64 a = atime_ns, m = mtime_ns;
    if (a == XPAR_TIME_NONE || m == XPAR_TIME_NONE) {
      xpar_stat_t sb;
      if (xpar_lstat(path, &sb) != 0) return -1;
      if (a == XPAR_TIME_NONE) a = sb.atime_ns;
      if (m == XPAR_TIME_NONE) m = sb.mtime_ns;
    }
    tv[0].tv_sec  = (time_t) (a / 1000000000LL);
    tv[0].tv_usec = (long)   ((a % 1000000000LL) / 1000);
    tv[1].tv_sec  = (time_t) (m / 1000000000LL);
    tv[1].tv_usec = (long)   ((m % 1000000000LL) / 1000);
    if (nofollow) return lutimes(path, tv);
    return utimes(path, tv);
  }
#else
  (void) path;  (void) nofollow;
  errno = ENOSYS;  return -1;
#endif
}

int xpar_set_owner(const char * path, int nofollow, u32 uid, u32 gid,
                   const char * owner, const char * group) {
  u32 u = uid, g = gid;
  if (owner && *owner) { u32 t;  if (xpar_uid_of(owner, &t) == 0) u = t; }
  if (group && *group) { u32 t;  if (xpar_gid_of(group, &t) == 0) g = t; }
  if (u == XPAR_ID_NONE && g == XPAR_ID_NONE) return 0;
  { uid_t su = u == XPAR_ID_NONE ? (uid_t) -1 : (uid_t) u;
    gid_t sg = g == XPAR_ID_NONE ? (gid_t) -1 : (gid_t) g;
    if (!nofollow) return chown(path, su, sg);
#if defined(HAVE_FCHOWNAT) && defined(HAVE_OPENAT) && \
    defined(O_DIRECTORY) && defined(O_NOFOLLOW) && \
    defined(HAVE_AT_SYMLINK_NOFOLLOW)
    { char * storage;
      const char * leaf;
      int dfd = parent_open(path, &storage, &leaf);
      if (dfd < 0) return -1;
      return at_done(dfd, storage,
                     fchownat(dfd, leaf, su, sg, AT_SYMLINK_NOFOLLOW)); }
#else
    errno = ENOTSUP;  return -1;
#endif
  }
}

int xpar_set_mode(const char * path, int nofollow, u32 mode) {
  mode_t m = (mode_t) (mode & XPAR_MODE_PERM);
  int fd, r;
  if (!nofollow) return chmod(path, m);

#if defined(HAVE_FCHMOD) && defined(HAVE_OPENAT) && defined(O_DIRECTORY) && \
    defined(O_NOFOLLOW)
  { char * storage;
    const char * leaf;
    int dfd = parent_open(path, &storage, &leaf);
    if (dfd < 0) return -1;
    fd = openat(dfd, leaf, O_RDONLY | O_NOFOLLOW | O_NONBLOCK
  #if defined(O_CLOEXEC)
                | O_CLOEXEC
  #endif
                );
    if (fd < 0) return at_done(dfd, storage, -1);
    r = fchmod(fd, m);
    close(fd);
    return at_done(dfd, storage, r); }
#else
  (void) fd;  (void) r;
  errno = ENOTSUP;  return -1;
#endif
}

int xpar_set_attrs(const char * path, int nofollow, u16 attrs) {
  (void) path;  (void) nofollow;  (void) attrs;
  errno = ENOTSUP;
  return -1;
}

sz xpar_listxattr(const char * path, int nofollow, char * buf, sz n) {
#if defined(XPAR_XATTR_L)
  ssize_t r;
  if (!nofollow) r = listxattr(path, buf, n);
  else           r = llistxattr(path, buf, n);
  if (r < 0 && errno == ERANGE) {
    r = nofollow ? llistxattr(path, NULL, 0) : listxattr(path, NULL, 0);
    if (r < 0) return XPAR_FS_NOSIZE;
    return (sz) r;
  }
  return r < 0 ? XPAR_FS_NOSIZE : (sz) r;
#elif defined(XPAR_XATTR_APPLE)
  ssize_t r = listxattr(path, buf, n, nofollow ? XATTR_NOFOLLOW : 0);
  if (r < 0 && errno == ERANGE)
    r = listxattr(path, NULL, 0, nofollow ? XATTR_NOFOLLOW : 0);
  return r < 0 ? XPAR_FS_NOSIZE : (sz) r;
#else
  (void) path;  (void) nofollow;  (void) buf;  (void) n;
  errno = ENOTSUP;
  return XPAR_FS_NOSIZE;
#endif
}

sz xpar_getxattr(const char * path, int nofollow, const char * name,
                 void * buf, sz n) {
#if defined(XPAR_XATTR_L)
  ssize_t r;
  if (!nofollow) r = getxattr(path, name, buf, n);
  else           r = lgetxattr(path, name, buf, n);
  if (r < 0 && errno == ERANGE) {
    r = nofollow ? lgetxattr(path, name, NULL, 0)
                 : getxattr(path, name, NULL, 0);
    if (r < 0) return XPAR_FS_NOSIZE;
    return (sz) r;
  }
  return r < 0 ? XPAR_FS_NOSIZE : (sz) r;
#elif defined(XPAR_XATTR_APPLE)
  ssize_t r = getxattr(path, name, buf, n, 0,
                       nofollow ? XATTR_NOFOLLOW : 0);
  if (r < 0 && errno == ERANGE)
    r = getxattr(path, name, NULL, 0, 0, nofollow ? XATTR_NOFOLLOW : 0);
  return r < 0 ? XPAR_FS_NOSIZE : (sz) r;
#else
  (void) path;  (void) nofollow;  (void) name;  (void) buf;  (void) n;
  errno = ENOTSUP;
  return XPAR_FS_NOSIZE;
#endif
}

int xpar_setxattr(const char * path, int nofollow, const char * name,
                  const void * val, sz n) {
#if defined(XPAR_XATTR_L)
  if (!nofollow) return setxattr(path, name, val, n, 0);
#if defined(HAVE_FSETXATTR) && defined(HAVE_OPENAT) && \
    defined(O_DIRECTORY) && defined(O_NOFOLLOW)
  { char * storage;
    const char * leaf;
    int dfd = parent_open(path, &storage, &leaf);
    int fd, rc;
    if (dfd < 0) return -1;
    fd = openat(dfd, leaf, O_RDONLY | O_NOFOLLOW | O_NONBLOCK
  #if defined(O_CLOEXEC)
                | O_CLOEXEC
  #endif
                );
    if (fd < 0) return at_done(dfd, storage, -1);
    rc = fsetxattr(fd, name, val, n, 0);
    close(fd);
    return at_done(dfd, storage, rc); }
#else
  errno = ENOTSUP;  return -1;
#endif
#elif defined(XPAR_XATTR_APPLE)
  return setxattr(path, name, val, n, 0, nofollow ? XATTR_NOFOLLOW : 0);
#else
  (void) path;  (void) nofollow;  (void) name;  (void) val;  (void) n;
  errno = ENOTSUP;  return -1;
#endif
}

#define XPAR_PW_BUF 1024

int xpar_uid_of(const char * name, u32 * uid) {
#if defined(HAVE_GETPWNAM_R) && defined(HAVE_PWD_H)
  struct passwd pw, * res = NULL;
  char buf[XPAR_PW_BUF];
  if (getpwnam_r(name, &pw, buf, sizeof buf, &res) != 0 || !res) return -1;
  *uid = (u32) pw.pw_uid;
  return 0;
#else
  (void) name;  (void) uid;  return -1;
#endif
}

int xpar_gid_of(const char * name, u32 * gid) {
#if defined(HAVE_GETGRNAM_R) && defined(HAVE_GRP_H)
  struct group gr, * res = NULL;
  char buf[XPAR_PW_BUF];
  if (getgrnam_r(name, &gr, buf, sizeof buf, &res) != 0 || !res) return -1;
  *gid = (u32) gr.gr_gid;
  return 0;
#else
  (void) name;  (void) gid;  return -1;
#endif
}

int xpar_name_of(u32 uid, char * buf, sz n) {
#if defined(HAVE_GETPWUID_R) && defined(HAVE_PWD_H)
  struct passwd pw, * res = NULL;
  char tmp[XPAR_PW_BUF];
  sz len;
  if (getpwuid_r((uid_t) uid, &pw, tmp, sizeof tmp, &res) != 0 || !res)
    return -1;
  len = xpar_strlen(pw.pw_name);
  if (len + 1 > n) return -1;
  xpar_memcpy(buf, pw.pw_name, len + 1);
  return 0;
#else
  (void) uid;  (void) buf;  (void) n;  return -1;
#endif
}

int xpar_group_of(u32 gid, char * buf, sz n) {
#if defined(HAVE_GETGRGID_R) && defined(HAVE_GRP_H)
  struct group gr, * res = NULL;
  char tmp[XPAR_PW_BUF];
  sz len;
  if (getgrgid_r((gid_t) gid, &gr, tmp, sizeof tmp, &res) != 0 || !res)
    return -1;
  len = xpar_strlen(gr.gr_name);
  if (len + 1 > n) return -1;
  xpar_memcpy(buf, gr.gr_name, len + 1);
  return 0;
#else
  (void) gid;  (void) buf;  (void) n;  return -1;
#endif
}

char * xpar_getcwd(void) {
  sz n = 512;
  for (;;) {
    char * p = (char *) xpar_alloc_raw(n);
    if (getcwd(p, n)) return p;
    xpar_free(p);
    if (errno != ERANGE || n > ((sz) 1 << 20)) return NULL;
    n *= 2;
  }
}
