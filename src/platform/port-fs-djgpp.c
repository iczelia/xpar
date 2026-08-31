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

/*  DJGPP filesystem metadata.  FAT attributes are supported; synthetic
    inode numbers are not trusted as link identities.  NOFOLLOW is safe
    because DOS has no symbolic links.  */

#if !defined(__DJGPP__)
#error "port-fs-djgpp.c compiled for a non-DJGPP target"
#endif

#include "common.h"
#include "port-fs.h"

#include <dirent.h>
#include <errno.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <utime.h>

/*  DJGPP does not define ENOTSUP.  */
#if !defined(ENOTSUP)
#define ENOTSUP ENOSYS
#endif

/*  The DOS attribute byte, int 21h AX=4300h. Bit 3 is the volume label and
    bit 4 the directory flag, neither of which is a settable attribute.  */
#define DOS_A_RDONLY  0x01
#define DOS_A_HIDDEN  0x02
#define DOS_A_SYSTEM  0x04
#define DOS_A_VOLUME  0x08
#define DOS_A_DIR     0x10
#define DOS_A_ARCHIVE 0x20

static u16 attrs_of(int a) {
  u16 r = 0;
  if (a & DOS_A_RDONLY)  r |= XPAR_ATTR_READONLY;
  if (a & DOS_A_HIDDEN)  r |= XPAR_ATTR_HIDDEN;
  if (a & DOS_A_SYSTEM)  r |= XPAR_ATTR_SYSTEM;
  if (a & DOS_A_ARCHIVE) r |= XPAR_ATTR_ARCHIVE;
  return r;
}

static int attrs_to_dos(u16 a, int current) {
  int w = current & (DOS_A_VOLUME | DOS_A_DIR);
  if (a & XPAR_ATTR_READONLY) w |= DOS_A_RDONLY;
  if (a & XPAR_ATTR_HIDDEN)   w |= DOS_A_HIDDEN;
  if (a & XPAR_ATTR_SYSTEM)   w |= DOS_A_SYSTEM;
  if (a & XPAR_ATTR_ARCHIVE)  w |= DOS_A_ARCHIVE;
  return w;
}

int xpar_lstat(const char * path, xpar_stat_t * out) {
  struct stat st;
  int a;
  if (stat(path, &st) != 0) return -1;
  out->size  = (u64) st.st_size;
  /* FAT stores attributes, not a POSIX mode. */
  out->mode  = XPAR_MODE_NONE;
  out->uid   = XPAR_ID_NONE;
  out->gid   = XPAR_ID_NONE;
  /* DJGPP synthesizes dev and inode, so report no identity. */
  out->dev   = 0;
  out->ino   = 0;
  out->nlink = 1;
  out->mtime_ns = (i64) st.st_mtime * 1000000000LL;
  out->atime_ns = (i64) st.st_atime * 1000000000LL;
  out->ctime_ns = XPAR_TIME_NONE;
  out->btime_ns = XPAR_TIME_NONE;
  a = _chmod(path, 0);
  out->attrs      = a < 0 ? 0 : attrs_of(a);
  out->is_symlink = false;   /*  MS-DOS has none  */
  out->is_dir     = S_ISDIR(st.st_mode) != 0;
  out->is_regular = !out->is_dir;
  return 0;
}

u32 xpar_fs_caps(const char * path) {
  struct stat st;
  if (stat(path, &st) != 0) return 0;
  /*  FAT timestamps are two-second granular for write time and a date
      only for access time, so XPAR_FS_NSEC_TIME is clear; there is no
      creation-time setter, so XPAR_FS_BTIME is clear too.  */
  return XPAR_FS_FATATTR | XPAR_FS_NOFOLLOW;
}

struct xpar_dir {
  DIR *       d;
  xpar_dirent ent;
};

xpar_dir * xpar_opendir(const char * path) {
  DIR * d = opendir(path);
  struct xpar_dir * h;
  if (!d) return NULL;
  h = xpar_alloc_raw(sizeof(*h));
  h->d = d;
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
    h->ent.is_symlink = false;
    h->ent.is_dir = de->d_type == DT_DIR;
    h->ent.is_regular = de->d_type == DT_REG;
    return &h->ent;
  }
}

void xpar_closedir(xpar_dir * h) {
  if (!h) return;
  closedir(h->d);
  xpar_free(h);
}

i64 xpar_readlink(const char * path, char * buf, sz n) {
  (void) path;  (void) buf;  (void) n;
  errno = ENOSYS;
  return -1;
}

int xpar_symlink(const char * target, const char * path) {
  (void) target;  (void) path;
  errno = ENOSYS;
  return -1;
}

int xpar_link(const char * existing, const char * newpath) {
  (void) existing;  (void) newpath;
  errno = ENOSYS;
  return -1;
}

int xpar_mkdir(const char * path, u32 mode) {
  (void) mode;
  return mkdir(path, 0777);
}

int xpar_mkdir_p(const char * path, u32 mode) {
  char * work = xpar_strdup(path);
  sz i, n = xpar_strlen(work);
  int rc = 0;
  (void) mode;
  for (i = 1; i <= n && rc == 0; i++) {
    char save;
    if (work[i] != '/' && work[i] != '\\' && work[i] != '\0') continue;
    /*  "C:\" is a drive root: there is nothing to create and mkdir would
        report EACCES.  */
    if (work[i - 1] == ':') continue;
    save = work[i];
    work[i] = '\0';
    if (mkdir(work, 0777) != 0 && errno != EEXIST) rc = -1;
    work[i] = save;
  }
  xpar_free(work);
  return rc;
}

int xpar_rmdir (const char * path) { return rmdir(path); }
int xpar_remove(const char * path) { return unlink(path); }

int xpar_rename(const char * from, const char * to) {
  /*  DOS rename fails if the target exists, where POSIX replaces it. The
      remove first matches the POSIX behaviour every caller expects.  */
  unlink(to);
  return rename(from, to);
}

int xpar_set_times(const char * path, int nofollow,
                   i64 atime_ns, i64 mtime_ns, i64 btime_ns) {
  struct utimbuf ut;
  xpar_stat_t sb;
  (void) nofollow;
  (void) btime_ns;   /*  FAT has no creation-time setter  */
  if (atime_ns == XPAR_TIME_NONE || mtime_ns == XPAR_TIME_NONE) {
    /*  utime writes both fields at once, so an omitted one is read back
        first. FAT quantises the result to two seconds either way.  */
    if (xpar_lstat(path, &sb) != 0) return -1;
    if (atime_ns == XPAR_TIME_NONE) atime_ns = sb.atime_ns;
    if (mtime_ns == XPAR_TIME_NONE) mtime_ns = sb.mtime_ns;
  }
  if (atime_ns == XPAR_TIME_NONE && mtime_ns == XPAR_TIME_NONE) return 0;
  ut.actime  = (time_t) (atime_ns / 1000000000LL);
  ut.modtime = (time_t) (mtime_ns / 1000000000LL);
  return utime(path, &ut);
}

int xpar_set_owner(const char * path, int nofollow, u32 uid, u32 gid,
                   const char * owner, const char * group) {
  (void) path;  (void) nofollow;  (void) uid;  (void) gid;
  (void) owner;  (void) group;
  errno = ENOTSUP;
  return -1;
}

int xpar_set_mode(const char * path, int nofollow, u32 mode) {
  /* Read-only state is restored through attrs, not mode. */
  (void) path;  (void) nofollow;  (void) mode;
  errno = ENOTSUP;
  return -1;
}

int xpar_set_attrs(const char * path, int nofollow, u16 attrs) {
  int cur = _chmod(path, 0);
  (void) nofollow;
  if (cur < 0) return -1;
  return _chmod(path, 1, attrs_to_dos(attrs & XPAR_ATTR_SETTABLE, cur)) < 0
         ? -1 : 0;
}

sz xpar_listxattr(const char * path, int nofollow, char * buf, sz n) {
  (void) path;  (void) nofollow;  (void) buf;  (void) n;
  errno = ENOTSUP;
  return XPAR_FS_NOSIZE;
}

sz xpar_getxattr(const char * path, int nofollow, const char * name,
                 void * buf, sz n) {
  (void) path;  (void) nofollow;  (void) name;  (void) buf;  (void) n;
  errno = ENOTSUP;
  return XPAR_FS_NOSIZE;
}

int xpar_setxattr(const char * path, int nofollow, const char * name,
                  const void * val, sz n) {
  (void) path;  (void) nofollow;  (void) name;  (void) val;  (void) n;
  errno = ENOTSUP;
  return -1;
}

int xpar_uid_of(const char * name, u32 * uid) {
  (void) name;  (void) uid;  return -1;
}
int xpar_gid_of(const char * name, u32 * gid) {
  (void) name;  (void) gid;  return -1;
}
int xpar_name_of(u32 uid, char * buf, sz n) {
  (void) uid;  (void) buf;  (void) n;  return -1;
}
int xpar_group_of(u32 gid, char * buf, sz n) {
  (void) gid;  (void) buf;  (void) n;  return -1;
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
