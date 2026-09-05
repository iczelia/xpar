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
#include <dir.h>
#include <dos.h>
#include <dpmi.h>
#include <errno.h>
#include <fcntl.h>
#include <go32.h>
#include <io.h>
#include <libc/dosio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/farptr.h>
#include <sys/stat.h>
#include <unistd.h>

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

static int dos_error(unsigned int err) {
  if (!err) return 0;
  errno = __doserr_to_errno(err);
  return -1;
}

static bool leap_year(u32 y) {
  return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
}

/*  Keep DOS wall times independent of libc timezone handling.  */
static i64 dos_time_ns(u16 date, u16 time) {
  static const u8 month_days[12] = {
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
  };
  u32 year = 1980 + (date >> 9);
  u32 month = (date >> 5) & 15;
  u32 day = date & 31;
  u32 hour = time >> 11;
  u32 minute = (time >> 5) & 63;
  u32 second = (time & 31) * 2;
  u64 days = 0;
  u32 y, m;
  if (!month || month > 12 || !day ||
      day > (u32) month_days[month - 1] +
              (u32) (month == 2 && leap_year(year)) ||
      hour > 23 || minute > 59 || second > 59)
    return XPAR_TIME_NONE;
  for (y = 1970; y < year; y++) days += leap_year(y) ? 366 : 365;
  for (m = 1; m < month; m++)
    days += month_days[m - 1] + (m == 2 && leap_year(year));
  days += day - 1;
  return (i64) (days * 86400 + hour * 3600 + minute * 60 + second) *
         1000000000LL;
}

static bool dos_time_pack(i64 ns, u16 * date, u16 * time) {
  static const u8 month_days[12] = {
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
  };
  u64 sec, days;
  u32 year = 1970, month = 1, n;
  if (ns < 0) return false;
  sec = (u64) ns / 1000000000ULL;
  days = sec / 86400;
  sec %= 86400;
  while (year <= 2107 && days >= (leap_year(year) ? 366U : 365U)) {
    days -= leap_year(year) ? 366U : 365U;
    year++;
  }
  if (year < 1980 || year > 2107) return false;
  for (;;) {
    n = month_days[month - 1] + (month == 2 && leap_year(year));
    if (days < n) break;
    days -= n;
    month++;
  }
  *date = (u16) (((year - 1980) << 9) | (month << 5) | (days + 1));
  *time = (u16) (((sec / 3600) << 11) |
                 (((sec / 60) % 60) << 5) | ((sec % 60) / 2));
  return true;
}

int xpar_lstat(const char * path, xpar_stat_t * out) {
  struct _find_t found;
  unsigned int attr, rc;
  unsigned long size;
  unsigned short date, time;
  bool is_dir;
  rc = _dos_getfileattr(path, &attr);
  if (rc) return dos_error(rc);
  is_dir = (attr & DOS_A_DIR) != 0;
  size = 0;
  date = time = 0;
  if (!is_dir) {
    rc = _dos_findfirst(path,
                        DOS_A_RDONLY | DOS_A_HIDDEN | DOS_A_SYSTEM |
                        DOS_A_DIR | DOS_A_ARCHIVE, &found);
    if (rc) return dos_error(rc);
    size = found.size;
    date = found.wr_date;
    time = found.wr_time;
    while (_dos_findnext(&found) == 0) { }
  }
  out->size  = is_dir ? 0 : (u64) size;
  /*  FAT stores attributes, not a POSIX mode.  */
  out->mode  = XPAR_MODE_NONE;
  out->uid   = XPAR_ID_NONE;
  out->gid   = XPAR_ID_NONE;
  /*  DJGPP synthesizes dev and inode, so report no identity.  */
  out->dev   = 0;
  out->ino   = 0;
  out->nlink = 1;
  out->mtime_ns = date ? dos_time_ns(date, time) : XPAR_TIME_NONE;
  out->atime_ns = XPAR_TIME_NONE;
  out->ctime_ns = XPAR_TIME_NONE;
  out->btime_ns = XPAR_TIME_NONE;
  out->attrs      = attrs_of(attr);
  out->is_symlink = false;   /*  MS-DOS has none  */
  out->is_dir     = is_dir;
  out->is_regular = !out->is_dir;
  return 0;
}

u32 xpar_fs_caps(const char * path) {
  unsigned int attr;
  if (_dos_getfileattr(path, &attr) != 0) return 0;
  /*  FAT timestamps are two-second granular for write time and a date
      only for access time, so XPAR_FS_NSEC_TIME is clear; there is no
      creation-time setter, so XPAR_FS_BTIME is clear too.  */
  return XPAR_FS_FATATTR | XPAR_FS_NOFOLLOW;
}

struct xpar_dir {
  struct _find_t found;
  bool           ready;
  char           name[sizeof ((struct _find_t *) 0)->name];
  xpar_dirent ent;
};

xpar_dir * xpar_opendir(const char * path) {
  xpar_stat_t st;
  struct xpar_dir * h;
  char * pattern;
  unsigned int rc;
  sz path_len;
  if (xpar_lstat(path, &st) != 0) return NULL;
  if (!st.is_dir) { errno = ENOTDIR;  return NULL; }
  h = xpar_alloc_raw(sizeof *h);
  path_len = xpar_strlen(path);
  xpar_asprintf(&pattern, "%s%s*.*", path,
                path_len && (path[path_len - 1] == '/' ||
                             path[path_len - 1] == '\\') ? "" : "/");
  rc = _dos_findfirst(pattern,
                      DOS_A_RDONLY | DOS_A_HIDDEN | DOS_A_SYSTEM |
                      DOS_A_DIR | DOS_A_ARCHIVE, &h->found);
  xpar_free(pattern);
  h->ready = rc == 0;
  h->ent.name = NULL;
  h->ent.is_dir = h->ent.is_symlink = h->ent.is_regular = false;
  return h;
}

const xpar_dirent * xpar_readdir(xpar_dir * h) {
  while (h->ready) {
    unsigned char attr = h->found.attrib;
    sz i;
    xpar_snprintf(h->name, sizeof h->name, "%s", h->found.name);
    h->ready = _dos_findnext(&h->found) == 0;
    for (i = 0; h->name[i]; i++)
      if (h->name[i] >= 'A' && h->name[i] <= 'Z') h->name[i] += 'a' - 'A';
    if (h->name[0] == '.' &&
        (h->name[1] == '\0' ||
         (h->name[1] == '.' && h->name[2] == '\0'))) continue;
    h->ent.name = h->name;
    h->ent.is_symlink = false;
    h->ent.is_dir = (attr & DOS_A_DIR) != 0;
    h->ent.is_regular = !h->ent.is_dir && (attr & DOS_A_VOLUME) == 0;
    return &h->ent;
  }
  return NULL;
}

void xpar_closedir(xpar_dir * h) {
  if (!h) return;
  while (h->ready) h->ready = _dos_findnext(&h->found) == 0;
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

static int dos_path_call(u16 lfn_ax, u16 dos_ax, const char * path) {
  __dpmi_regs r;
  if (xpar_strlen(path) + 1 > __tb_size) { errno = ENAMETOOLONG;  return -1; }
  _put_path(path);
  memset(&r, 0, sizeof r);
  r.x.ax = _use_lfn(NULL) ? lfn_ax : dos_ax;
  r.x.ds = __tb_segment;
  r.x.dx = __tb_offset;
  __dpmi_int(0x21, &r);
  if (!(r.x.flags & 1)) return 0;
  errno = __doserr_to_errno(r.x.ax);
  return -1;
}

static int dos_rename(const char * from, const char * to) {
  __dpmi_regs r;
  sz fn = xpar_strlen(from) + 1, tn = xpar_strlen(to) + 1;
  if (fn + tn > __tb_size) { errno = ENAMETOOLONG;  return -1; }
  _put_path2(to, (int) fn);
  _put_path(from);
  memset(&r, 0, sizeof r);
  r.x.ax = _use_lfn(NULL) ? 0x7156 : 0x5600;
  r.x.ds = r.x.es = __tb_segment;
  r.x.dx = __tb_offset;
  r.x.di = (u16) fn;
  __dpmi_int(0x21, &r);
  if (!(r.x.flags & 1)) return 0;
  errno = __doserr_to_errno(r.x.ax);
  return -1;
}

int xpar_mkdir(const char * path, u32 mode) {
  unsigned int attr;
  (void) mode;
  if (dos_path_call(0x7139, 0x3900, path) == 0) return 0;
  if (_dos_getfileattr(path, &attr) == 0 && (attr & DOS_A_DIR))
    errno = EEXIST;
  return -1;
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
    if (xpar_mkdir(work, mode) != 0 && errno != EEXIST) rc = -1;
    work[i] = save;
  }
  xpar_free(work);
  return rc;
}

int xpar_rmdir (const char * path) {
  return dos_path_call(0x713a, 0x3a00, path);
}
int xpar_remove(const char * path) {
  return dos_path_call(0x7141, 0x4100, path);
}

int xpar_rename(const char * from, const char * to) {
  /*  DOS rename fails if the target exists, where POSIX replaces it. The
      remove first matches the POSIX behaviour every caller expects.  */
  (void) dos_path_call(0x7141, 0x4100, to);
  return dos_rename(from, to);
}

int xpar_set_times(const char * path, int nofollow,
                   i64 atime_ns, i64 mtime_ns, i64 btime_ns) {
  u16 date, time;
  int fd;
  unsigned int rc, close_rc;
  (void) nofollow;
  (void) btime_ns;
  if (mtime_ns == XPAR_TIME_NONE)
    return atime_ns == XPAR_TIME_NONE ? 0 : (errno = ENOTSUP, -1);
  if (!dos_time_pack(mtime_ns, &date, &time)) { errno = EOVERFLOW;  return -1; }
  rc = _dos_open(path, 0, &fd);
  if (rc) return dos_error(rc);
  rc = _dos_setftime(fd, date, time);
  close_rc = _dos_close(fd);
  if (rc) return dos_error(rc);
  if (close_rc) return dos_error(close_rc);
  if (atime_ns != XPAR_TIME_NONE) { errno = ENOTSUP;  return -1; }
  return 0;
}

int xpar_set_owner(const char * path, int nofollow, u32 uid, u32 gid,
                   const char * owner, const char * group) {
  (void) path;  (void) nofollow;  (void) uid;  (void) gid;
  (void) owner;  (void) group;
  errno = ENOTSUP;
  return -1;
}

int xpar_set_mode(const char * path, int nofollow, u32 mode) {
  /*  Read-only state is restored through attrs, not mode.  */
  (void) path;  (void) nofollow;  (void) mode;
  errno = ENOTSUP;
  return -1;
}

int xpar_set_attrs(const char * path, int nofollow, u16 attrs) {
  unsigned int cur, rc;
  (void) nofollow;
  rc = _dos_getfileattr(path, &cur);
  if (rc) return dos_error(rc);
  rc = _dos_setfileattr(path,
                        attrs_to_dos(attrs & XPAR_ATTR_SETTABLE, cur));
  return dos_error(rc);
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
  __dpmi_regs r;
  char dos_path[256];
  char * out;
  sz i, n;
  unsigned int drive;
  _dos_getdrive(&drive);
  memset(&r, 0, sizeof r);
  r.h.ah = 0x47;
  r.h.dl = 0;
  r.x.ds = (u16) (__tb >> 4);
  r.x.si = (u16) (__tb & 15);
  __dpmi_int(0x21, &r);
  if (r.x.flags & 1) { errno = __doserr_to_errno(r.x.ax);  return NULL; }
  dosmemget(__tb, sizeof dos_path, dos_path);
  dos_path[sizeof dos_path - 1] = 0;
  n = xpar_strlen(dos_path);
  out = xpar_alloc_raw(n + 4);
  out[0] = (char) ('a' + drive - 1);
  out[1] = ':';
  out[2] = '/';
  Fi(n,
    char c = dos_path[i];
    if (c == '\\') c = '/';
    if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    out[i + 3] = c);
  out[n + 3] = 0;
  return out;
}
