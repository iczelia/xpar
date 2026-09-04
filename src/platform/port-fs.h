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

/*  Filesystem metadata and no-follow setters.  */

#ifndef XPAR_PORT_FS_H
#define XPAR_PORT_FS_H

#include "port.h"

/*  Capabilities.  */

#define XPAR_FS_LINKID     (1u << 0)  /*  (dev, ino) is a real identity  */
#define XPAR_FS_HARDLINK   (1u << 1)  /*  links can be created here  */
#define XPAR_FS_OWNER      (1u << 2)  /*  uid, gid and names exist  */
#define XPAR_FS_XATTR      (1u << 3)
#define XPAR_FS_BTIME      (1u << 4)  /*  creation time reads AND sets  */
#define XPAR_FS_NSEC_TIME  (1u << 5)  /*  sub-second timestamps  */
#define XPAR_FS_FATATTR    (1u << 6)  /*  the attrs bit field is meaningful  */
#define XPAR_FS_NOFOLLOW   (1u << 7)  /*  every setter can skip a symlink  */

/*  Filesystem capabilities at `path`, or 0; query parents before creation.  */
u32 xpar_fs_caps(const char * path);

/*  Stat.  */
#define XPAR_TIME_NONE  INT64_MIN

/*  POSIX permission bits only; entry type is separate.  */
#define XPAR_MODE_SETUID  04000u
#define XPAR_MODE_SETGID  02000u
#define XPAR_MODE_STICKY  01000u
#define XPAR_MODE_PERM    07777u
#define XPAR_MODE_NONE    0xffffffffu   /*  host has no mode bits  */

#define XPAR_ID_NONE      0xffffffffu   /*  no uid/gid  */

#define XPAR_ATTR_READONLY    (1u << 0)
#define XPAR_ATTR_HIDDEN      (1u << 1)
#define XPAR_ATTR_SYSTEM      (1u << 2)
#define XPAR_ATTR_EXEC        (1u << 3)
#define XPAR_ATTR_RAWNAME     (1u << 4)
#define XPAR_ATTR_ARCHIVE     (1u << 5)
#define XPAR_ATTR_NOINDEX     (1u << 6)
#define XPAR_ATTR_SPARSE      (1u << 7)
#define XPAR_ATTR_COMPRESSED  (1u << 8)
#define XPAR_ATTR_ENCRYPTED   (1u << 9)
#define XPAR_ATTR_SETID       (1u << 10)

/*  Settable attributes; setters drop advisory bits.  */
#define XPAR_ATTR_SETTABLE                                                    \
  (XPAR_ATTR_READONLY | XPAR_ATTR_HIDDEN | XPAR_ATTR_SYSTEM |                 \
   XPAR_ATTR_ARCHIVE  | XPAR_ATTR_NOINDEX)

typedef struct {
  u64  size;
  u32  mode;                    /*  XPAR_MODE_NONE where there is none  */
  u32  uid, gid;                /*  XPAR_ID_NONE where there is none  */
  u64  dev, ino;                /*  an identity only under XPAR_FS_LINKID  */
  u64  nlink;                   /*  1 where the host does not count links  */
  i64  mtime_ns, atime_ns;
  i64  ctime_ns, btime_ns;      /*  XPAR_TIME_NONE when unavailable  */
  u16  attrs;                   /*  0 without XPAR_FS_FATATTR  */
  bool is_symlink, is_dir, is_regular;
} xpar_stat_t;

/*  lstat without following the final symlink.  */
int xpar_lstat(const char * path, xpar_stat_t * out);

/*  Directory iteration.  */

typedef struct xpar_dir xpar_dir;

typedef struct {
  const char * name;   /*  owned by the xpar_dir; dead after the next call  */
  bool is_dir, is_symlink, is_regular;
} xpar_dirent;

/*  Iterate entries excluding "." and ".."; types never follow links.  */
xpar_dir *          xpar_opendir (const char * path);
const xpar_dirent * xpar_readdir (xpar_dir * d);   /*  d non-NULL; NULL at end  */
void                xpar_closedir(xpar_dir * d);   /*  NULL is a no-op  */

/*  Links and directories.  */

/*  Read a NUL-terminated link target; fail if it does not fit.  */
i64 xpar_readlink(const char * path, char * buf, sz n);

int xpar_symlink(const char * target, const char * path);

int xpar_link(const char * existing, const char * newpath);

/*  The working directory as an absolute path, or NULL; caller frees.  */
char * xpar_getcwd(void);

int xpar_mkdir  (const char * path, u32 mode);
int xpar_mkdir_p(const char * path, u32 mode);  /*  existing path is OK  */
int xpar_rmdir  (const char * path);
int xpar_remove (const char * path);
int xpar_rename (const char * from, const char * to);

/*  Metadata setters fail when `nofollow` cannot be honoured.  */

/*  XPAR_TIME_NONE leaves a field unchanged. Unsupported btime is ignored.  */
int xpar_set_times(const char * path, int nofollow,
                   i64 atime_ns, i64 mtime_ns, i64 btime_ns);

int xpar_set_owner(const char * path, int nofollow, u32 uid, u32 gid,
                   const char * owner, const char * group);

int xpar_set_mode (const char * path, int nofollow, u32 mode);
int xpar_set_attrs(const char * path, int nofollow, u16 attrs);

/*  Xattrs return the required size; XPAR_FS_NOSIZE reports failure.  */

#define XPAR_FS_NOSIZE  ((sz) -1)

/*  NUL-separated names, matching listxattr.  */
sz xpar_listxattr(const char * path, int nofollow, char * buf, sz n);
sz xpar_getxattr (const char * path, int nofollow, const char * name,
                  void * buf, sz n);
int xpar_setxattr(const char * path, int nofollow, const char * name,
                  const void * val, sz n);

int xpar_uid_of  (const char * name, u32 * uid);
int xpar_gid_of  (const char * name, u32 * gid);
int xpar_name_of (u32 uid, char * buf, sz n);
int xpar_group_of(u32 gid, char * buf, sz n);

#endif
