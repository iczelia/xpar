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

/*  Win32 filesystem metadata.  NTFS provides stable link identities,
    hard links, handle-based times, and no-follow attributes.  Unsupported
    ownership, xattrs, and POSIX modes are never invented.  */

#if !(defined(_WIN32) || defined(__MINGW32__) || defined(__MINGW64__))
#error "port-fs-win32.c compiled for a non-Windows target"
#endif

#if !defined(_WIN32_WINNT)
  #define _WIN32_WINNT 0x0600
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "common.h"
#include "port-fs.h"
#include "port-win-path.h"


#if !defined(FILE_ATTRIBUTE_NOT_CONTENT_INDEXED)
  #define FILE_ATTRIBUTE_NOT_CONTENT_INDEXED 0x00002000
#endif
#if !defined(FILE_FLAG_OPEN_REPARSE_POINT)
  #define FILE_FLAG_OPEN_REPARSE_POINT 0x00200000
#endif
#if !defined(IO_REPARSE_TAG_SYMLINK)
  #define IO_REPARSE_TAG_SYMLINK 0xA000000CUL
#endif
#if !defined(IO_REPARSE_TAG_MOUNT_POINT)
  #define IO_REPARSE_TAG_MOUNT_POINT 0xA0000003UL
#endif
#if !defined(FSCTL_GET_REPARSE_POINT)
  #define FSCTL_GET_REPARSE_POINT 0x000900A8UL
#endif

static void fail_unsupported(void) { SetLastError(ERROR_NOT_SUPPORTED); }

#if defined(XPAR_WIN_LEGACY)
  typedef char    xchar;
#else
  typedef wchar_t xchar;
#endif

/*  Returns a heap block the caller frees with xpar_free.  */
#if !defined(XPAR_WIN_LEGACY)
static wchar_t * path_text(const char * s) { return xpar_win_wide(s); }
static xchar   * path_conv(const char * s) { return xpar_win_path(s); }
static char * path_back(const wchar_t * w, int wlen) {
  return xpar_win_utf8(w, wlen);
}
#else
static xchar * path_conv(const char * s) { return xpar_strdup(s); }
#endif

static i64 ft_ns(FILETIME ft) {
  u64 t = ((u64) ft.dwHighDateTime << 32) | (u64) ft.dwLowDateTime;
  if (t == 0) return XPAR_TIME_NONE;
  /*  Times before 1970 are legal on NTFS and go negative here, which is
      what the container stores too, so the subtraction is signed.  */
  return ((i64) t - (i64) WIN_EPOCH_DELTA_100NS) * 100;
}

static FILETIME ns_ft(i64 ns) {
  FILETIME ft;
  u64 t = (u64) (ns / 100 + (i64) WIN_EPOCH_DELTA_100NS);
  ft.dwLowDateTime  = (DWORD) (t & 0xFFFFFFFFu);
  ft.dwHighDateTime = (DWORD) (t >> 32);
  return ft;
}

static u16 attrs_of(DWORD a) {
  u16 r = 0;
  if (a & FILE_ATTRIBUTE_READONLY)  r |= XPAR_ATTR_READONLY;
  if (a & FILE_ATTRIBUTE_HIDDEN)    r |= XPAR_ATTR_HIDDEN;
  if (a & FILE_ATTRIBUTE_SYSTEM)    r |= XPAR_ATTR_SYSTEM;
  if (a & FILE_ATTRIBUTE_ARCHIVE)   r |= XPAR_ATTR_ARCHIVE;
  if (a & FILE_ATTRIBUTE_NOT_CONTENT_INDEXED) r |= XPAR_ATTR_NOINDEX;
  if (a & FILE_ATTRIBUTE_SPARSE_FILE)  r |= XPAR_ATTR_SPARSE;
  if (a & FILE_ATTRIBUTE_COMPRESSED)   r |= XPAR_ATTR_COMPRESSED;
  if (a & FILE_ATTRIBUTE_ENCRYPTED)    r |= XPAR_ATTR_ENCRYPTED;
  return r;
}

static DWORD attrs_to_win(u16 a, DWORD current) {
  DWORD w = current & ~(DWORD) (FILE_ATTRIBUTE_READONLY |
                                FILE_ATTRIBUTE_HIDDEN |
                                FILE_ATTRIBUTE_SYSTEM |
                                FILE_ATTRIBUTE_ARCHIVE |
                                FILE_ATTRIBUTE_NOT_CONTENT_INDEXED);
  if (a & XPAR_ATTR_READONLY) w |= FILE_ATTRIBUTE_READONLY;
  if (a & XPAR_ATTR_HIDDEN)   w |= FILE_ATTRIBUTE_HIDDEN;
  if (a & XPAR_ATTR_SYSTEM)   w |= FILE_ATTRIBUTE_SYSTEM;
  if (a & XPAR_ATTR_ARCHIVE)  w |= FILE_ATTRIBUTE_ARCHIVE;
  if (a & XPAR_ATTR_NOINDEX)  w |= FILE_ATTRIBUTE_NOT_CONTENT_INDEXED;
  return w ? w : FILE_ATTRIBUTE_NORMAL;
}

static HANDLE open_entry(const char * path, DWORD access) {
  xchar * wp = path_conv(path);
  HANDLE h;
  DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  DWORD flags = FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT;
  if (!wp) { SetLastError(ERROR_INVALID_NAME);  return INVALID_HANDLE_VALUE; }
#if defined(XPAR_WIN_LEGACY)
  h = CreateFileA(wp, access, share, NULL, OPEN_EXISTING, flags, NULL);
#else
  h = CreateFileW(wp, access, share, NULL, OPEN_EXISTING, flags, NULL);
#endif
  xpar_free(wp);
  return h;
}

int xpar_lstat(const char * path, xpar_stat_t * out) {
#if defined(XPAR_WIN_LEGACY)
  WIN32_FIND_DATAA fd;
  HANDLE fh;

  out->mode  = XPAR_MODE_NONE;
  out->uid   = XPAR_ID_NONE;
  out->gid   = XPAR_ID_NONE;
  out->dev   = 0;  out->ino = 0;  out->nlink = 1;
  out->btime_ns = XPAR_TIME_NONE;
  out->is_symlink = out->is_dir = out->is_regular = false;

  fh = FindFirstFileA(path, &fd);
  if (fh == INVALID_HANDLE_VALUE) return -1;
  FindClose(fh);
  out->size     = ((u64) fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
  out->attrs    = attrs_of(fd.dwFileAttributes);
  out->mtime_ns = ft_ns(fd.ftLastWriteTime);
  out->atime_ns = ft_ns(fd.ftLastAccessTime);
  out->ctime_ns = XPAR_TIME_NONE;
  out->btime_ns = ft_ns(fd.ftCreationTime);
  out->is_dir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  out->is_regular = !out->is_dir;
  return 0;
#else
  HANDLE h;
  BY_HANDLE_FILE_INFORMATION bi;

  out->mode  = XPAR_MODE_NONE;
  out->uid   = XPAR_ID_NONE;
  out->gid   = XPAR_ID_NONE;
  out->dev   = 0;  out->ino = 0;  out->nlink = 1;
  out->btime_ns = XPAR_TIME_NONE;
  out->is_symlink = out->is_dir = out->is_regular = false;

  h = open_entry(path, FILE_READ_ATTRIBUTES);
  if (h == INVALID_HANDLE_VALUE) return -1;
  if (!GetFileInformationByHandle(h, &bi)) { CloseHandle(h);  return -1; }
  CloseHandle(h);

  out->size     = ((u64) bi.nFileSizeHigh << 32) | bi.nFileSizeLow;
  out->attrs    = attrs_of(bi.dwFileAttributes);
  out->mtime_ns = ft_ns(bi.ftLastWriteTime);
  out->atime_ns = ft_ns(bi.ftLastAccessTime);
  /*  Win32 has no inode-change time; the field is absent rather than
      filled with the creation time, which would be a different fact.  */
  out->ctime_ns = XPAR_TIME_NONE;
  out->btime_ns = ft_ns(bi.ftCreationTime);
  out->dev      = (u64) bi.dwVolumeSerialNumber;
  out->ino      = ((u64) bi.nFileIndexHigh << 32) | bi.nFileIndexLow;
  out->nlink    = (u64) bi.nNumberOfLinks;
  out->is_symlink = (bi.dwFileAttributes &
                     FILE_ATTRIBUTE_REPARSE_POINT) != 0;
  out->is_dir     = (bi.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  out->is_regular = !out->is_dir && !out->is_symlink;
  return 0;
#endif
}

u32 xpar_fs_caps(const char * path) {
  u32 c = XPAR_FS_FATATTR | XPAR_FS_BTIME;
#if defined(XPAR_WIN_LEGACY)
  /*  MS-DOS-derived Windows has no reparse points, so there is nothing a
      metadata call could traverse and nofollow is satisfied by the absence
      of the hazard rather than by an implementation.  */
  (void) path;
  c |= XPAR_FS_NOFOLLOW;
  return c;
#else
  { wchar_t * wp = path_conv(path);
    wchar_t root[MAX_PATH], fsname[64];
    DWORD flags = 0, maxcomp = 0;
    if (!wp) return 0;
    if (!GetVolumePathNameW(wp, root, MAX_PATH)) { xpar_free(wp);  return c; }
    xpar_free(wp);
    if (!GetVolumeInformationW(root, NULL, 0, NULL, &maxcomp, &flags,
                               fsname, 64)) return c;
    /*  The identity and the link capability travel together and both are
        NTFS-only: FAT has no file index worth the name, so both bits stay
        clear and every alias becomes a copy at extract time (rule 8).  */
    if (fsname[0] == L'N' && fsname[1] == L'T' &&
        fsname[2] == L'F' && fsname[3] == L'S')
      c |= XPAR_FS_LINKID | XPAR_FS_HARDLINK | XPAR_FS_NSEC_TIME;
  #if _WIN32_WINNT >= 0x0600
    /*  SetFileInformationByHandle is what makes the attribute setter act
        on a reparse point instead of through it. Without it there is no
        symlink-safe attribute call at all, so the bit is withheld.  */
    c |= XPAR_FS_NOFOLLOW;
  #endif
    return c; }
#endif
}

struct xpar_dir {
#if defined(XPAR_WIN_LEGACY)
  WIN32_FIND_DATAA fd;
#else
  WIN32_FIND_DATAW fd;
  char *           name8;
#endif
  HANDLE      h;
  bool        pending;   /*  FindFirstFile already produced an entry  */
  xpar_dirent ent;
};

xpar_dir * xpar_opendir(const char * path) {
  struct xpar_dir * d = xpar_alloc_raw(sizeof(*d));
  char * pattern = NULL;
  sz len = xpar_strlen(path);
  xchar * wp;
  xpar_asprintf(&pattern, "%s%s*", path,
                len && (path[len - 1] == '\\' || path[len - 1] == '/')
                  ? "" : "\\");
  wp = path_conv(pattern);
  xpar_free(pattern);
  if (!wp) { xpar_free(d);  return NULL; }
#if defined(XPAR_WIN_LEGACY)
  d->h = FindFirstFileA(wp, &d->fd);
#else
  d->name8 = NULL;
  d->h = FindFirstFileW(wp, &d->fd);
#endif
  xpar_free(wp);
  if (d->h == INVALID_HANDLE_VALUE) { xpar_free(d);  return NULL; }
  d->pending = true;
  return d;
}

const xpar_dirent * xpar_readdir(xpar_dir * d) {
  for (;;) {
    if (!d->pending) {
#if defined(XPAR_WIN_LEGACY)
      if (!FindNextFileA(d->h, &d->fd)) return NULL;
#else
      if (!FindNextFileW(d->h, &d->fd)) return NULL;
#endif
    }
    d->pending = false;
#if defined(XPAR_WIN_LEGACY)
    if (d->fd.cFileName[0] == '.' &&
        (d->fd.cFileName[1] == '\0' ||
         (d->fd.cFileName[1] == '.' && d->fd.cFileName[2] == '\0')))
      continue;
    d->ent.name = d->fd.cFileName;
#else
    if (d->fd.cFileName[0] == L'.' &&
        (d->fd.cFileName[1] == L'\0' ||
         (d->fd.cFileName[1] == L'.' && d->fd.cFileName[2] == L'\0')))
      continue;
    xpar_free(d->name8);
    d->name8 = path_back(d->fd.cFileName, -1);
    if (!d->name8) continue;
    d->ent.name = d->name8;
#endif
    d->ent.is_symlink =
      (d->fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
    d->ent.is_dir =
      (d->fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    d->ent.is_regular = !d->ent.is_dir && !d->ent.is_symlink;
    return &d->ent;
  }
}

void xpar_closedir(xpar_dir * d) {
  if (!d) return;
  if (d->h != INVALID_HANDLE_VALUE) FindClose(d->h);
#if !defined(XPAR_WIN_LEGACY)
  xpar_free(d->name8);
#endif
  xpar_free(d);
}

#if !defined(XPAR_WIN_LEGACY)
typedef struct {
  DWORD  ReparseTag;
  WORD   ReparseDataLength;
  WORD   Reserved;
  WORD   SubstituteNameOffset;
  WORD   SubstituteNameLength;
  WORD   PrintNameOffset;
  WORD   PrintNameLength;
  DWORD  Flags;
  WCHAR  PathBuffer[1];
} xpar_reparse_symlink;
#endif

i64 xpar_readlink(const char * path, char * buf, sz n) {
#if defined(XPAR_WIN_LEGACY)
  (void) path;  (void) buf;  (void) n;
  fail_unsupported();
  return -1;
#else
  u8 raw[16384];   /*  MAXIMUM_REPARSE_DATA_BUFFER_SIZE  */
  DWORD got = 0;
  HANDLE h = open_entry(path, FILE_READ_ATTRIBUTES);
  xpar_reparse_symlink * rp;
  char * target;
  sz tlen;
  if (h == INVALID_HANDLE_VALUE) return -1;
  if (!DeviceIoControl(h, FSCTL_GET_REPARSE_POINT, NULL, 0, raw, sizeof raw,
                       &got, NULL)) { CloseHandle(h);  return -1; }
  CloseHandle(h);
  rp = (xpar_reparse_symlink *) (void *) raw;
  if (rp->ReparseTag != IO_REPARSE_TAG_SYMLINK) {
    fail_unsupported();
    return -1;
  }
  /*  The print name is the form a user typed; the substitute name carries
      the \??\ device prefix. Prefer the print name and fall back.  */
  if (rp->PrintNameLength)
    target = path_back((const WCHAR *) ((u8 *) rp->PathBuffer +
                                        rp->PrintNameOffset),
                       (int) (rp->PrintNameLength / sizeof(WCHAR)));
  else
    target = path_back((const WCHAR *) ((u8 *) rp->PathBuffer +
                                        rp->SubstituteNameOffset),
                       (int) (rp->SubstituteNameLength / sizeof(WCHAR)));
  if (!target) return -1;
  tlen = xpar_strlen(target);
  if (tlen + 1 > n) {
    xpar_free(target);
    SetLastError(ERROR_BUFFER_OVERFLOW);
    return -1;
  }
  xpar_memcpy(buf, target, tlen + 1);
  xpar_free(target);
  return (i64) tlen;
#endif
}

int xpar_symlink(const char * target, const char * path) {
#if !defined(XPAR_WIN_LEGACY) && (_WIN32_WINNT >= 0x0600)
  /*  Creating a symbolic link needs SeCreateSymbolicLinkPrivilege or
      developer mode, so a plain failure here is expected and the caller
      reports it rather than aborting the extraction.  */
  xchar * wt = path_text(target);
  xchar * wp = path_conv(path);
  BOOL ok = FALSE;
  if (wt && wp) ok = CreateSymbolicLinkW(wp, wt, 0);
  xpar_free(wt);  xpar_free(wp);
  return ok ? 0 : -1;
#else
  (void) target;  (void) path;
  fail_unsupported();
  return -1;
#endif
}

int xpar_link(const char * existing, const char * newpath) {
#if !defined(XPAR_WIN_LEGACY) && (_WIN32_WINNT >= 0x0500)
  xchar * we = path_conv(existing);
  xchar * wn = path_conv(newpath);
  BOOL ok = FALSE;
  if (we && wn) ok = CreateHardLinkW(wn, we, NULL);
  xpar_free(we);  xpar_free(wn);
  return ok ? 0 : -1;
#else
  (void) existing;  (void) newpath;
  fail_unsupported();
  return -1;
#endif
}

int xpar_mkdir(const char * path, u32 mode) {
  xchar * wp = path_conv(path);
  BOOL ok;
  (void) mode;   /*  no POSIX mode on this host; see the file abstract  */
  if (!wp) return -1;
#if defined(XPAR_WIN_LEGACY)
  ok = CreateDirectoryA(wp, NULL);
#else
  ok = CreateDirectoryW(wp, NULL);
#endif
  xpar_free(wp);
  return ok ? 0 : -1;
}

int xpar_mkdir_p(const char * path, u32 mode) {
  char * work = xpar_strdup(path);
  sz i, n = xpar_strlen(work);
  int rc = 0;
  for (i = 1; i <= n && rc == 0; i++) {
    char save;
    if (work[i] != '/' && work[i] != '\\' && work[i] != '\0') continue;
    /*  "C:\" is a drive root and cannot be created; skip the prefix that
        ends in a colon.  */
    if (i >= 1 && work[i - 1] == ':') continue;
    save = work[i];
    work[i] = '\0';
    if (xpar_mkdir(work, mode) != 0) {
      xchar * wp;
      DWORD a;
      if (GetLastError() != ERROR_ALREADY_EXISTS) { rc = -1;  break; }
      wp = path_conv(work);
      if (!wp) { rc = -1;  break; }
#if defined(XPAR_WIN_LEGACY)
      a = GetFileAttributesA(wp);
#else
      a = GetFileAttributesW(wp);
#endif
      xpar_free(wp);
      if (a == INVALID_FILE_ATTRIBUTES ||
          !(a & FILE_ATTRIBUTE_DIRECTORY) ||
          (a & FILE_ATTRIBUTE_REPARSE_POINT)) {
        SetLastError(ERROR_CANT_ACCESS_FILE);  rc = -1;
      }
    }
    work[i] = save;
  }
  xpar_free(work);
  return rc;
}

int xpar_rmdir(const char * path) {
  xchar * wp = path_conv(path);
  BOOL ok;
  if (!wp) return -1;
#if defined(XPAR_WIN_LEGACY)
  ok = RemoveDirectoryA(wp);
#else
  ok = RemoveDirectoryW(wp);
#endif
  xpar_free(wp);
  return ok ? 0 : -1;
}

int xpar_remove(const char * path) {
  xchar * wp = path_conv(path);
  BOOL ok;
  if (!wp) return -1;
#if defined(XPAR_WIN_LEGACY)
  ok = DeleteFileA(wp);
#else
  ok = DeleteFileW(wp);
#endif
  xpar_free(wp);
  return ok ? 0 : -1;
}

int xpar_rename(const char * from, const char * to) {
  xchar * wf = path_conv(from);
  xchar * wt = path_conv(to);
  BOOL ok = FALSE;
  if (wf && wt) {
#if defined(XPAR_WIN_LEGACY)
    /*  9x has no MoveFileEx replace flag; remove the target first, which
        is the same window MoveFileEx would close and the best 9x has.  */
    DeleteFileA(wt);
    ok = MoveFileA(wf, wt);
#else
    ok = MoveFileExW(wf, wt, MOVEFILE_REPLACE_EXISTING);
#endif
  }
  xpar_free(wf);  xpar_free(wt);
  return ok ? 0 : -1;
}

int xpar_set_times(const char * path, int nofollow,
                   i64 atime_ns, i64 mtime_ns, i64 btime_ns) {
  HANDLE h;
  FILETIME a, m, b;
  BOOL ok;
  (void) nofollow;   /*  open_entry never traverses; see below  */
  h = open_entry(path, FILE_WRITE_ATTRIBUTES);
  if (h == INVALID_HANDLE_VALUE) return -1;
  a = ns_ft(atime_ns);  m = ns_ft(mtime_ns);  b = ns_ft(btime_ns);
  ok = SetFileTime(h,
                   btime_ns == XPAR_TIME_NONE ? NULL : &b,
                   atime_ns == XPAR_TIME_NONE ? NULL : &a,
                   mtime_ns == XPAR_TIME_NONE ? NULL : &m);
  CloseHandle(h);
  return ok ? 0 : -1;
}

int xpar_set_owner(const char * path, int nofollow, u32 uid, u32 gid,
                   const char * owner, const char * group) {
  (void) path;  (void) nofollow;  (void) uid;  (void) gid;
  (void) owner;  (void) group;
  fail_unsupported();
  return -1;
}

int xpar_set_mode(const char * path, int nofollow, u32 mode) {
  (void) path;  (void) nofollow;  (void) mode;
  fail_unsupported();
  return -1;
}

int xpar_set_attrs(const char * path, int nofollow, u16 attrs) {
#if _WIN32_WINNT >= 0x0600 && !defined(XPAR_WIN_LEGACY)
  FILE_BASIC_INFO fbi;
  HANDLE h = open_entry(path, FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES);
  BOOL ok;
  (void) nofollow;
  if (h == INVALID_HANDLE_VALUE) return -1;
  if (!GetFileInformationByHandleEx(h, FileBasicInfo, &fbi, sizeof fbi)) {
    CloseHandle(h);
    return -1;
  }
  /*  Zeroing the four time fields tells the call to leave them alone,
      which keeps an attribute change from also stamping the file.  */
  fbi.CreationTime.QuadPart   = 0;
  fbi.LastAccessTime.QuadPart = 0;
  fbi.LastWriteTime.QuadPart  = 0;
  fbi.ChangeTime.QuadPart     = 0;
  fbi.FileAttributes = attrs_to_win(attrs & XPAR_ATTR_SETTABLE,
                                    fbi.FileAttributes);
  ok = SetFileInformationByHandle(h, FileBasicInfo, &fbi, sizeof fbi);
  CloseHandle(h);
  return ok ? 0 : -1;
#else
  /*  SetFileAttributes is path-based and follows a reparse point, so on a
      target without SetFileInformationByHandle there is no symlink-safe
      way to do this and the request is refused rather than redirected.
      XPAR_FS_NOFOLLOW is clear on such a build for exactly this field.  */
  xchar * wp;
  DWORD cur;
  BOOL ok;
  if (nofollow) { fail_unsupported();  return -1; }
  wp = path_conv(path);
  if (!wp) return -1;
  #if defined(XPAR_WIN_LEGACY)
  cur = GetFileAttributesA(wp);
  ok = cur != INVALID_FILE_ATTRIBUTES &&
       SetFileAttributesA(wp, attrs_to_win(attrs & XPAR_ATTR_SETTABLE, cur));
  #else
  cur = GetFileAttributesW(wp);
  ok = cur != INVALID_FILE_ATTRIBUTES &&
       SetFileAttributesW(wp, attrs_to_win(attrs & XPAR_ATTR_SETTABLE, cur));
  #endif
  xpar_free(wp);
  return ok ? 0 : -1;
#endif
}

sz xpar_listxattr(const char * path, int nofollow, char * buf, sz n) {
  (void) path;  (void) nofollow;  (void) buf;  (void) n;
  fail_unsupported();
  return XPAR_FS_NOSIZE;
}

sz xpar_getxattr(const char * path, int nofollow, const char * name,
                 void * buf, sz n) {
  (void) path;  (void) nofollow;  (void) name;  (void) buf;  (void) n;
  fail_unsupported();
  return XPAR_FS_NOSIZE;
}

int xpar_setxattr(const char * path, int nofollow, const char * name,
                  const void * val, sz n) {
  (void) path;  (void) nofollow;  (void) name;  (void) val;  (void) n;
  fail_unsupported();
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
