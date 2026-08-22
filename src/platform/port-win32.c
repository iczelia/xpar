/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License only.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.  */

/*  Win32 host I/O, UTF path conversion, and portable formatting.

    NT uses OVERLAPPED positional I/O.  Legacy Windows serialises
    SetFilePointer and transfers per handle.  */

#if !(defined(_WIN32) || defined(__MINGW32__) || defined(__MINGW64__))
#error "port-win32.c compiled for a non-Windows target"
#endif

#if !defined(_WIN32_WINNT)
  #define _WIN32_WINNT 0x0600
#endif
#if !defined(WINVER)
  #define WINVER _WIN32_WINNT
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
/*  WIN32_LEAN_AND_MEAN drops the crypto headers, and HCRYPTPROV is needed
    for the pre-XP random-bytes path below.  */
#include <wincrypt.h>

#include "common.h"

#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#if defined(XPAR_WIN_LEGACY)
const char * xpar_getenv(const char * name) {
  static char * value;
  static DWORD cap;
  DWORD need = GetEnvironmentVariableA(name, NULL, 0);
  if (!need) return NULL;
  if (need > cap) {
    char * p = value ? HeapReAlloc(GetProcessHeap(), 0, value, need)
                     : HeapAlloc(GetProcessHeap(), 0, need);
    if (!p) return NULL;
    value = p;  cap = need;
  }
  return GetEnvironmentVariableA(name, value, cap) ? value : NULL;
}
#else
const char * xpar_getenv(const char * name) { return getenv(name); }
#endif

#if !defined(XPAR_WIN_LEGACY)

static wchar_t * to_wide(const char * s) {
  int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
  wchar_t * w;
  if (n <= 0) return NULL;
  w = HeapAlloc(GetProcessHeap(), 0, (sz) n * sizeof(wchar_t));
  if (!w) return NULL;
  if (MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n) <= 0) {
    HeapFree(GetProcessHeap(), 0, w);
    return NULL;
  }
  return w;
}

static wchar_t * to_wide_path(const char * s) {
  wchar_t * raw = to_wide(s), * full, * out;
  DWORD n, got;
  sz bytes;
  if (!raw) return NULL;
  if ((raw[0] == L'\\' && raw[1] == L'\\' &&
       (raw[2] == L'?' || raw[2] == L'.') && raw[3] == L'\\') ||
      (raw[0] == L'\\' && raw[1] == L'?' && raw[2] == L'?' &&
       raw[3] == L'\\')) {
    HeapFree(GetProcessHeap(), 0, raw);
    SetLastError(ERROR_INVALID_NAME);
    return NULL;
  }
  n = GetFullPathNameW(raw, 0, NULL, NULL);
  if (!n || n > 32768u) {
    HeapFree(GetProcessHeap(), 0, raw);
    return NULL;
  }
  bytes = ((sz) n + 1) * sizeof(*full);
  full = HeapAlloc(GetProcessHeap(), 0, bytes);
  if (!full) {
    HeapFree(GetProcessHeap(), 0, raw);
    return NULL;
  }
  got = GetFullPathNameW(raw, n + 1, full, NULL);
  HeapFree(GetProcessHeap(), 0, raw);
  if (!got || got > n) {
    HeapFree(GetProcessHeap(), 0, full);
    return NULL;
  }
  if (full[0] == L'\\' && full[1] == L'\\') {
    out = HeapAlloc(GetProcessHeap(), 0,
                    ((sz) got + 7) * sizeof(*out));
    if (out) {
      xpar_memcpy(out, L"\\\\?\\UNC\\", 8 * sizeof(*out));
      xpar_memcpy(out + 8, full + 2, ((sz) got - 1) * sizeof(*out));
    }
  } else {
    out = HeapAlloc(GetProcessHeap(), 0,
                    ((sz) got + 5) * sizeof(*out));
    if (out) {
      xpar_memcpy(out, L"\\\\?\\", 4 * sizeof(*out));
      xpar_memcpy(out + 4, full, ((sz) got + 1) * sizeof(*out));
    }
  }
  HeapFree(GetProcessHeap(), 0, full);
  return out;
}

static char * to_utf8(const wchar_t * w) {
  int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
  char * s;
  if (n <= 0) return NULL;
  s = HeapAlloc(GetProcessHeap(), 0, (sz) n);
  if (!s) return NULL;
  if (WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL) <= 0) {
    HeapFree(GetProcessHeap(), 0, s);
    return NULL;
  }
  return s;
}

#endif

struct xpar_file {
  HANDLE           h;
  DWORD            kind;     /*  FILE_TYPE_DISK / PIPE / CHAR / UNKNOWN  */
  bool             owned;
  bool             at_eof;
  bool             locked;   /*  unlocking a handle that never locked errors  */
  DWORD            last_err;
#if defined(XPAR_WIN_LEGACY)
  CRITICAL_SECTION seek_cs;  /*  serialises seek-plus-transfer  */
#endif
};

/*  Left to static zero-initialisation, which is the right starting state
    for every field including the legacy critical section; the handles
    themselves arrive in xpar_host_init.  */
static struct xpar_file g_stdin;
static struct xpar_file g_stdout;
static struct xpar_file g_stderr;

xpar_file * const xpar_stdin  = &g_stdin;
xpar_file * const xpar_stdout = &g_stdout;
xpar_file * const xpar_stderr = &g_stderr;

void xpar_host_init(void) {
  g_stdin.h  = GetStdHandle(STD_INPUT_HANDLE);
  g_stdout.h = GetStdHandle(STD_OUTPUT_HANDLE);
  g_stderr.h = GetStdHandle(STD_ERROR_HANDLE);
  g_stdin.kind  = g_stdin.h  ? GetFileType(g_stdin.h)  : FILE_TYPE_UNKNOWN;
  g_stdout.kind = g_stdout.h ? GetFileType(g_stdout.h) : FILE_TYPE_UNKNOWN;
  g_stderr.kind = g_stderr.h ? GetFileType(g_stderr.h) : FILE_TYPE_UNKNOWN;
#if !defined(XPAR_WIN_LEGACY)
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
}

#if !defined(XPAR_WIN_LEGACY)
static wchar_t * win_final_name(HANDLE h) {
  DWORD n = GetFinalPathNameByHandleW(h, NULL, 0,
                                      FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
  wchar_t * out;
  if (!n) return NULL;
  out = HeapAlloc(GetProcessHeap(), 0, ((sz) n + 1) * sizeof(*out));
  if (!out) return NULL;
  if (!GetFinalPathNameByHandleW(h, out, n + 1,
                                 FILE_NAME_NORMALIZED | VOLUME_NAME_DOS)) {
    HeapFree(GetProcessHeap(), 0, out);
    return NULL;
  }
  return out;
}

static wchar_t * win_root_end(wchar_t * path) {
  wchar_t * p = path;
  if (p[0] == L'\\' && p[1] == L'\\' && p[2] == L'?' &&
      p[3] == L'\\') {
    if ((p[4] == L'U' || p[4] == L'u') &&
        (p[5] == L'N' || p[5] == L'n') &&
        (p[6] == L'C' || p[6] == L'c') && p[7] == L'\\') {
      p += 8;
      while (*p && *p != L'\\') p++;
      if (*p) p++;
      while (*p && *p != L'\\') p++;
      return *p ? p + 1 : p;
    }
    if (p[4] && p[5] == L':' && p[6] == L'\\') return p + 7;
  }
  if (p[0] && p[1] == L':' && (p[2] == L'\\' || p[2] == L'/'))
    return p + 3;
  if (p[0] == L'\\' && p[1] == L'\\') {
    p += 2;
    while (*p && *p != L'\\' && *p != L'/') p++;
    if (*p) p++;
    while (*p && *p != L'\\' && *p != L'/') p++;
    if (*p) p++;
  }
  return p;
}

static bool win_safe_prefixes(wchar_t * full) {
  wchar_t * p;
  for (p = win_root_end(full); *p; p++) {
    DWORD a;
    wchar_t keep;
    if ((*p != L'\\' && *p != L'/') || !p[1]) continue;
    keep = *p;  *p = 0;
    a = GetFileAttributesW(full);
    *p = keep;
    if (a == INVALID_FILE_ATTRIBUTES ||
        !(a & FILE_ATTRIBUTE_DIRECTORY) ||
        (a & FILE_ATTRIBUTE_REPARSE_POINT)) {
      SetLastError(ERROR_CANT_ACCESS_FILE);
      return false;
    }
  }
  return true;
}

static bool win_same_parent(const wchar_t * parent, HANDLE h) {
  wchar_t * final = win_final_name(h), * slash;
  bool same = false;
  if (!final) return false;
  slash = wcsrchr(final, L'\\');
  if (slash) {
    *slash = 0;
    same = _wcsicmp(parent, final) == 0;
  }
  HeapFree(GetProcessHeap(), 0, final);
  return same;
}

static HANDLE win_open_nofollow(const wchar_t * path, DWORD access,
                                DWORD share, DWORD creation, DWORD attrs) {
  sz n = wcslen(path);
  DWORD made;
  wchar_t * full, * slash, * root, * parent;
  wchar_t keep;
  HANDLE ph, h;
  BY_HANDLE_FILE_INFORMATION info;
  FILE_DISPOSITION_INFO dispose;
  full = HeapAlloc(GetProcessHeap(), 0, ((sz) n + 1) * sizeof(*full));
  if (!full) { SetLastError(ERROR_OUTOFMEMORY);  return INVALID_HANDLE_VALUE; }
  xpar_memcpy(full, path, (n + 1) * sizeof(*full));
  if (!win_safe_prefixes(full)) {
    HeapFree(GetProcessHeap(), 0, full);
    return INVALID_HANDLE_VALUE;
  }
  slash = wcsrchr(full, L'\\');
  if (!slash) slash = wcsrchr(full, L'/');
  if (!slash) {
    HeapFree(GetProcessHeap(), 0, full);
    SetLastError(ERROR_INVALID_NAME);
    return INVALID_HANDLE_VALUE;
  }
  root = win_root_end(full);
  if (slash + 1 == root) { keep = slash[1];  slash[1] = 0; }
  else                   { keep = *slash;    *slash = 0; }
  ph = CreateFileW(full, FILE_READ_ATTRIBUTES,
                   FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                   NULL, OPEN_EXISTING,
                   FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
                   NULL);
  if (slash + 1 == root) slash[1] = keep;
  else                   *slash = keep;
  if (ph == INVALID_HANDLE_VALUE) {
    HeapFree(GetProcessHeap(), 0, full);
    return INVALID_HANDLE_VALUE;
  }
  parent = win_final_name(ph);
  CloseHandle(ph);
  if (!parent) {
    HeapFree(GetProcessHeap(), 0, full);
    return INVALID_HANDLE_VALUE;
  }
  if (slash + 1 == root) { keep = slash[1];  slash[1] = 0; }
  else                   { keep = *slash;    *slash = 0; }
  if (_wcsicmp(parent, full) != 0) {
    if (slash + 1 == root) slash[1] = keep;
    else                   *slash = keep;
    HeapFree(GetProcessHeap(), 0, parent);
    HeapFree(GetProcessHeap(), 0, full);
    SetLastError(ERROR_CANT_ACCESS_FILE);
    return INVALID_HANDLE_VALUE;
  }
  if (slash + 1 == root) slash[1] = keep;
  else                   *slash = keep;
  if (creation != OPEN_EXISTING) access |= DELETE;
  h = CreateFileW(full, access, share, NULL, creation,
                  attrs | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  made = GetLastError();
  HeapFree(GetProcessHeap(), 0, full);
  if (h == INVALID_HANDLE_VALUE) {
    HeapFree(GetProcessHeap(), 0, parent);
    return h;
  }
  if (!win_same_parent(parent, h) ||
      !GetFileInformationByHandle(h, &info) ||
      (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
    dispose.DeleteFile = creation == CREATE_NEW ||
                         (creation == OPEN_ALWAYS &&
                          made != ERROR_ALREADY_EXISTS);
    if (dispose.DeleteFile)
      (void) SetFileInformationByHandle(h, FileDispositionInfo,
                                        &dispose, sizeof dispose);
    CloseHandle(h);
    h = INVALID_HANDLE_VALUE;
    SetLastError(ERROR_CANT_ACCESS_FILE);
  }
  HeapFree(GetProcessHeap(), 0, parent);
  return h;
}
#endif

xpar_file * xpar_open(const char * path, int flags) {
  DWORD access = 0, share = FILE_SHARE_READ, creation;
  DWORD attrs = FILE_ATTRIBUTE_NORMAL;
  int acc = flags & 3;
  HANDLE h;
  struct xpar_file * f;

  if (acc == XPAR_O_WRONLY)     access = GENERIC_WRITE;
  else if (acc == XPAR_O_RDWR)  access = GENERIC_READ | GENERIC_WRITE;
  else                          access = GENERIC_READ;
  if (flags & XPAR_O_APPEND)    access |= FILE_APPEND_DATA;

  if ((flags & XPAR_O_CREAT) && (flags & XPAR_O_EXCL))  creation = CREATE_NEW;
  else if ((flags & XPAR_O_CREAT) && (flags & XPAR_O_TRUNC))
    creation = CREATE_ALWAYS;
  else if (flags & XPAR_O_CREAT) creation = OPEN_ALWAYS;
  else if (flags & XPAR_O_TRUNC) creation = TRUNCATE_EXISTING;
  else                           creation = OPEN_EXISTING;
#if defined(XPAR_WIN_LEGACY)
  h = CreateFileA(path, access, share, NULL, creation, attrs, NULL);
#else
  { wchar_t * wp = to_wide_path(path);
    if (!wp) { SetLastError(ERROR_INVALID_NAME);  return NULL; }
    if (flags & XPAR_O_NOFOLLOW)
      h = win_open_nofollow(wp, access, share, creation, attrs);
    else
      h = CreateFileW(wp, access, share, NULL, creation, attrs, NULL);
    HeapFree(GetProcessHeap(), 0, wp); }
#endif
  if (h == INVALID_HANDLE_VALUE) return NULL;
  f = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*f));
  if (!f) { CloseHandle(h);  SetLastError(ERROR_OUTOFMEMORY);  return NULL; }
  f->h = h;  f->kind = GetFileType(h);  f->owned = true;
#if defined(XPAR_WIN_LEGACY)
  InitializeCriticalSection(&f->seek_cs);
#endif
  return f;
}

int xpar_close(xpar_file * f) {
  int r;
  if (!f || !f->owned) return 0;
  r = CloseHandle(f->h) ? 0 : -1;
#if defined(XPAR_WIN_LEGACY)
  DeleteCriticalSection(&f->seek_cs);
#endif
  HeapFree(GetProcessHeap(), 0, f);
  return r;
}

#define WIN_CHUNK 0x40000000u

sz xpar_read(xpar_file * f, void * buf, sz n) {
  sz total = 0;
  char * p = (char *) buf;
  while (total < n) {
    DWORD want = (n - total > WIN_CHUNK) ? WIN_CHUNK : (DWORD) (n - total);
    DWORD got;
    if (!ReadFile(f->h, p + total, want, &got, NULL)) {
      f->last_err = GetLastError();
      /*  A closed pipe is the writer's end of file, not a read error.  */
      if (f->last_err == ERROR_BROKEN_PIPE) { f->at_eof = true;
                                              f->last_err = 0; }
      return total;
    }
    if (got == 0) { f->at_eof = true;  break; }
    total += got;
  }
  return total;
}

sz xpar_write(xpar_file * f, const void * buf, sz n) {
  sz total = 0;
  const char * p = (const char *) buf;
  while (total < n) {
    DWORD want = (n - total > WIN_CHUNK) ? WIN_CHUNK : (DWORD) (n - total);
    DWORD wrote;
    if (!WriteFile(f->h, p + total, want, &wrote, NULL)) {
      f->last_err = GetLastError();
      return total;
    }
    if (wrote == 0) break;
    total += wrote;
  }
  return total;
}

int xpar_seek(xpar_file * f, i64 off, int whence) {
  DWORD method = whence == XPAR_SEEK_SET ? FILE_BEGIN
               : whence == XPAR_SEEK_CUR ? FILE_CURRENT : FILE_END;
#if _WIN32_WINNT >= 0x0500
  LARGE_INTEGER li;
  li.QuadPart = off;
  if (!SetFilePointerEx(f->h, li, NULL, method)) {
    f->last_err = GetLastError();
    return -1;
  }
#else
  /*  Pre-Win2K has no SetFilePointerEx, and SetFilePointer signals failure
      with a value that is also a legal position, so the error case is
      disambiguated through GetLastError.  */
  LONG hi = (LONG) ((u64) off >> 32);
  DWORD got;
  SetLastError(NO_ERROR);
  got = SetFilePointer(f->h, (LONG) ((u64) off & 0xFFFFFFFFu), &hi, method);
  if (got == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) {
    f->last_err = GetLastError();
    return -1;
  }
#endif
  f->at_eof = false;
  return 0;
}

i64 xpar_tell(xpar_file * f) {
#if _WIN32_WINNT >= 0x0500
  LARGE_INTEGER zero, pos;
  zero.QuadPart = 0;
  if (!SetFilePointerEx(f->h, zero, &pos, FILE_CURRENT)) return -1;
  return (i64) pos.QuadPart;
#else
  LONG hi = 0;
  DWORD lo;
  SetLastError(NO_ERROR);
  lo = SetFilePointer(f->h, 0, &hi, FILE_CURRENT);
  if (lo == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) return -1;
  return (i64) (((u64) (u32) hi << 32) | (u64) (u32) lo);
#endif
}

int xpar_flush(xpar_file * f) {
  if (f->kind != FILE_TYPE_DISK) return 0;
  return FlushFileBuffers(f->h) ? 0 : -1;
}

int xpar_fsync(xpar_file * f) { return xpar_flush(f); }

i64 xpar_size(xpar_file * f) {
#if _WIN32_WINNT >= 0x0500
  LARGE_INTEGER li;
  if (!GetFileSizeEx(f->h, &li)) return -1;
  return (i64) li.QuadPart;
#else
  DWORD hi = 0, lo;
  SetLastError(NO_ERROR);
  lo = GetFileSize(f->h, &hi);
  if (lo == INVALID_FILE_SIZE && GetLastError() != NO_ERROR) return -1;
  return (i64) (((u64) hi << 32) | (u64) lo);
#endif
}

bool xpar_is_seekable(xpar_file * f) { return f->kind == FILE_TYPE_DISK; }

bool xpar_is_tty(xpar_file * f) {
  DWORD mode;
  if (f->kind != FILE_TYPE_CHAR) return false;
  return GetConsoleMode(f->h, &mode) != 0;
}

bool xpar_eof  (xpar_file * f) { return f->at_eof; }
int  xpar_error(xpar_file * f) { return (int) f->last_err; }

#define WIN_LOCK_LO  0xFFFFFFFFu
#define WIN_LOCK_HI  0xFFFFFFFFu

int xpar_lock(xpar_file * f, bool exclusive) {
#if defined(XPAR_WIN_LEGACY)
  (void) exclusive;
  if (!LockFile(f->h, 0, 0, WIN_LOCK_LO, WIN_LOCK_HI)) {
    f->last_err = GetLastError();
    return -1;
  }
#else
  OVERLAPPED ov;
  DWORD flags = LOCKFILE_FAIL_IMMEDIATELY;
  xpar_memset(&ov, 0, sizeof ov);
  if (exclusive) flags |= LOCKFILE_EXCLUSIVE_LOCK;
  if (!LockFileEx(f->h, flags, 0, WIN_LOCK_LO, WIN_LOCK_HI, &ov)) {
    f->last_err = GetLastError();
    return -1;
  }
#endif
  f->locked = true;
  return 0;
}

int xpar_unlock(xpar_file * f) {
  BOOL ok;
  if (!f->locked) return 0;
#if defined(XPAR_WIN_LEGACY)
  ok = UnlockFile(f->h, 0, 0, WIN_LOCK_LO, WIN_LOCK_HI);
#else
  { OVERLAPPED ov;
    xpar_memset(&ov, 0, sizeof ov);
    ok = UnlockFileEx(f->h, 0, WIN_LOCK_LO, WIN_LOCK_HI, &ov); }
#endif
  if (!ok) { f->last_err = GetLastError();  return -1; }
  f->locked = false;
  return 0;
}

bool xpar_lock_supported(void) { return true; }

sz xpar_pread(xpar_file * f, void * buf, sz n, u64 off) {
  sz total = 0;
  char * p = (char *) buf;
  while (total < n) {
    DWORD want = (n - total > WIN_CHUNK) ? WIN_CHUNK : (DWORD) (n - total);
    DWORD got = 0;
    BOOL ok;
#if defined(XPAR_WIN_LEGACY)
    EnterCriticalSection(&f->seek_cs);
    { LONG hi = (LONG) ((off + total) >> 32);
      DWORD pos;
      SetLastError(NO_ERROR);
      pos = SetFilePointer(f->h, (LONG) ((off + total) & 0xFFFFFFFFu), &hi,
                           FILE_BEGIN);
      ok = !(pos == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR);
      if (ok) ok = ReadFile(f->h, p + total, want, &got, NULL); }
    LeaveCriticalSection(&f->seek_cs);
#else
    OVERLAPPED ov;
    xpar_memset(&ov, 0, sizeof ov);
    ov.Offset     = (DWORD) ((off + total) & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD) ((off + total) >> 32);
    ok = ReadFile(f->h, p + total, want, &got, &ov);
    /*  On a handle opened without FILE_FLAG_OVERLAPPED the call still
        completes synchronously and honours ov.Offset; ERROR_HANDLE_EOF is
        how a read past the end reports itself in this form.  */
    if (!ok && GetLastError() == ERROR_HANDLE_EOF) break;
#endif
    /*  A short count is the whole answer here: the EOF flag describes the
        sequential cursor, which a positional read has not moved.  */
    if (!ok) { f->last_err = GetLastError();  break; }
    if (got == 0) break;
    total += got;
  }
  return total;
}

bool xpar_pread_batch(xpar_read_req * r, sz count) {
  sz i;
  for (i = 0; i < count; i++)
    r[i].result = r[i].file
                    ? xpar_pread(r[i].file, r[i].buf, r[i].length,
                                 r[i].offset) : 0;
  return false;
}

sz xpar_pwrite(xpar_file * f, const void * buf, sz n, u64 off) {
  sz total = 0;
  const char * p = (const char *) buf;
  while (total < n) {
    DWORD want = (n - total > WIN_CHUNK) ? WIN_CHUNK : (DWORD) (n - total);
    DWORD wrote = 0;
    BOOL ok;
#if defined(XPAR_WIN_LEGACY)
    EnterCriticalSection(&f->seek_cs);
    { LONG hi = (LONG) ((off + total) >> 32);
      DWORD pos;
      SetLastError(NO_ERROR);
      pos = SetFilePointer(f->h, (LONG) ((off + total) & 0xFFFFFFFFu), &hi,
                           FILE_BEGIN);
      ok = !(pos == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR);
      if (ok) ok = WriteFile(f->h, p + total, want, &wrote, NULL); }
    LeaveCriticalSection(&f->seek_cs);
#else
    OVERLAPPED ov;
    xpar_memset(&ov, 0, sizeof ov);
    ov.Offset     = (DWORD) ((off + total) & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD) ((off + total) >> 32);
    ok = WriteFile(f->h, p + total, want, &wrote, &ov);
#endif
    if (!ok) { f->last_err = GetLastError();  break; }
    if (wrote == 0) break;
    total += wrote;
  }
  return total;
}

int xpar_ftruncate(xpar_file * f, u64 length) {
  i64 keep = xpar_tell(f);
  int r = 0;
  if (xpar_seek(f, (i64) length, XPAR_SEEK_SET) != 0) return -1;
  if (!SetEndOfFile(f->h)) { f->last_err = GetLastError();  r = -1; }
  /*  SetEndOfFile truncates at the file pointer, so the pointer the caller
      left behind has to be put back or the next xpar_write lands in the
      wrong place.  */
  if (keep >= 0) xpar_seek(f, keep, XPAR_SEEK_SET);
  return r;
}

/*  NTFS journals directory metadata, and Win32 has no supported way to
    flush a directory handle in any case: FlushFileBuffers wants write
    access, which CreateFile will not grant on a directory. Reporting
    success is honest here, unlike inventing a flush that does nothing.  */
int xpar_fsync_dir(const char * path) { (void) path;  return 0; }

sz xpar_xread(xpar_file * f, void * p, sz n) {
  sz got = xpar_read(f, p, n);
  if (f->last_err) FATAL_PERROR("read");
  return got;
}

void xpar_xwrite(xpar_file * f, const void * p, sz n) {
  if (xpar_write(f, p, n) != n) FATAL_PERROR("write");
}

void xpar_xwritev(xpar_file * f, const xpar_write_part * part, u32 count) {
  u32 i;
  for (i = 0; i < count; i++)
    xpar_xwrite(f, part[i].data, part[i].length);
}

void xpar_xclose(xpar_file * f) {
  if (!f) return;
  if (f->kind == FILE_TYPE_DISK && !FlushFileBuffers(f->h))
    FATAL_PERROR("flush");
  if (f->owned) {
    if (!CloseHandle(f->h)) FATAL_PERROR("close");
#if defined(XPAR_WIN_LEGACY)
    DeleteCriticalSection(&f->seek_cs);
#endif
    HeapFree(GetProcessHeap(), 0, f);
  }
}

xpar_mmap xpar_map(const char * path) {
  xpar_mmap m;
  HANDLE fh, fm;
  DWORD size_hi = 0, size_lo;
  u64 total;
  m.map = NULL;  m.size = 0;  m.valid = false;

#if defined(XPAR_WIN_LEGACY)
  fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                   FILE_ATTRIBUTE_NORMAL, NULL);
#else
  { wchar_t * wp = to_wide_path(path);
    if (!wp) return m;
    fh = CreateFileW(wp, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                     FILE_ATTRIBUTE_NORMAL, NULL);
    HeapFree(GetProcessHeap(), 0, wp); }
#endif
  if (fh == INVALID_HANDLE_VALUE) return m;

  SetLastError(NO_ERROR);
  size_lo = GetFileSize(fh, &size_hi);
  if (size_lo == INVALID_FILE_SIZE && GetLastError() != NO_ERROR) {
    CloseHandle(fh);
    return m;
  }
  total = ((u64) size_hi << 32) | (u64) size_lo;
  /*  A file larger than the address space cannot be mapped; the caller
      streams it instead, which every caller of xpar_map can do.  */
  if (total == 0 || total > (u64) (sz) -1) { CloseHandle(fh);  return m; }

#if defined(XPAR_WIN_LEGACY)
  fm = CreateFileMappingA(fh, NULL, PAGE_READONLY, size_hi, size_lo, NULL);
#else
  fm = CreateFileMappingW(fh, NULL, PAGE_READONLY, size_hi, size_lo, NULL);
#endif
  if (!fm) { CloseHandle(fh);  return m; }
  m.map = (u8 *) MapViewOfFile(fm, FILE_MAP_READ, 0, 0, 0);
  CloseHandle(fm);
  CloseHandle(fh);
  if (!m.map) return m;
  m.size = (sz) total;
  m.valid = true;
  return m;
}

void xpar_unmap(xpar_mmap * m) {
  if (m->map) UnmapViewOfFile(m->map);
  m->map = NULL;  m->size = 0;  m->valid = false;
}

/*  Win32 takes access hints at CreateFile time (FILE_FLAG_SEQUENTIAL_SCAN)
    and offers nothing to say afterwards, so these are no-ops rather than
    approximations.  */
void xpar_advise_sequential(xpar_file * f, u64 off, u64 len) {
  (void) f;  (void) off;  (void) len;
}
void xpar_advise_random(xpar_file * f, u64 off, u64 len) {
  (void) f;  (void) off;  (void) len;
}

/*  -----------------------------------------------------------------------
    Memory  */

void * xpar_malloc(sz n) {
  void * p = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, n ? n : 1);
  if (!p) FATAL("Out of memory.");
  return p;
}

void * xpar_calloc(sz n, sz size) {
  if (n && size && n > (sz) -1 / size) FATAL("Allocation size overflow.");
  { void * p = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY,
                         n && size ? n * size : 1);
    if (!p) FATAL("Out of memory.");
    return p; }
}

void * xpar_alloc_raw(sz n) {
  void * p = HeapAlloc(GetProcessHeap(), 0, n ? n : 1);
  if (!p) FATAL("Out of memory.");
  return p;
}

void * xpar_realloc(void * p, sz n) {
  void * q;
  if (!p) return xpar_alloc_raw(n);
  q = HeapReAlloc(GetProcessHeap(), 0, p, n ? n : 1);
  if (!q) FATAL("Out of memory.");
  return q;
}

void xpar_free(void * p) {
  if (p) HeapFree(GetProcessHeap(), 0, p);
}

/*  VirtualAlloc would round every request up to 64 KiB of address space,
    so the aligned allocator over-allocates on the heap and stores the base
    pointer in the word below the block. xpar_free_aligned is therefore not
    interchangeable with xpar_free.  */
void * xpar_alloc_aligned(sz n, sz align) {
  u8 * raw;
  uintptr_t a;
  sz pad;
  if (align < sizeof(void *)) align = sizeof(void *);
  if (!xpar_is_pow2(align)) FATAL("Alignment is not a power of two.");
  if (n == 0) n = 1;
  pad = align + sizeof(void *);
  if (n > (sz) -1 - pad) FATAL("Allocation size overflow.");
  raw = HeapAlloc(GetProcessHeap(), 0, n + pad);
  if (!raw) FATAL("Out of memory.");
  a = ((uintptr_t) raw + sizeof(void *) + align - 1) &
      ~(uintptr_t) (align - 1);
  ((void **) a)[-1] = raw;
  return (void *) a;
}

void xpar_free_aligned(void * p) {
  if (p) HeapFree(GetProcessHeap(), 0, ((void **) p)[-1]);
}

u64 xpar_physical_memory(void) {
#if _WIN32_WINNT >= 0x0500
  MEMORYSTATUSEX ms;
  ms.dwLength = sizeof ms;
  if (GlobalMemoryStatusEx(&ms)) return (u64) ms.ullTotalPhys;
  return 0;
#else
  /*  The pre-Win2K structure saturates its fields at 2 GiB, which is above
      anything a 9x host has and therefore harmless here.  */
  MEMORYSTATUS ms;
  ms.dwLength = sizeof ms;
  GlobalMemoryStatus(&ms);
  return (u64) ms.dwTotalPhys;
#endif
}

/*  There is no supported Win32 query for rotational media short of an
    IOCTL against the raw volume, which needs privileges this program does
    not ask for. Unknown answers false, which costs a planner heuristic and
    never correctness (DESIGN 5.3).  */
bool xpar_is_rotational(const char * path) { (void) path;  return false; }

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

char * xpar_strdup(const char * s) {
  sz n = strlen(s) + 1;
  char * c = xpar_alloc_raw(n);
  memcpy(c, s, n);
  return c;
}

char * xpar_strndup(const char * s, sz n) {
  sz len = 0;
  char * c;
  while (len < n && s[len]) len++;
  c = xpar_alloc_raw(len + 1);
  memcpy(c, s, len);
  c[len] = '\0';
  return c;
}

int xpar_parse_u64(const char * s, u64 * out) {
  u64 v = 0;
  if (!s || !*s) return -1;
  if (*s == '+') s++;
  if (!*s) return -1;
  while (*s) {
    u64 nv;
    if (*s < '0' || *s > '9') return -1;
    nv = v * 10 + (u64) (*s - '0');
    if (nv < v) return -1;
    v = nv;  s++;
  }
  *out = v;
  return 0;
}

int xpar_parse_i64(const char * s, i64 * out) {
  int neg = 0;
  u64 v;
  if (!s || !*s) return -1;
  if (*s == '-') { neg = 1;  s++; }
  if (xpar_parse_u64(s, &v) != 0) return -1;
  if (neg) {
    if (v > (u64) INT64_MAX + 1) return -1;
    *out = v == (u64) INT64_MAX + 1 ? INT64_MIN : -(i64) v;
  } else {
    if (v > (u64) INT64_MAX) return -1;
    *out = (i64) v;
  }
  return 0;
}

typedef struct {
  char * buf;
  sz     cap;
  sz     pos;   /*  bytes that WOULD be written, for snprintf sizing  */
} fmt_ctx;

static void emit_c(fmt_ctx * c, char ch) {
  if (c->buf && c->pos + 1 < c->cap) c->buf[c->pos] = ch;
  c->pos++;
}
static void emit_str(fmt_ctx * c, const char * s, sz n) {
  for (sz i = 0; i < n; i++) emit_c(c, s[i]);
}
static void emit_pad(fmt_ctx * c, int n, char ch) {
  while (n-- > 0) emit_c(c, ch);
}

enum { F_MINUS = 1, F_PLUS = 2, F_SPACE = 4, F_ZERO = 8, F_HASH = 16 };

static void emit_uint(fmt_ctx * c, u64 v, int base, int upper,
                      int width, int prec, int flags) {
  char tmp[32];
  int n = 0, len, pad_zero, total, pad_sp;
  const char * digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
  if (v == 0 && prec == 0) n = 0;
  else do { tmp[n++] = digits[v % (u64) base];  v /= (u64) base; } while (v);
  len = n;
  /*  POSIX: an explicit precision on an integer conversion voids '0'.  */
  pad_zero = prec > len ? prec - len : 0;
  total    = len + pad_zero;
  pad_sp   = width > total ? width - total : 0;
  if (!(flags & F_MINUS) && !(flags & F_ZERO)) emit_pad(c, pad_sp, ' ');
  if (!(flags & F_MINUS) && (flags & F_ZERO) && prec < 0)
    emit_pad(c, pad_sp, '0');
  emit_pad(c, pad_zero, '0');
  while (n) emit_c(c, tmp[--n]);
  if (flags & F_MINUS) emit_pad(c, pad_sp, ' ');
}

static void emit_int(fmt_ctx * c, i64 v, int width, int prec, int flags) {
  char sign = 0;
  u64 uv;
  /*  Negating INT64_MIN is undefined, so the magnitude is built as
      (-(v+1))+1, which stays in range at every step.  */
  if (v < 0)                { sign = '-';  uv = (u64) (-(v + 1)) + 1; }
  else if (flags & F_PLUS)  { sign = '+';  uv = (u64) v; }
  else if (flags & F_SPACE) { sign = ' ';  uv = (u64) v; }
  else                      { uv = (u64) v; }
  if (sign) {
    if (width > 0 && !(flags & F_MINUS) && !(flags & F_ZERO)) {
      char tmp[32];
      int n = 0, len, pad_zero, total, pad_sp;
      if (uv == 0 && prec == 0) n = 0;
      else { u64 x = uv;
             do { tmp[n++] = (char) ('0' + (x % 10));  x /= 10; } while (x); }
      len      = n;
      pad_zero = prec > len ? prec - len : 0;
      total    = 1 + len + pad_zero;
      pad_sp   = width > total ? width - total : 0;
      emit_pad(c, pad_sp, ' ');
      emit_c(c, sign);
      emit_pad(c, pad_zero, '0');
      while (n) emit_c(c, tmp[--n]);
      return;
    }
    emit_c(c, sign);
    width = width > 0 ? width - 1 : 0;
  }
  emit_uint(c, uv, 10, 0, width, prec, flags);
}

static void emit_double(fmt_ctx * c, double v, int width, int prec,
                        int flags) {
  char sign = 0, ibuf[24];
  u64 ip, fp, mult = 1;
  double frac;
  int in = 0, total, pad;
  if (prec < 0) prec = 6;
  if (v < 0)                { sign = '-';  v = -v; }
  else if (flags & F_PLUS)  { sign = '+'; }
  else if (flags & F_SPACE) { sign = ' '; }
  ip   = (u64) v;
  frac = v - (double) ip;
  for (int i = 0; i < prec; i++) mult *= 10;
  fp = (u64) (frac * (double) mult + 0.5);
  if (fp >= mult) { ip++;  fp -= mult; }
  if (ip == 0) ibuf[in++] = '0';
  else { u64 x = ip;
         while (x) { ibuf[in++] = (char) ('0' + (x % 10));  x /= 10; } }
  total = in + (prec > 0 ? 1 + prec : 0) + (sign ? 1 : 0);
  pad   = width > total ? width - total : 0;
  if (!(flags & F_MINUS) && !(flags & F_ZERO)) emit_pad(c, pad, ' ');
  if (sign) emit_c(c, sign);
  if (!(flags & F_MINUS) && (flags & F_ZERO)) emit_pad(c, pad, '0');
  while (in) emit_c(c, ibuf[--in]);
  if (prec > 0) {
    char fbuf[24];
    int fn = 0;
    u64 x = fp;
    emit_c(c, '.');
    for (int i = 0; i < prec; i++) { fbuf[fn++] = (char) ('0' + (x % 10));
                                     x /= 10; }
    while (fn) emit_c(c, fbuf[--fn]);
  }
  if (flags & F_MINUS) emit_pad(c, pad, ' ');
}

int xpar_vsnprintf(char * buf, sz cap, const char * fmt, va_list ap) {
  fmt_ctx c;
  c.buf = buf;  c.cap = cap;  c.pos = 0;
  while (*fmt) {
    int flags = 0, width = 0, prec = -1, longness = 0;
    char spec;
    if (*fmt != '%') { emit_c(&c, *fmt++);  continue; }
    fmt++;
    for (;; fmt++) {
      if      (*fmt == '-') flags |= F_MINUS;
      else if (*fmt == '+') flags |= F_PLUS;
      else if (*fmt == ' ') flags |= F_SPACE;
      else if (*fmt == '0') flags |= F_ZERO;
      else if (*fmt == '#') flags |= F_HASH;
      else break;
    }
    if (*fmt == '*') { width = va_arg(ap, int);  fmt++; }
    else while (*fmt >= '0' && *fmt <= '9')
      { width = width * 10 + (*fmt - '0');  fmt++; }
    if (*fmt == '.') {
      fmt++;  prec = 0;
      if (*fmt == '*') { prec = va_arg(ap, int);  fmt++; }
      else while (*fmt >= '0' && *fmt <= '9')
        { prec = prec * 10 + (*fmt - '0');  fmt++; }
    }
    /*  0 = int, 1 = long, 2 = long long, 3 = size_t  */
    if (*fmt == 'z') { longness = 3;  fmt++; }
    else if (*fmt == 'l') {
      fmt++;
      if (*fmt == 'l') { longness = 2;  fmt++; }
      else longness = 1;
    } else if (*fmt == 'h') {
      fmt++;
      if (*fmt == 'h') fmt++;   /*  default promotions make both int  */
    }
    spec = *fmt;
    if (spec) fmt++;
    switch (spec) {
      case 'd': case 'i': {
        i64 v;
        if      (longness == 2) v = va_arg(ap, long long);
        else if (longness == 1) v = va_arg(ap, long);
        else if (longness == 3) v = (i64) va_arg(ap, ptrdiff_t);
        else                    v = va_arg(ap, int);
        emit_int(&c, v, width, prec, flags);
        break;
      }
      case 'u': case 'x': case 'X': {
        u64 v;
        if      (longness == 2) v = va_arg(ap, unsigned long long);
        else if (longness == 1) v = va_arg(ap, unsigned long);
        else if (longness == 3) v = (u64) va_arg(ap, sz);
        else                    v = va_arg(ap, unsigned);
        emit_uint(&c, v, spec == 'u' ? 10 : 16, spec == 'X',
                  width, prec, flags);
        break;
      }
      case 'p': {
        void * p = va_arg(ap, void *);
        emit_str(&c, "0x", 2);
        emit_uint(&c, (u64) (uintptr_t) p, 16, 0,
                  (int) (sizeof(void *) * 2), -1, F_ZERO);
        break;
      }
      case 'c': {
        int ch  = va_arg(ap, int);
        int pad = width > 1 ? width - 1 : 0;
        if (!(flags & F_MINUS)) emit_pad(&c, pad, ' ');
        emit_c(&c, (char) ch);
        if (flags & F_MINUS) emit_pad(&c, pad, ' ');
        break;
      }
      case 's': {
        const char * s = va_arg(ap, const char *);
        sz slen = 0;
        int pad;
        if (!s) s = "(null)";
        while (s[slen] && (prec < 0 || slen < (sz) prec)) slen++;
        pad = width > (int) slen ? width - (int) slen : 0;
        if (!(flags & F_MINUS)) emit_pad(&c, pad, ' ');
        emit_str(&c, s, slen);
        if (flags & F_MINUS) emit_pad(&c, pad, ' ');
        break;
      }
      case 'f': emit_double(&c, va_arg(ap, double), width, prec, flags);
                break;
      case '%': emit_c(&c, '%');  break;
      default:  emit_c(&c, '%');  if (spec) emit_c(&c, spec);  break;
    }
  }
  if (c.buf && c.cap > 0)
    c.buf[c.pos < c.cap ? c.pos : c.cap - 1] = '\0';
  return (int) c.pos;
}

int xpar_snprintf(char * buf, sz cap, const char * fmt, ...) {
  va_list ap;  int r;
  va_start(ap, fmt);
  r = xpar_vsnprintf(buf, cap, fmt, ap);
  va_end(ap);
  return r;
}

int xpar_asprintf(char ** out, const char * fmt, ...) {
  va_list ap, ap2;  int n;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  n = xpar_vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (n < 0) { va_end(ap2);  *out = NULL;  return -1; }
  *out = xpar_alloc_raw((sz) n + 1);
  xpar_vsnprintf(*out, (sz) n + 1, fmt, ap2);
  va_end(ap2);
  return n;
}

static void write_raw(xpar_file * f, const char * s, sz n) {
#if defined(XPAR_WIN_LEGACY)
  if (f == xpar_stdout || f == xpar_stderr) {
    char out[512];
    sz i, used = 0;
    for (i = 0; i < n; i++) {
      sz need = s[i] == '\n' && (i == 0 || s[i - 1] != '\r') ? 2 : 1;
      if (sizeof out - used < need) {
        if (xpar_write(f, out, used) != used) return;
        used = 0;
      }
      if (need == 2) out[used++] = '\r';
      out[used++] = s[i];
    }
    if (used) xpar_write(f, out, used);
    return;
  }
#else
  if (xpar_is_tty(f)) {
    /*  A console is UTF-16 underneath: WriteFile of UTF-8 bytes renders
        mojibake regardless of the code page on several Windows versions,
        so the text is converted and written as wide characters.  */
    int wn = MultiByteToWideChar(CP_UTF8, 0, s, (int) n, NULL, 0);
    if (wn > 0) {
      wchar_t stack[512], * w = stack;
      if ((sz) wn > sizeof stack / sizeof stack[0])
        w = HeapAlloc(GetProcessHeap(), 0, (sz) wn * sizeof(wchar_t));
      if (w) {
        DWORD written;
        MultiByteToWideChar(CP_UTF8, 0, s, (int) n, w, wn);
        WriteConsoleW(f->h, w, (DWORD) wn, &written, NULL);
        if (w != stack) HeapFree(GetProcessHeap(), 0, w);
        return;
      }
    }
  }
#endif
  xpar_write(f, s, n);
}

int xpar_vfprintf(xpar_file * f, const char * fmt, va_list ap) {
  char stack[1024];
  va_list ap2;
  int n;
  va_copy(ap2, ap);
  n = xpar_vsnprintf(stack, sizeof stack, fmt, ap);
  if (n < (int) sizeof stack) {
    va_end(ap2);
    write_raw(f, stack, (sz) n);
    return n;
  }
  { char * big = xpar_alloc_raw((sz) n + 1);
    xpar_vsnprintf(big, (sz) n + 1, fmt, ap2);
    va_end(ap2);
    write_raw(f, big, (sz) n);
    xpar_free(big); }
  return n;
}

int xpar_fprintf(xpar_file * f, const char * fmt, ...) {
  va_list ap;  int r;
  va_start(ap, fmt);
  r = xpar_vfprintf(f, fmt, ap);
  va_end(ap);
  return r;
}

int xpar_fputs(const char * s, xpar_file * f) {
  sz n = xpar_strlen(s);
  write_raw(f, s, n);
  return (int) n;
}

void xpar_exit(int code) { ExitProcess((UINT) code); }

static char g_errbuf[256];

const char * xpar_strerror(int err) {
#if defined(XPAR_WIN_LEGACY)
  DWORD n = FormatMessageA(
    FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
    (DWORD) err, 0, g_errbuf, sizeof g_errbuf - 1, NULL);
  if (n == 0) {
    xpar_snprintf(g_errbuf, sizeof g_errbuf, "Windows error %d", err);
    return g_errbuf;
  }
  while (n > 0 && (g_errbuf[n - 1] == '\r' || g_errbuf[n - 1] == '\n' ||
                   g_errbuf[n - 1] == '.'  || g_errbuf[n - 1] == ' ')) n--;
  g_errbuf[n] = '\0';
  return g_errbuf;
#else
  wchar_t wbuf[160];
  DWORD n = FormatMessageW(
    FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
    (DWORD) err, 0, wbuf, sizeof wbuf / sizeof wbuf[0] - 1, NULL);
  if (n == 0) {
    xpar_snprintf(g_errbuf, sizeof g_errbuf, "Windows error %d", err);
    return g_errbuf;
  }
  /*  FormatMessage appends a full stop and a line break that would land in
      the middle of a one-line diagnostic.  */
  while (n > 0 && (wbuf[n - 1] == L'\r' || wbuf[n - 1] == L'\n' ||
                   wbuf[n - 1] == L'.'  || wbuf[n - 1] == L' ')) n--;
  wbuf[n] = L'\0';
  if (WideCharToMultiByte(CP_UTF8, 0, wbuf, -1, g_errbuf, sizeof g_errbuf,
                          NULL, NULL) <= 0)
    xpar_snprintf(g_errbuf, sizeof g_errbuf, "Windows error %d", err);
  return g_errbuf;
#endif
}

int xpar_errno(void) { return (int) GetLastError(); }

u64 xpar_usec_now(void) {
  static LARGE_INTEGER freq;
  LARGE_INTEGER ctr;
  u64 q, r;
  if (freq.QuadPart == 0 && !QueryPerformanceFrequency(&freq)) return 0;
  QueryPerformanceCounter(&ctr);
  /*  Split the division so that a counter near 2^63 does not overflow the
      multiply by a million.  */
  q = (u64) ctr.QuadPart / (u64) freq.QuadPart;
  r = (u64) ctr.QuadPart % (u64) freq.QuadPart;
  return q * 1000000ULL + r * 1000000ULL / (u64) freq.QuadPart;
}

/*  FILETIME counts 100 ns ticks from 1601-01-01, and 11644473600 is the
    number of seconds from there to the Unix epoch (369 years, 89 of them
    leap).  */
#define WIN_EPOCH_DELTA_100NS  116444736000000000ULL

i64 xpar_wall_ns(void) {
  FILETIME ft;
  u64 t;
  GetSystemTimeAsFileTime(&ft);
  t = ((u64) ft.dwHighDateTime << 32) | (u64) ft.dwLowDateTime;
  return (i64) ((t - WIN_EPOCH_DELTA_100NS) * 100ULL);
}

/*  RtlGenRandom is the documented user-mode entry to the system CSPRNG and
    is reached by ordinal name through advapi32 so that the program does
    not acquire a link-time dependency on it; the pre-XP path is
    CryptGenRandom, reached the same way. Failure is fatal: a keyed
    authentication tag seeded from a weak source verifies correctly and
    forges trivially, which looks exactly like working.  */
typedef BOOLEAN (WINAPI * rtl_gen_random_fn)(PVOID, ULONG);
typedef BOOL (WINAPI * crypt_acquire_fn)(HCRYPTPROV *, LPCSTR, LPCSTR,
                                         DWORD, DWORD);
typedef BOOL (WINAPI * crypt_gen_fn)(HCRYPTPROV, DWORD, BYTE *);
typedef BOOL (WINAPI * crypt_release_fn)(HCRYPTPROV, DWORD);

void xpar_random_bytes(void * buf, sz n) {
  HMODULE adv = LoadLibraryA("advapi32.dll");
  if (adv) {
    rtl_gen_random_fn gen =
      (rtl_gen_random_fn) (void *) GetProcAddress(adv, "SystemFunction036");
    if (gen) {
      sz done = 0;
      while (done < n) {
        ULONG want = n - done > 0x10000000u ? 0x10000000u
                                            : (ULONG) (n - done);
        if (!gen((u8 *) buf + done, want)) break;
        done += want;
      }
      if (done == n) { FreeLibrary(adv);  return; }
    }
    { crypt_acquire_fn acq = (crypt_acquire_fn) (void *)
        GetProcAddress(adv, "CryptAcquireContextA");
      crypt_gen_fn cgen = (crypt_gen_fn) (void *)
        GetProcAddress(adv, "CryptGenRandom");
      crypt_release_fn rel = (crypt_release_fn) (void *)
        GetProcAddress(adv, "CryptReleaseContext");
      HCRYPTPROV prov = 0;
      if (acq && cgen && rel &&
          acq(&prov, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
        BOOL ok = cgen(prov, (DWORD) n, (BYTE *) buf);
        rel(prov, 0);
        if (ok) { FreeLibrary(adv);  return; }
      } }
    FreeLibrary(adv);
  }
  FATAL("No source of cryptographically strong random bytes.");
}

#if !defined(XPAR_WIN_LEGACY)

static int grow_w(wchar_t ** pbuf, sz * pcap) {
  wchar_t * nb;
  if (*pcap > ((sz) -1) / (2 * sizeof(wchar_t))) return 0;
  nb = HeapReAlloc(GetProcessHeap(), 0, *pbuf, *pcap * 2 * sizeof(wchar_t));
  if (!nb) return 0;
  *pbuf = nb;  *pcap *= 2;
  return 1;
}

static int split_cmdline(const wchar_t * cmd, wchar_t *** out) {
  int argc = 0;
  wchar_t ** argv = NULL, * buf = NULL;
  for (int pass = 0; pass < 2; pass++) {
    const wchar_t * p = cmd;
    if (pass == 1) {
      argv = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY,
                       (sz) (argc + 1) * sizeof(wchar_t *));
      if (!argv) return -1;
    }
    argc = 0;
    while (*p) {
      sz blen = 0, bcap = 0;
      int in_quote = 0;
      while (*p == L' ' || *p == L'\t') p++;
      if (!*p) break;
      buf = NULL;
      if (pass == 1) {
        bcap = 64;
        buf = HeapAlloc(GetProcessHeap(), 0, bcap * sizeof(wchar_t));
        if (!buf) goto fail;
      }
      while (*p) {
        if (!in_quote && (*p == L' ' || *p == L'\t')) break;
        if (*p == L'\\') {
          int nbs = 0;
          while (*p == L'\\') { nbs++;  p++; }
          if (*p == L'"') {
            if (pass == 1)
              for (int i = 0; i < nbs / 2; i++) {
                if (blen + 1 >= bcap && !grow_w(&buf, &bcap)) goto fail;
                buf[blen++] = L'\\';
              }
            if (nbs & 1) {
              if (pass == 1) {
                if (blen + 1 >= bcap && !grow_w(&buf, &bcap)) goto fail;
                buf[blen++] = L'"';
              }
              p++;
            } else { in_quote = !in_quote;  p++; }
          } else if (pass == 1) {
            for (int i = 0; i < nbs; i++) {
              if (blen + 1 >= bcap && !grow_w(&buf, &bcap)) goto fail;
              buf[blen++] = L'\\';
            }
          }
        } else if (*p == L'"') {
          if (in_quote && p[1] == L'"') {
            if (pass == 1) {
              if (blen + 1 >= bcap && !grow_w(&buf, &bcap)) goto fail;
              buf[blen++] = L'"';
            }
            p += 2;
          } else { in_quote = !in_quote;  p++; }
        } else {
          if (pass == 1) {
            if (blen + 1 >= bcap && !grow_w(&buf, &bcap)) goto fail;
            buf[blen++] = *p;
          }
          p++;
        }
      }
      if (pass == 1) { buf[blen] = L'\0';  argv[argc] = buf;  buf = NULL; }
      argc++;
    }
    if (pass == 1) { *out = argv;  return argc; }
  }
  return -1;
fail:
  if (buf) HeapFree(GetProcessHeap(), 0, buf);
  if (argv) {
    for (int j = 0; j < argc; j++)
      if (argv[j]) HeapFree(GetProcessHeap(), 0, argv[j]);
    HeapFree(GetProcessHeap(), 0, argv);
  }
  return -1;
}

static int utf8_argv(int * argc_out, char *** argv_out) {
  wchar_t ** wargv;
  char ** argv;
  int wargc = split_cmdline(GetCommandLineW(), &wargv), i;
  if (wargc < 0) return -1;
  argv = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY,
                   (sz) (wargc + 1) * sizeof(char *));
  if (!argv) return -1;
  for (i = 0; i < wargc; i++) {
    argv[i] = to_utf8(wargv[i]);
    HeapFree(GetProcessHeap(), 0, wargv[i]);
    if (!argv[i]) return -1;
  }
  HeapFree(GetProcessHeap(), 0, wargv);
  *argc_out = wargc;
  *argv_out = argv;
  return 0;
}

int main(int argc, char ** argv) {
  int wargc;
  char ** wargv;
  xpar_host_init();
  if (utf8_argv(&wargc, &wargv) == 0) return xpar_main(wargc, wargv);
  return xpar_main(argc, argv);
}

#endif
