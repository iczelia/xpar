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

/*  Win32 host backend. Targets vista (default, wide API) or
    win95/XPAR_WIN_LEGACY (narrow API + event-based condvars).  */

#if !(defined(_WIN32) || defined(__MINGW32__) || defined(__MINGW64__))
#error "host-win32.c compiled on a non-Windows target"
#endif

/*  Default to Vista baseline when no target set (authoritative: configure.ac).  */
#if !defined(_WIN32_WINNT)
  #define _WIN32_WINNT 0x0600
#endif
#if !defined(WINVER)
  #define WINVER _WIN32_WINNT
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "common.h"

/*  UTF-8 <-> UTF-16 helpers; modern (wide) path only. Legacy uses -A APIs.  */

#if !defined(XPAR_WIN_LEGACY)
static wchar_t * utf8_to_wide_heap(const char * s) {
  int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
  if (n <= 0) return NULL;
  wchar_t * w = HeapAlloc(GetProcessHeap(), 0, (sz) n * sizeof(wchar_t));
  if (!w) return NULL;
  if (MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n) <= 0) {
    HeapFree(GetProcessHeap(), 0, w); return NULL;
  }
  return w;
}

static char * wide_to_utf8_heap(const wchar_t * w) {
  int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
  if (n <= 0) return NULL;
  char * s = HeapAlloc(GetProcessHeap(), 0, (sz) n);
  if (!s) return NULL;
  if (WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL) <= 0) {
    HeapFree(GetProcessHeap(), 0, s); return NULL;
  }
  return s;
}
#endif

/*  ============================================================================
    xpar_file: opaque wrapper around a Win32 HANDLE
    ============================================================================  */

struct xpar_file {
  HANDLE h;
  DWORD  kind;        /*  FILE_TYPE_DISK / PIPE / CHAR / UNKNOWN  */
  bool   owned;       /*  true for xpar_open()'d, false for the 3 std handles  */
  bool   at_eof;
  DWORD  last_err;
};

static struct xpar_file g_stdin  = { NULL, 0, false, false, 0 };
static struct xpar_file g_stdout = { NULL, 0, false, false, 0 };
static struct xpar_file g_stderr = { NULL, 0, false, false, 0 };

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
  /*  Modern Windows: force UTF-8 console. Win9x: skip (unsupported).  */
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
}

/*  ============================================================================
    Open / close
    ============================================================================  */

xpar_file * xpar_open(const char * path, int flags) {
  DWORD access = 0, share = FILE_SHARE_READ, creation = 0;
  if (flags & XPAR_O_READ)  access |= GENERIC_READ;
  if (flags & XPAR_O_WRITE) access |= GENERIC_WRITE;
  if (flags & XPAR_O_APPEND) access |= FILE_APPEND_DATA;

  if ((flags & XPAR_O_CREATE) && (flags & XPAR_O_EXCLUSIVE))    creation = CREATE_NEW;
  else if ((flags & XPAR_O_CREATE) && (flags & XPAR_O_TRUNCATE)) creation = CREATE_ALWAYS;
  else if (flags & XPAR_O_CREATE)                                creation = OPEN_ALWAYS;
  else if (flags & XPAR_O_TRUNCATE)                              creation = TRUNCATE_EXISTING;
  else                                                           creation = OPEN_EXISTING;

#if defined(XPAR_WIN_LEGACY)
  HANDLE h = CreateFileA(path, access, share, NULL, creation,
                         FILE_ATTRIBUTE_NORMAL, NULL);
#else
  wchar_t * wpath = utf8_to_wide_heap(path);
  if (!wpath) { SetLastError(ERROR_INVALID_NAME); return NULL; }
  HANDLE h = CreateFileW(wpath, access, share, NULL, creation,
                         FILE_ATTRIBUTE_NORMAL, NULL);
  HeapFree(GetProcessHeap(), 0, wpath);
#endif
  if (h == INVALID_HANDLE_VALUE) return NULL;
  struct xpar_file * f = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*f));
  if (!f) { CloseHandle(h); SetLastError(ERROR_OUTOFMEMORY); return NULL; }
  f->h = h;
  f->kind = GetFileType(h);
  f->owned = true;
  return f;
}

int xpar_close(xpar_file * f) {
  if (!f || !f->owned) return 0;
  int r = CloseHandle(f->h) ? 0 : -1;
  HeapFree(GetProcessHeap(), 0, f);
  return r;
}

/*  ============================================================================
    read / write / seek
    ============================================================================  */

sz xpar_read(xpar_file * f, void * buf, sz n) {
  /*  Loop for POSIX-fread-like semantics: partial pipe reads are stitched
      into a single full read; caller sees a short count only on EOF/error.  */
  sz total = 0;
  char * p = buf;
  while (total < n) {
    DWORD chunk = (n - total > 0x40000000) ? 0x40000000 : (DWORD)(n - total);
    DWORD got;
    if (!ReadFile(f->h, p + total, chunk, &got, NULL)) {
      f->last_err = GetLastError();
      if (f->last_err == ERROR_BROKEN_PIPE) f->at_eof = true;
      return total;
    }
    if (got == 0) { f->at_eof = true; break; }
    total += got;
  }
  return total;
}

sz xpar_write(xpar_file * f, const void * buf, sz n) {
  DWORD written;
  /*  WriteFile may short-write on >= 4 GiB buffers; loop to be safe.  */
  sz total = 0;
  const char * p = buf;
  while (total < n) {
    DWORD chunk = (n - total > 0x40000000) ? 0x40000000 : (DWORD)(n - total);
    if (!WriteFile(f->h, p + total, chunk, &written, NULL)) {
      f->last_err = GetLastError();
      return total;
    }
    if (written == 0) break;
    total += written;
  }
  return total;
}

int xpar_seek(xpar_file * f, i64 off, int whence) {
  DWORD method = whence == XPAR_SEEK_SET ? FILE_BEGIN
               : whence == XPAR_SEEK_CUR ? FILE_CURRENT
                                         : FILE_END;
#if _WIN32_WINNT >= 0x0500
  LARGE_INTEGER li; li.QuadPart = off;
  return SetFilePointerEx(f->h, li, NULL, method) ? 0 : -1;
#else
  /*  Pre-Win2K: 32-bit SetFilePointer, high-dword split; disambiguate
      via GetLastError.  */
  LONG lo = (LONG)((u64) off & 0xFFFFFFFFu);
  LONG hi = (LONG)((u64) off >> 32);
  SetLastError(NO_ERROR);
  DWORD got = SetFilePointer(f->h, lo, &hi, method);
  if (got == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) return -1;
  return 0;
#endif
}

i64 xpar_tell(xpar_file * f) {
#if _WIN32_WINNT >= 0x0500
  LARGE_INTEGER zero = {{0, 0}}, pos;
  if (!SetFilePointerEx(f->h, zero, &pos, FILE_CURRENT)) return -1;
  return (i64) pos.QuadPart;
#else
  LONG hi = 0;
  SetLastError(NO_ERROR);
  DWORD lo = SetFilePointer(f->h, 0, &hi, FILE_CURRENT);
  if (lo == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) return -1;
  return (i64)(((u64)(u32) hi << 32) | (u64)(u32) lo);
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
  /*  GetFileSize: low in return, high in *hi; disambiguate via GetLastError.  */
  DWORD hi = 0;
  SetLastError(NO_ERROR);
  DWORD lo = GetFileSize(f->h, &hi);
  if (lo == INVALID_FILE_SIZE && GetLastError() != NO_ERROR) return -1;
  return (i64)(((u64) hi << 32) | (u64) lo);
#endif
}

bool xpar_is_seekable(xpar_file * f) { return f->kind == FILE_TYPE_DISK; }

bool xpar_is_tty(xpar_file * f) {
  if (f->kind != FILE_TYPE_CHAR) return false;
  DWORD mode;
  return GetConsoleMode(f->h, &mode) ? true : false;
}

bool xpar_eof  (xpar_file * f) { return f->at_eof; }
int  xpar_error(xpar_file * f) { return (int) f->last_err; }

/*  ============================================================================
    Safe helpers
    ============================================================================  */

sz xpar_xread(xpar_file * f, void * p, sz n) {
  sz got = xpar_read(f, p, n);
  if (f->last_err && f->last_err != ERROR_BROKEN_PIPE) FATAL_PERROR("read");
  return got;
}
void xpar_xwrite(xpar_file * f, const void * p, sz n) {
  if (xpar_write(f, p, n) != n) FATAL_PERROR("write");
}
void xpar_xclose(xpar_file * f) {
  if (!f) return;
  if (!FlushFileBuffers(f->h) && f->kind == FILE_TYPE_DISK) {
    DWORD e = GetLastError();
    if (e != ERROR_INVALID_HANDLE) { SetLastError(e); FATAL_PERROR("flush"); }
  }
  if (f->owned) {
    if (!CloseHandle(f->h)) FATAL_PERROR("close");
    HeapFree(GetProcessHeap(), 0, f);
  }
}
void xpar_notty(xpar_file * f) {
  if (xpar_is_tty(f))
    FATAL("Refusing to read/write binary data from/to a terminal.");
}

/*  ============================================================================
    Filesystem
    ============================================================================  */

int xpar_stat_path(const char * path, xpar_stat_t * out) {
#if defined(XPAR_WIN_LEGACY)
  /*  Win95 lacks GetFileAttributesExA; use FindFirstFileA instead.  */
  WIN32_FIND_DATAA fd;
  HANDLE h = FindFirstFileA(path, &fd);
  if (h != INVALID_HANDLE_VALUE) {
    FindClose(h);
    out->size = ((u64) fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
    out->is_dir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? true : false;
    out->is_regular = !out->is_dir;
    return 0;
  }
  /*  Drive roots reject FindFirstFileA; fall back to GetFileAttributesA.  */
  DWORD attrs = GetFileAttributesA(path);
  if (attrs == INVALID_FILE_ATTRIBUTES) return -1;
  out->size = 0;
  out->is_dir = (attrs & FILE_ATTRIBUTE_DIRECTORY) ? true : false;
  out->is_regular = !out->is_dir;
  return 0;
#else
  wchar_t * wpath = utf8_to_wide_heap(path);
  if (!wpath) return -1;
  WIN32_FILE_ATTRIBUTE_DATA info;
  BOOL ok = GetFileAttributesExW(wpath, GetFileExInfoStandard, &info);
  HeapFree(GetProcessHeap(), 0, wpath);
  if (!ok) return -1;
  out->size = ((u64) info.nFileSizeHigh << 32) | info.nFileSizeLow;
  out->is_dir = (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? true : false;
  out->is_regular = !out->is_dir;
  return 0;
#endif
}
int xpar_remove(const char * path) {
#if defined(XPAR_WIN_LEGACY)
  return DeleteFileA(path) ? 0 : -1;
#else
  wchar_t * wpath = utf8_to_wide_heap(path);
  if (!wpath) return -1;
  BOOL ok = DeleteFileW(wpath);
  HeapFree(GetProcessHeap(), 0, wpath);
  return ok ? 0 : -1;
#endif
}
int xpar_same_file(const char * a, const char * b) {
#if defined(XPAR_WIN_LEGACY)
  /*  No BY_HANDLE_FILE_INFORMATION on Win9x; compare full pathnames.  */
  char pa[MAX_PATH], pb[MAX_PATH];
  if (!GetFullPathNameA(a, MAX_PATH, pa, NULL)) return -1;
  if (!GetFullPathNameA(b, MAX_PATH, pb, NULL)) return -1;
  return lstrcmpiA(pa, pb) == 0 ? 1 : 0;
#else
  wchar_t * wa = utf8_to_wide_heap(a);
  if (!wa) return -1;
  wchar_t * wb = utf8_to_wide_heap(b);
  if (!wb) { HeapFree(GetProcessHeap(), 0, wa); return -1; }
  HANDLE ha = CreateFileW(wa, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                          NULL, OPEN_EXISTING,
                          FILE_FLAG_BACKUP_SEMANTICS, NULL);
  HANDLE hb = CreateFileW(wb, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                          NULL, OPEN_EXISTING,
                          FILE_FLAG_BACKUP_SEMANTICS, NULL);
  HeapFree(GetProcessHeap(), 0, wa);
  HeapFree(GetProcessHeap(), 0, wb);
  if (ha == INVALID_HANDLE_VALUE || hb == INVALID_HANDLE_VALUE) {
    if (ha != INVALID_HANDLE_VALUE) CloseHandle(ha);
    if (hb != INVALID_HANDLE_VALUE) CloseHandle(hb);
    return -1;
  }
  BY_HANDLE_FILE_INFORMATION ia, ib;
  BOOL oka = GetFileInformationByHandle(ha, &ia);
  BOOL okb = GetFileInformationByHandle(hb, &ib);
  CloseHandle(ha); CloseHandle(hb);
  if (!oka || !okb) return -1;
  return (ia.dwVolumeSerialNumber == ib.dwVolumeSerialNumber
       && ia.nFileIndexHigh       == ib.nFileIndexHigh
       && ia.nFileIndexLow        == ib.nFileIndexLow) ? 1 : 0;
#endif
}

/*  ============================================================================
    Memory map
    ============================================================================  */

xpar_mmap xpar_map(const char * path) {
  xpar_mmap m = { NULL, 0 };
#if defined(XPAR_WIN_LEGACY)
  HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
#else
  wchar_t * wpath = utf8_to_wide_heap(path);
  if (!wpath) return m;
  HANDLE fh = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  HeapFree(GetProcessHeap(), 0, wpath);
#endif
  if (fh == INVALID_HANDLE_VALUE) return m;

  DWORD size_hi, size_lo;
#if _WIN32_WINNT >= 0x0500
  LARGE_INTEGER fsize;
  if (!GetFileSizeEx(fh, &fsize) || fsize.QuadPart == 0) {
    CloseHandle(fh); return m;
  }
  size_hi = fsize.HighPart; size_lo = fsize.LowPart;
  u64 total = (u64) fsize.QuadPart;
#else
  size_hi = 0;
  SetLastError(NO_ERROR);
  size_lo = GetFileSize(fh, &size_hi);
  if (size_lo == INVALID_FILE_SIZE && GetLastError() != NO_ERROR) {
    CloseHandle(fh); return m;
  }
  u64 total = ((u64) size_hi << 32) | (u64) size_lo;
  if (total == 0) { CloseHandle(fh); return m; }
#endif

#if defined(XPAR_WIN_LEGACY)
  HANDLE fm = CreateFileMappingA(fh, NULL, PAGE_READONLY,
                                 size_hi, size_lo, NULL);
#else
  HANDLE fm = CreateFileMappingW(fh, NULL, PAGE_READONLY,
                                 size_hi, size_lo, NULL);
#endif
  if (!fm) { CloseHandle(fh); return m; }
  m.map = MapViewOfFile(fm, FILE_MAP_READ, 0, 0, 0);
  if (!m.map) { CloseHandle(fm); CloseHandle(fh); return m; }
  m.size = (sz) total;
  CloseHandle(fm); CloseHandle(fh);
  return m;
}
void xpar_unmap(xpar_mmap * m) {
  if (m->map) UnmapViewOfFile(m->map);
  m->map = NULL; m->size = 0;
}

/*  ============================================================================
    Allocation
    ============================================================================  */

void * xpar_malloc(sz n) {
  if (n == 0) n = 1;
  void * p = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, n);
  if (!p) FATAL("out of memory");
  return p;
}
void * xpar_alloc_raw(sz n) {
  if (n == 0) n = 1;
  void * p = HeapAlloc(GetProcessHeap(), 0, n);
  if (!p) FATAL("out of memory");
  return p;
}
void * xpar_realloc(void * p, sz n) {
  if (!p) return xpar_alloc_raw(n);
  void * q = HeapReAlloc(GetProcessHeap(), 0, p, n ? n : 1);
  if (!q) FATAL("out of memory");
  return q;
}
void xpar_free(void * p) {
  if (p) HeapFree(GetProcessHeap(), 0, p);
}

/*  ============================================================================
    Freestanding strings: byte-loop fallbacks for __builtin_str*.  */

sz strlen(const char * s) {
  const char * p = s;
  while (*p) p++;
  return (sz)(p - s);
}
int strcmp(const char * a, const char * b) {
  while (*a && *a == *b) { a++; b++; }
  return (int)(u8)*a - (int)(u8)*b;
}
int strncmp(const char * a, const char * b, sz n) {
  while (n && *a && *a == *b) { a++; b++; n--; }
  if (n == 0) return 0;
  return (int)(u8)*a - (int)(u8)*b;
}

/*  Higher-level xpar_str* helpers.  */
char * xpar_strdup(const char * s) {
  sz n = xpar_strlen(s) + 1;
  char * c = xpar_alloc_raw(n);
  for (sz i = 0; i < n; i++) c[i] = s[i];
  return c;
}
char * xpar_strndup(const char * s, sz n) {
  sz len = 0;
  while (len < n && s[len]) len++;
  char * c = xpar_alloc_raw(len + 1);
  for (sz i = 0; i < len; i++) c[i] = s[i];
  c[len] = '\0';
  return c;
}
/*  Unprefixed mem* for gcc-emitted calls; x86_64 uses rep movsb/stosb/cmpsb.  */

#if defined(__x86_64__)

void * memcpy(void * d, const void * s, sz n) {
  void * ret = d;
  __asm__ volatile ("rep movsb"
    : "+D"(d), "+S"(s), "+c"(n)
    :
    : "memory");
  return ret;
}

void * memset(void * d, int c, sz n) {
  void * ret = d;
  __asm__ volatile ("rep stosb"
    : "+D"(d), "+c"(n)
    : "a"((unsigned char) c)
    : "memory");
  return ret;
}

void * memmove(void * d, const void * s, sz n) {
  void * ret = d;
  if ((const unsigned char *) d < (const unsigned char *) s ||
      (const unsigned char *) d >= (const unsigned char *) s + n) {
    __asm__ volatile ("rep movsb"
      : "+D"(d), "+S"(s), "+c"(n)
      :
      : "memory");
  } else {
    d = (unsigned char *) d + n - 1;
    s = (const unsigned char *) s + n - 1;
    __asm__ volatile ("std\n\trep movsb\n\tcld"
      : "+D"(d), "+S"(s), "+c"(n)
      :
      : "memory");
  }
  return ret;
}

int memcmp(const void * a, const void * b, sz n) {
  if (n == 0) return 0;
  const unsigned char * ap = a;
  const unsigned char * bp = b;
  __asm__ volatile ("cld\n\trepe cmpsb"
    : "+S"(ap), "+D"(bp), "+c"(n)
    :
    : "memory", "cc");
  /*  Both pointers land one past the last compared byte.  */
  return (int) ap[-1] - (int) bp[-1];
}

#else  /*  portable fallback; dead path on mingw-w64.  */

void * memcpy(void * d, const void * s, sz n) {
  unsigned char * dp = d; const unsigned char * sp = s;
  while (n--) *dp++ = *sp++;
  return d;
}
void * memmove(void * d, const void * s, sz n) {
  unsigned char * dp = d; const unsigned char * sp = s;
  if (dp < sp || dp >= sp + n) { while (n--) *dp++ = *sp++; }
  else { dp += n; sp += n; while (n--) *--dp = *--sp; }
  return d;
}
void * memset(void * d, int c, sz n) {
  unsigned char * dp = d; unsigned char v = (unsigned char) c;
  while (n--) *dp++ = v;
  return d;
}
int memcmp(const void * a, const void * b, sz n) {
  const unsigned char * ap = a; const unsigned char * bp = b;
  while (n--) { if (*ap != *bp) return (int)*ap - (int)*bp; ap++; bp++; }
  return 0;
}

#endif

/*  xpar_mem... and xpar_str... are macros over __builtin_* that route to
    the symbols above.  */

/*  ============================================================================
    Numeric parsing: decimal only
    ============================================================================  */

int xpar_parse_i64(const char * s, i64 * out) {
  if (!s || !*s) return -1;
  int neg = 0;
  if (*s == '-') { neg = 1; s++; } else if (*s == '+') { s++; }
  if (!*s) return -1;
  u64 v = 0;
  while (*s) {
    if (*s < '0' || *s > '9') return -1;
    u64 nv = v * 10 + (u64)(*s - '0');
    if (nv < v) return -1; /*  overflow  */
    v = nv; s++;
  }
  if (neg) {
    if (v > (u64) INT64_MAX + 1) return -1;
    /*  v == INT64_MAX+1 maps to INT64_MIN; -(i64)v for that value is
        implementation-defined, so handle it explicitly.  */
    *out = v == (u64) INT64_MAX + 1 ? INT64_MIN : -(i64) v;
  } else {
    if (v > (u64) INT64_MAX) return -1;
    *out = (i64) v;
  }
  return 0;
}
int xpar_parse_u64(const char * s, u64 * out) {
  if (!s || !*s) return -1;
  if (*s == '+') s++;
  if (!*s) return -1;
  u64 v = 0;
  while (*s) {
    if (*s < '0' || *s > '9') return -1;
    u64 nv = v * 10 + (u64)(*s - '0');
    if (nv < v) return -1;
    v = nv; s++;
  }
  *out = v;
  return 0;
}

/*  ============================================================================
    Mini-printf
    ============================================================================  */

typedef struct {
  char * buf;
  sz     cap;
  sz     pos;       /*  total chars that WOULD be written, for snprintf-sizing  */
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
  char tmp[32]; int n = 0;
  const char * digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
  if (v == 0 && prec == 0) n = 0;
  else do { tmp[n++] = digits[v % base]; v /= base; } while (v);
  int len = n;
  /*  POSIX: explicit precision on d/i/o/u/x/X voids '0' flag.  */
  int pad_zero = prec > len ? prec - len : 0;
  int total = len + pad_zero;
  int pad_sp = width > total ? width - total : 0;
  if (!(flags & F_MINUS) && !(flags & F_ZERO)) emit_pad(c, pad_sp, ' ');
  if (!(flags & F_MINUS) && (flags & F_ZERO) && prec < 0) emit_pad(c, pad_sp, '0');
  emit_pad(c, pad_zero, '0');
  while (n) emit_c(c, tmp[--n]);
  if (flags & F_MINUS) emit_pad(c, pad_sp, ' ');
}

static void emit_int(fmt_ctx * c, i64 v, int width, int prec, int flags) {
  char sign = 0;
  u64 uv;
  if (v < 0) { sign = '-'; uv = (u64)(-(v+1)) + 1; }
  else if (flags & F_PLUS)  { sign = '+'; uv = (u64) v; }
  else if (flags & F_SPACE) { sign = ' '; uv = (u64) v; }
  else { uv = (u64) v; }
  if (sign) {
    if (width > 0 && !(flags & F_MINUS) && !(flags & F_ZERO)) {
      /*  space-pad, then sign  */
      char tmp[32]; int n = 0;
      if (uv == 0 && prec == 0) n = 0;
      else { u64 x = uv; do { tmp[n++] = '0' + (x % 10); x /= 10; } while (x); }
      int len = n;
      int pad_zero = prec > len ? prec - len : 0;
      int total = 1 + len + pad_zero;
      int pad_sp = width > total ? width - total : 0;
      emit_pad(c, pad_sp, ' ');
      emit_c(c, sign);
      emit_pad(c, pad_zero, '0');
      while (n) emit_c(c, tmp[--n]);
      return;
    } else if (flags & F_ZERO) {
      emit_c(c, sign);
      width = width > 0 ? width - 1 : 0;
    } else {
      emit_c(c, sign);
      width = width > 0 ? width - 1 : 0;
    }
  }
  emit_uint(c, uv, 10, 0, width, prec, flags);
}

static void emit_double(fmt_ctx * c, double v, int width, int prec, int flags) {
  if (prec < 0) prec = 6;
  char sign = 0;
  if (v < 0)                      { sign = '-'; v = -v; }
  else if (flags & F_PLUS)        { sign = '+'; }
  else if (flags & F_SPACE)       { sign = ' '; }
  u64 ip = (u64) v;
  double frac = v - (double) ip;
  /*  Round to prec fractional digits.  */
  u64 mult = 1;
  for (int i = 0; i < prec; i++) mult *= 10;
  u64 fp = (u64)(frac * (double) mult + 0.5);
  if (fp >= mult) { ip++; fp -= mult; }

  char ibuf[24]; int in = 0;
  if (ip == 0) ibuf[in++] = '0';
  else { u64 x = ip; while (x) { ibuf[in++] = '0' + (x % 10); x /= 10; } }
  int total = in + (prec > 0 ? (1 + prec) : 0) + (sign ? 1 : 0);
  int pad = width > total ? width - total : 0;
  if (!(flags & F_MINUS) && !(flags & F_ZERO)) emit_pad(c, pad, ' ');
  if (sign) emit_c(c, sign);
  if (!(flags & F_MINUS) && (flags & F_ZERO)) emit_pad(c, pad, '0');
  while (in) emit_c(c, ibuf[--in]);
  if (prec > 0) {
    emit_c(c, '.');
    /*  Emit fp zero-padded to prec digits.  */
    char fbuf[24]; int fn = 0;
    u64 x = fp;
    for (int i = 0; i < prec; i++) { fbuf[fn++] = '0' + (x % 10); x /= 10; }
    while (fn) emit_c(c, fbuf[--fn]);
  }
  if (flags & F_MINUS) emit_pad(c, pad, ' ');
}

int xpar_vsnprintf(char * buf, sz cap, const char * fmt, va_list ap) {
  fmt_ctx c = { buf, cap, 0 };
  while (*fmt) {
    if (*fmt != '%') { emit_c(&c, *fmt++); continue; }
    fmt++;
    int flags = 0;
    for (;; fmt++) {
      if (*fmt == '-') flags |= F_MINUS;
      else if (*fmt == '+') flags |= F_PLUS;
      else if (*fmt == ' ') flags |= F_SPACE;
      else if (*fmt == '0') flags |= F_ZERO;
      else if (*fmt == '#') flags |= F_HASH;
      else break;
    }
    int width = 0;
    if (*fmt == '*') { width = va_arg(ap, int); fmt++; }
    else while (*fmt >= '0' && *fmt <= '9') { width = width * 10 + (*fmt - '0'); fmt++; }
    int prec = -1;
    if (*fmt == '.') {
      fmt++; prec = 0;
      if (*fmt == '*') { prec = va_arg(ap, int); fmt++; }
      else while (*fmt >= '0' && *fmt <= '9') { prec = prec * 10 + (*fmt - '0'); fmt++; }
    }
    int longness = 0;  /*  0 = int, 1 = long, 2 = long long, 3 = size_t  */
    if (*fmt == 'z') { longness = 3; fmt++; }
    else if (*fmt == 'l') {
      fmt++;
      if (*fmt == 'l') { longness = 2; fmt++; }
      else longness = 1;
    } else if (*fmt == 'h') {
      fmt++;
      if (*fmt == 'h') fmt++;  /*  treat hh/h as int anyway  */
    }
    char spec = *fmt; if (spec) fmt++;
    switch (spec) {
      case 'd': case 'i': {
        i64 v;
        if (longness == 2) v = va_arg(ap, long long);
        else if (longness == 1) v = va_arg(ap, long);
        else if (longness == 3) v = (i64) va_arg(ap, ptrdiff_t);
        else v = va_arg(ap, int);
        emit_int(&c, v, width, prec, flags);
        break;
      }
      case 'u': {
        u64 v;
        if (longness == 2) v = va_arg(ap, unsigned long long);
        else if (longness == 1) v = va_arg(ap, unsigned long);
        else if (longness == 3) v = (u64) va_arg(ap, sz);
        else v = va_arg(ap, unsigned);
        emit_uint(&c, v, 10, 0, width, prec, flags);
        break;
      }
      case 'x': case 'X': {
        u64 v;
        if (longness == 2) v = va_arg(ap, unsigned long long);
        else if (longness == 1) v = va_arg(ap, unsigned long);
        else if (longness == 3) v = (u64) va_arg(ap, sz);
        else v = va_arg(ap, unsigned);
        emit_uint(&c, v, 16, spec == 'X', width, prec, flags);
        break;
      }
      case 'p': {
        void * p = va_arg(ap, void *);
        emit_str(&c, "0x", 2);
        emit_uint(&c, (u64)(uintptr_t) p, 16, 0, 16, -1, F_ZERO);
        break;
      }
      case 'c': {
        int ch = va_arg(ap, int);
        int pad = width > 1 ? width - 1 : 0;
        if (!(flags & F_MINUS)) emit_pad(&c, pad, ' ');
        emit_c(&c, (char) ch);
        if (flags & F_MINUS) emit_pad(&c, pad, ' ');
        break;
      }
      case 's': {
        const char * s = va_arg(ap, const char *);
        if (!s) s = "(null)";
        sz slen = 0; while (s[slen] && (prec < 0 || slen < (sz) prec)) slen++;
        int pad = width > (int) slen ? width - (int) slen : 0;
        if (!(flags & F_MINUS)) emit_pad(&c, pad, ' ');
        emit_str(&c, s, slen);
        if (flags & F_MINUS) emit_pad(&c, pad, ' ');
        break;
      }
      case 'f': {
        double v = va_arg(ap, double);
        emit_double(&c, v, width, prec, flags);
        break;
      }
      case '%': emit_c(&c, '%'); break;
      default: emit_c(&c, '%'); if (spec) emit_c(&c, spec); break;
    }
  }
  if (c.buf && c.cap > 0) {
    if (c.pos < c.cap) c.buf[c.pos] = '\0';
    else c.buf[c.cap - 1] = '\0';
  }
  return (int) c.pos;
}

int xpar_snprintf(char * buf, sz cap, const char * fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = xpar_vsnprintf(buf, cap, fmt, ap);
  va_end(ap);
  return r;
}

int xpar_asprintf(char ** out, const char * fmt, ...) {
  va_list ap; va_start(ap, fmt);
  va_list ap2; va_copy(ap2, ap);
  int n = xpar_vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (n < 0) { va_end(ap2); *out = NULL; return -1; }
  *out = xpar_alloc_raw((sz) n + 1);
  xpar_vsnprintf(*out, (sz) n + 1, fmt, ap2);
  va_end(ap2);
  return n;
}

/*  ============================================================================
    Console-aware output
    ============================================================================  */

static void write_raw(xpar_file * f, const char * s, sz n) {
#if !defined(XPAR_WIN_LEGACY)
  if (xpar_is_tty(f)) {
    /*  Console: UTF-8 -> UTF-16 + WriteConsoleW for Unicode rendering.  */
    int wn = MultiByteToWideChar(CP_UTF8, 0, s, (int) n, NULL, 0);
    if (wn > 0) {
      wchar_t stack[512]; wchar_t * w = stack;
      if ((sz) wn > sizeof(stack)/sizeof(stack[0]))
        w = HeapAlloc(GetProcessHeap(), 0, (sz) wn * sizeof(wchar_t));
      if (w) {
        MultiByteToWideChar(CP_UTF8, 0, s, (int) n, w, wn);
        DWORD written;
        WriteConsoleW(f->h, w, (DWORD) wn, &written, NULL);
        if (w != stack) HeapFree(GetProcessHeap(), 0, w);
        return;
      }
    }
    /*  Fall through to raw WriteFile on conversion failure.  */
  }
#endif
  DWORD written;
  sz off = 0;
  while (off < n) {
    DWORD chunk = (n - off > 0x40000000) ? 0x40000000 : (DWORD)(n - off);
    if (!WriteFile(f->h, s + off, chunk, &written, NULL)) break;
    off += written;
    if (written == 0) break;
  }
}

int xpar_vfprintf(xpar_file * f, const char * fmt, va_list ap) {
  char stack[1024];
  va_list ap2; va_copy(ap2, ap);
  int n = xpar_vsnprintf(stack, sizeof(stack), fmt, ap);
  if (n < (int) sizeof(stack)) {
    write_raw(f, stack, (sz) n);
    va_end(ap2);
    return n;
  }
  /*  overflow the stack buffer; heap-allocate and retry.  */
  char * big = xpar_alloc_raw((sz) n + 1);
  xpar_vsnprintf(big, (sz) n + 1, fmt, ap2);
  va_end(ap2);
  write_raw(f, big, (sz) n);
  xpar_free(big);
  return n;
}

int xpar_fprintf(xpar_file * f, const char * fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int r = xpar_vfprintf(f, fmt, ap);
  va_end(ap);
  return r;
}

int xpar_fputs(const char * s, xpar_file * f) {
  sz n = xpar_strlen(s);
  write_raw(f, s, n);
  return (int) n;
}

/*  ============================================================================
    Process, errors, time
    ============================================================================  */

__attribute__((noreturn)) void xpar_exit(int code) { ExitProcess((UINT) code); }

static __declspec(thread) char tls_errbuf[256];

const char * xpar_strerror(int err) {
#if defined(XPAR_WIN_LEGACY)
  /*  Win9x: FormatMessageA, CP_ACP bytes returned verbatim.  */
  DWORD n = FormatMessageA(
    FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
    NULL, (DWORD) err, 0, tls_errbuf, sizeof(tls_errbuf) - 1, NULL);
  if (n == 0) {
    xpar_snprintf(tls_errbuf, sizeof(tls_errbuf),
                  "Windows error %d", err);
    return tls_errbuf;
  }
  while (n > 0 && (tls_errbuf[n-1] == '\r' || tls_errbuf[n-1] == '\n' ||
                   tls_errbuf[n-1] == '.'  || tls_errbuf[n-1] == ' ')) n--;
  tls_errbuf[n] = '\0';
  return tls_errbuf;
#else
  wchar_t wbuf[160];
  DWORD n = FormatMessageW(
    FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
    NULL, (DWORD) err, 0, wbuf, sizeof(wbuf)/sizeof(wbuf[0]) - 1, NULL);
  if (n == 0) {
    xpar_snprintf(tls_errbuf, sizeof(tls_errbuf),
                  "Windows error %d", err);
    return tls_errbuf;
  }
  /*  Strip trailing CR/LF/period that FormatMessageW tends to append.  */
  while (n > 0 && (wbuf[n-1] == L'\r' || wbuf[n-1] == L'\n' ||
                   wbuf[n-1] == L'.'  || wbuf[n-1] == L' ')) n--;
  wbuf[n] = L'\0';
  int r = WideCharToMultiByte(CP_UTF8, 0, wbuf, -1,
                              tls_errbuf, sizeof(tls_errbuf), NULL, NULL);
  if (r <= 0) {
    xpar_snprintf(tls_errbuf, sizeof(tls_errbuf), "Windows error %d", err);
  }
  return tls_errbuf;
#endif
}

int xpar_errno(void) { return (int) GetLastError(); }

u64 xpar_usec_now(void) {
  static LARGE_INTEGER freq = { .QuadPart = 0 };
  if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
  LARGE_INTEGER ctr; QueryPerformanceCounter(&ctr);
  /*  Overflow-safe microseconds conversion.  */
  u64 q = (u64) ctr.QuadPart / (u64) freq.QuadPart;
  u64 r = (u64) ctr.QuadPart % (u64) freq.QuadPart;
  return q * 1000000ULL + r * 1000000ULL / (u64) freq.QuadPart;
}

/*  ============================================================================
    Thread / mutex / condvar primitives (Win32-backed)
    ============================================================================  */

#if defined(XPAR_WIN_LEGACY)
struct xpar_thread { char _unused; };
#else
struct xpar_thread { HANDLE h; };
#endif

#if _WIN32_WINNT >= 0x0600

/*  Vista+: SRWLOCK + CONDITION_VARIABLE, self-contained.  */

struct xpar_mutex { SRWLOCK            lock; };
struct xpar_cond  { CONDITION_VARIABLE cv;   };

xpar_mutex * xpar_mutex_new(void) {
  xpar_mutex * m = xpar_alloc_raw(sizeof(*m));
  InitializeSRWLock(&m->lock);
  return m;
}
void xpar_mutex_free(xpar_mutex * m) { xpar_free(m); }
void xpar_mutex_lock  (xpar_mutex * m) { AcquireSRWLockExclusive(&m->lock); }
void xpar_mutex_unlock(xpar_mutex * m) { ReleaseSRWLockExclusive(&m->lock); }

xpar_cond * xpar_cond_new(void) {
  xpar_cond * c = xpar_alloc_raw(sizeof(*c));
  InitializeConditionVariable(&c->cv);
  return c;
}
void xpar_cond_free(xpar_cond * c) { xpar_free(c); }
void xpar_cond_wait(xpar_cond * c, xpar_mutex * m) {
  SleepConditionVariableSRW(&c->cv, &m->lock, INFINITE, 0);
}
void xpar_cond_signal   (xpar_cond * c) { WakeConditionVariable   (&c->cv); }
void xpar_cond_broadcast(xpar_cond * c) { WakeAllConditionVariable(&c->cv); }

#else

/*  Win9x: threading disabled (cpu_count=1), primitives are stubs.  */

struct xpar_mutex { char _unused; };
struct xpar_cond  { char _unused; };

xpar_mutex * xpar_mutex_new(void)  { return xpar_alloc_raw(sizeof(xpar_mutex)); }
void xpar_mutex_free(xpar_mutex * m) { xpar_free(m); }
void xpar_mutex_lock  (xpar_mutex * m) { (void) m; }
void xpar_mutex_unlock(xpar_mutex * m) { (void) m; }

xpar_cond * xpar_cond_new(void) { return xpar_alloc_raw(sizeof(xpar_cond)); }
void xpar_cond_free(xpar_cond * c) { xpar_free(c); }
void xpar_cond_wait(xpar_cond * c, xpar_mutex * m) {
  (void) c; (void) m;
  FATAL("xpar_cond_wait: threading disabled on Win9x legacy build");
}
void xpar_cond_signal   (xpar_cond * c) { (void) c; }
void xpar_cond_broadcast(xpar_cond * c) { (void) c; }

#endif

#if defined(XPAR_WIN_LEGACY)

/*  Legacy: thread entry points unreachable, stubbed.  */
xpar_thread * xpar_thread_start(void (*fn)(void *), void * ctx) {
  (void) fn; (void) ctx;
  FATAL("xpar_thread_start: threading disabled on Win9x legacy build");
}
void xpar_thread_join(xpar_thread * t) {
  (void) t;
  FATAL("xpar_thread_join: threading disabled on Win9x legacy build");
}

#else

struct xpar_thread_shim { void (*fn)(void *); void * ctx; };
static DWORD WINAPI xpar_thread_trampoline(LPVOID p) {
  struct xpar_thread_shim s = *(struct xpar_thread_shim *) p;
  xpar_free(p);
  s.fn(s.ctx);
  return 0;
}
xpar_thread * xpar_thread_start(void (*fn)(void *), void * ctx) {
  struct xpar_thread_shim * s = xpar_alloc_raw(sizeof(*s));
  s->fn = fn; s->ctx = ctx;
  xpar_thread * t = xpar_alloc_raw(sizeof(*t));
  t->h = CreateThread(NULL, 0, xpar_thread_trampoline, s, 0, NULL);
  if (!t->h) FATAL("CreateThread");
  return t;
}
void xpar_thread_join(xpar_thread * t) {
  WaitForSingleObject(t->h, INFINITE);
  CloseHandle(t->h);
  xpar_free(t);
}

#endif

int xpar_cpu_count(void) {
#if defined(XPAR_WIN_LEGACY)
  /*  Pin to 1 on Win9x; keeps threadpool serial regardless of host.  */
  return 1;
#else
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (int) si.dwNumberOfProcessors;
#endif
}

/*  Wide-argv splitter implementing MSDN's C command-line parsing rules.
    Emits UTF-8 argv via HeapAlloc; argv[0] verbatim.  */

#if defined(XPAR_WIN_LEGACY)

/*  Legacy: parse the A command line; bytes pass through as CP_ACP.  */

static int split_cmdline_a(const char * cmd, char *** out_argv) {
  int argc = 0;
  for (int pass = 0; pass < 2; pass++) {
    const char * p = cmd;
    char ** argv = NULL;
    if (pass == 1) {
      argv = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY,
                       (sz)(argc + 1) * sizeof(char *));
      if (!argv) return -1;
    }
    argc = 0;
    while (*p) {
      while (*p == ' ' || *p == '\t') p++;
      if (!*p) break;
      char * buf = NULL; sz blen = 0, bcap = 0;
      if (pass == 1) {
        bcap = 64;
        buf = HeapAlloc(GetProcessHeap(), 0, bcap);
        if (!buf) { HeapFree(GetProcessHeap(), 0, argv); return -1; }
      }
      int in_quote = 0;
      while (*p) {
        if (!in_quote && (*p == ' ' || *p == '\t')) break;
        if (*p == '\\') {
          int nbs = 0;
          while (*p == '\\') { nbs++; p++; }
          if (*p == '"') {
            int slashes = nbs / 2;
            if (pass == 1) {
              for (int i = 0; i < slashes; i++) {
                if (blen + 1 >= bcap) {
                  bcap *= 2;
                  buf = HeapReAlloc(GetProcessHeap(), 0, buf, bcap);
                }
                buf[blen++] = '\\';
              }
            }
            if (nbs & 1) {
              if (pass == 1) {
                if (blen + 1 >= bcap) {
                  bcap *= 2;
                  buf = HeapReAlloc(GetProcessHeap(), 0, buf, bcap);
                }
                buf[blen++] = '"';
              }
              p++;
            } else {
              in_quote = !in_quote;
              p++;
            }
          } else {
            if (pass == 1) {
              for (int i = 0; i < nbs; i++) {
                if (blen + 1 >= bcap) {
                  bcap *= 2;
                  buf = HeapReAlloc(GetProcessHeap(), 0, buf, bcap);
                }
                buf[blen++] = '\\';
              }
            }
          }
        } else if (*p == '"') {
          if (in_quote && p[1] == '"') {
            if (pass == 1) {
              if (blen + 1 >= bcap) {
                bcap *= 2;
                buf = HeapReAlloc(GetProcessHeap(), 0, buf, bcap);
              }
              buf[blen++] = '"';
            }
            p += 2;
          } else {
            in_quote = !in_quote;
            p++;
          }
        } else {
          if (pass == 1) {
            if (blen + 1 >= bcap) {
              bcap *= 2;
              buf = HeapReAlloc(GetProcessHeap(), 0, buf, bcap);
            }
            buf[blen++] = *p;
          }
          p++;
        }
      }
      if (pass == 1) {
        buf[blen] = '\0';
        argv[argc] = buf;
      }
      argc++;
    }
    if (pass == 1) { *out_argv = argv; return argc; }
  }
  return -1;
}

int xpar_win_utf8_argv(int * argc_out, char *** argv_out) {
  int argc = split_cmdline_a(GetCommandLineA(), argv_out);
  if (argc < 0) return -1;
  *argc_out = argc;
  return 0;
}

#else

static int split_cmdline_utf16(const wchar_t * cmd, wchar_t *** out_wargv) {
  /*  Two passes: count args, then fill.  */
  int argc = 0;
  for (int pass = 0; pass < 2; pass++) {
    const wchar_t * p = cmd;
    wchar_t ** wargv = NULL;
    if (pass == 1) {
      wargv = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY,
                        (sz)(argc + 1) * sizeof(wchar_t *));
      if (!wargv) return -1;
    }
    argc = 0;
    while (*p) {
      while (*p == L' ' || *p == L'\t') p++;
      if (!*p) break;
      /*  Start a new arg. Collect into a buffer on pass 1.  */
      wchar_t * buf = NULL; sz blen = 0, bcap = 0;
      if (pass == 1) {
        bcap = 64;
        buf = HeapAlloc(GetProcessHeap(), 0, bcap * sizeof(wchar_t));
        if (!buf) { HeapFree(GetProcessHeap(), 0, wargv); return -1; }
      }
      int in_quote = 0;
      while (*p) {
        if (!in_quote && (*p == L' ' || *p == L'\t')) break;
        if (*p == L'\\') {
          int nbs = 0;
          while (*p == L'\\') { nbs++; p++; }
          if (*p == L'"') {
            int slashes = nbs / 2;
            if (pass == 1) {
              for (int i = 0; i < slashes; i++) {
                if (blen + 1 >= bcap) {
                  bcap *= 2;
                  buf = HeapReAlloc(GetProcessHeap(), 0, buf, bcap * sizeof(wchar_t));
                }
                buf[blen++] = L'\\';
              }
            }
            if (nbs & 1) {
              if (pass == 1) {
                if (blen + 1 >= bcap) {
                  bcap *= 2;
                  buf = HeapReAlloc(GetProcessHeap(), 0, buf, bcap * sizeof(wchar_t));
                }
                buf[blen++] = L'"';
              }
              p++;
            } else {
              in_quote = !in_quote;
              p++;
            }
          } else {
            if (pass == 1) {
              for (int i = 0; i < nbs; i++) {
                if (blen + 1 >= bcap) {
                  bcap *= 2;
                  buf = HeapReAlloc(GetProcessHeap(), 0, buf, bcap * sizeof(wchar_t));
                }
                buf[blen++] = L'\\';
              }
            }
          }
        } else if (*p == L'"') {
          if (in_quote && p[1] == L'"') {
            if (pass == 1) {
              if (blen + 1 >= bcap) {
                bcap *= 2;
                buf = HeapReAlloc(GetProcessHeap(), 0, buf, bcap * sizeof(wchar_t));
              }
              buf[blen++] = L'"';
            }
            p += 2;
          } else {
            in_quote = !in_quote;
            p++;
          }
        } else {
          if (pass == 1) {
            if (blen + 1 >= bcap) {
              bcap *= 2;
              buf = HeapReAlloc(GetProcessHeap(), 0, buf, bcap * sizeof(wchar_t));
            }
            buf[blen++] = *p;
          }
          p++;
        }
      }
      if (pass == 1) {
        buf[blen] = L'\0';
        wargv[argc] = buf;
      }
      argc++;
    }
    if (pass == 1) { *out_wargv = wargv; return argc; }
  }
  return -1;
}

int xpar_win_utf8_argv(int * argc_out, char *** argv_out) {
  wchar_t ** wargv;
  int wargc = split_cmdline_utf16(GetCommandLineW(), &wargv);
  if (wargc < 0) return -1;
  char ** argv = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY,
                           (sz)(wargc + 1) * sizeof(char *));
  if (!argv) return -1;
  for (int i = 0; i < wargc; i++) {
    argv[i] = wide_to_utf8_heap(wargv[i]);
    HeapFree(GetProcessHeap(), 0, wargv[i]);
    if (!argv[i]) {
      for (int j = 0; j < i; j++) HeapFree(GetProcessHeap(), 0, argv[j]);
      HeapFree(GetProcessHeap(), 0, argv);
      HeapFree(GetProcessHeap(), 0, wargv);
      return -1;
    }
  }
  HeapFree(GetProcessHeap(), 0, wargv);
  *argc_out = wargc;
  *argv_out = argv;
  return 0;
}

#endif  /*  XPAR_WIN_LEGACY vs. modern argv split  */

/*  ============================================================================
    Entry point. Replaces mainCRTStartup entirely. Link with
      -nostartfiles -Wl,-e,xpar_entry -nodefaultlibs -lgcc -lkernel32
    ============================================================================  */

__attribute__((noreturn)) void __cdecl xpar_entry(void) {
  xpar_host_init();
  int argc; char ** argv;
  if (xpar_win_utf8_argv(&argc, &argv) < 0) ExitProcess(2);
  int rc = xpar_main(argc, argv);
  ExitProcess((UINT) rc);
}
