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

/*  Shared UTF conversion and extended-length Windows paths.
    Include after <windows.h> and "common.h".  */

#ifndef XPAR_PORT_WIN_PATH_H
#define XPAR_PORT_WIN_PATH_H

/*  FILETIME counts 100 ns ticks from 1601; the offset to the Unix epoch.  */
#define WIN_EPOCH_DELTA_100NS  116444736000000000ULL

#if !defined(XPAR_WIN_LEGACY)

/*  Returned blocks use xpar_alloc_raw and xpar_free.  */

static wchar_t * xpar_win_wide(const char * s) {
  int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
  wchar_t * w;
  if (n <= 0) return NULL;
  w = (wchar_t *) xpar_alloc_raw((sz) n * sizeof(wchar_t));
  if (MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n) <= 0) {
    xpar_free(w);
    return NULL;
  }
  return w;
}

/*  wlen == -1 includes the input terminator; output is always terminated.  */
static char * xpar_win_utf8(const wchar_t * w, int wlen) {
  int n = WideCharToMultiByte(CP_UTF8, 0, w, wlen, NULL, 0, NULL, NULL);
  char * s;
  if (n < 0) return NULL;
  s = (char *) xpar_alloc_raw((sz) n + 1);
  if (n && WideCharToMultiByte(CP_UTF8, 0, w, wlen, s, n, NULL, NULL) <= 0) {
    xpar_free(s);
    return NULL;
  }
  s[n] = '\0';
  return s;
}

/*  Normalize to \\?\ form; reject pre-prefixed namespace paths.  */
static wchar_t * xpar_win_path(const char * s) {
  wchar_t * raw = xpar_win_wide(s), * full, * out;
  DWORD n, got;
  if (!raw) return NULL;
  if ((raw[0] == L'\\' && raw[1] == L'\\' &&
       (raw[2] == L'?' || raw[2] == L'.') && raw[3] == L'\\') ||
      (raw[0] == L'\\' && raw[1] == L'?' && raw[2] == L'?' &&
       raw[3] == L'\\')) {
    xpar_free(raw);
    SetLastError(ERROR_INVALID_NAME);
    return NULL;
  }
  n = GetFullPathNameW(raw, 0, NULL, NULL);
  if (!n || n > 32768u) {
    xpar_free(raw);
    return NULL;
  }
  full = (wchar_t *) xpar_alloc_raw(((sz) n + 1) * sizeof(*full));
  got = GetFullPathNameW(raw, n + 1, full, NULL);
  xpar_free(raw);
  if (!got || got > n) {
    xpar_free(full);
    return NULL;
  }
  if (full[0] == L'\\' && full[1] == L'\\') {
    out = (wchar_t *) xpar_alloc_raw(((sz) got + 7) * sizeof(*out));
    xpar_memcpy(out, L"\\\\?\\UNC\\", 8 * sizeof(*out));
    xpar_memcpy(out + 8, full + 2, ((sz) got - 1) * sizeof(*out));
  } else {
    out = (wchar_t *) xpar_alloc_raw(((sz) got + 5) * sizeof(*out));
    xpar_memcpy(out, L"\\\\?\\", 4 * sizeof(*out));
    xpar_memcpy(out + 4, full, ((sz) got + 1) * sizeof(*out));
  }
  xpar_free(full);
  return out;
}

#endif

#endif
