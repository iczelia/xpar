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

/*  Volume naming and volume-name recognition.  */

#include "volname.h"

#include "pathname.h"

char * xpar_vname_index(const char * base, u32 gen) {
  char * s;
  if (!gen) xpar_asprintf(&s, "%s" XPAR_EXT, base);
  else      xpar_asprintf(&s, "%s.g%03" PRIu32 XPAR_EXT, base, gen);
  return s;
}

char * xpar_vname_recovery(const char * base, u32 gen, u64 first, u64 count,
                           int wfirst, int wcount) {
  char * s;
  if (!gen)
    xpar_asprintf(&s, "%s.v%0*" PRIu64 "+%0*" PRIu64 XPAR_EXT, base,
                  wfirst, first,
                  wcount, count);
  else
    xpar_asprintf(&s, "%s.g%03" PRIu32 ".v%0*" PRIu64 "+%0*" PRIu64 XPAR_EXT, base, gen,
                  wfirst, first,
                  wcount, count);
  return s;
}

char * xpar_vname_data(const char * base, u32 gen, u32 index, int width) {
  char * s;
  if (!gen) xpar_asprintf(&s, "%s.d%0*" PRIu32, base, width, index);
  else      xpar_asprintf(&s, "%s.g%03" PRIu32 ".d%0*" PRIu32, base, gen, width, index);
  return s;
}

char * xpar_vname_label(const char * data_name) {
  char * s;
  xpar_asprintf(&s, "%s" XPAR_EXT, data_name);
  return s;
}

void xpar_vname_widths(u64 max_first, u64 max_count,
                       int * wfirst, int * wcount) {
  *wfirst = MAX(xpar_digits10(max_first), 2);
  *wcount = MAX(xpar_digits10(max_count), 2);
}

static char fold(char c) { return c >= 'A' && c <= 'Z' ? (char) (c + 32) : c; }

bool xpar_vname_has_ext(const char * name) {
  sz n = xpar_strlen(name), i;
  static const char ext[] = XPAR_EXT;
  if (n <= XPAR_EXT_LEN) return false;
  for (i = 0; i < XPAR_EXT_LEN; i++)
    if (fold(name[n - XPAR_EXT_LEN + i]) != ext[i]) return false;
  return true;
}

/*  Both recognisers work on the part before the extension and consume it
    left to right, so an unexpected byte anywhere fails the whole name
    rather than being read as part of the next field.  */

bool xpar_vname_is_index(const char * name, const char * stem) {
  sz n = xpar_strlen(name), p = xpar_strlen(stem), i;
  if (!xpar_vname_has_ext(name) || n - XPAR_EXT_LEN < p ||
      xpar_strncmp(name, stem, p)) return false;
  n -= XPAR_EXT_LEN;
  if (p == n) return true;
  if (name[p++] != '.' || p == n || name[p] != 'g') return false;
  i = p + 1;
  return xpar_scan_digits(name, &i, n) && i == n;
}

bool xpar_vname_is_member(const char * name, const char * stem) {
  sz n = xpar_strlen(name), p = xpar_strlen(stem), i;
  if (!xpar_vname_has_ext(name) || n - XPAR_EXT_LEN < p ||
      xpar_strncmp(name, stem, p)) return false;
  n -= XPAR_EXT_LEN;
  if (p == n) return true;
  if (name[p++] != '.' || p == n) return false;
  if (name[p] == 'g') {
    i = p + 1;
    if (!xpar_scan_digits(name, &i, n)) return false;
    if (i == n) return true;
    if (name[i++] != '.' || i == n || name[i] != 'v') return false;
    p = i;
  }
  if (name[p] != 'v') return false;
  i = p + 1;
  if (!xpar_scan_digits(name, &i, n) || i == n || name[i++] != '+')
    return false;
  return xpar_scan_digits(name, &i, n) && i == n;
}
