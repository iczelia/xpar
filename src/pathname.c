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

/*  Path-name manipulation and private staging files.  */

#include "pathname.h"

#include "port-fs.h"

/*  Random stage names prevent pre-planted symlinks in shared directories.  */
#define STAGE_TRIES  32
#define STAGE_RANDOM 8

/*  Backslashes are separators only on Windows and DOS.  */
bool xpar_path_sep(char c) {
#if defined(_WIN32) || defined(__CYGWIN__) || defined(__MSDOS__)
  return c == '/' || c == '\\';
#else
  return c == '/';
#endif
}

const char * xpar_path_base(const char * path) {
  const char * last = path;
  for (const char * p = path; *p; p++)
    if (xpar_path_sep(*p)) last = p + 1;
  return last;
}

char * xpar_path_dir(const char * path) {
  sz n = (sz) (xpar_path_base(path) - path);
  if (!n) return xpar_strdup(".");
  /*  "/name" keeps the root separator; "dir/name" drops its own.  */
  return xpar_strndup(path, n > 1 ? n - 1 : 1);
}

char * xpar_path_join_n(const char * dir, const char * name, u32 n) {
  sz d = dir ? xpar_strlen(dir) : 0;
  sz sep = d && !xpar_path_sep(dir[d - 1]);
  char * p = (char *) xpar_alloc_raw(d + sep + n + 1);
  if (d) xpar_memcpy(p, dir, d);
  if (sep) p[d] = '/';
  if (n) xpar_memcpy(p + d + sep, name, n);
  p[d + sep + n] = 0;
  return p;
}

char * xpar_path_join(const char * dir, const char * name) {
  return xpar_path_join_n(dir, name, (u32) xpar_strlen(name));
}

bool xpar_path_ends_with(const char * s, const char * suffix) {
  sz n = xpar_strlen(s), m = xpar_strlen(suffix);
  return n >= m && xpar_strcmp(s + n - m, suffix) == 0;
}

char * xpar_path_norm(const char * path) {
  sz n = xpar_strlen(path), i = 0, j = 0;
  char * out = (char *) xpar_alloc_raw(n + 2);
  /*  Remove leading "./" and repeated separators.  */
  while (path[i] == '.' && xpar_path_sep(path[i + 1])) {
    i += 2;
    while (xpar_path_sep(path[i])) i++;
  }
  for (; i < n; i++) {
    if (xpar_path_sep(path[i]) && j && xpar_path_sep(out[j - 1])) continue;
    out[j++] = path[i];
  }
  while (j > 1 && xpar_path_sep(out[j - 1])) j--;
  if (!j) out[j++] = '.';
  out[j] = 0;
  return out;
}

bool xpar_path_same(const char * a, const char * b) {
  char * x = xpar_path_norm(a), * y = xpar_path_norm(b);
  bool eq = xpar_strcmp(x, y) == 0;
  xpar_free(x);  xpar_free(y);
  return eq;
}

/*  Escape control bytes; results live in a rotating buffer.  */
char * xpar_name_escape(const char * s) {
  static char * ring[XPAR_ESCAPE_RING];
  static u32 at = 0;
  sz n = s ? xpar_strlen(s) : 0, i, j = 0;
  char * out = (char *) xpar_alloc_raw(4 * n + 1);
  static const char d[] = "0123456789ABCDEF";
  for (i = 0; i < n; i++) {
    u8 b = (u8) s[i];
    if (b >= 0x20 && b != 0x7F) { out[j++] = (char) b;  continue; }
    out[j++] = '\\';  out[j++] = 'x';
    out[j++] = d[b >> 4];  out[j++] = d[b & 15];
  }
  out[j] = 0;
  xpar_free(ring[at]);
  ring[at] = out;
  at = (at + 1) % XPAR_ESCAPE_RING;
  return out;
}

bool xpar_scan_digits(const char * s, sz * at, sz end) {
  sz i = *at;
  if (i == end || s[i] < '0' || s[i] > '9') return false;
  while (i < end && s[i] >= '0' && s[i] <= '9') i++;
  *at = i;
  return true;
}

static const char * scan_root;

void xpar_path_scan_set(const char * dir) { scan_root = dir; }
const char * xpar_path_scan(void) { return scan_root; }

static bool path_is_file(const char * p) {
  xpar_stat_t st;
  return xpar_lstat(p, &st) == 0 && !st.is_dir;
}

char * xpar_path_vol(const char * dir, const char * name) {
  char * p = xpar_path_join(dir, name), * q;
  if (!scan_root || path_is_file(p)) return p;
  q = xpar_path_join(scan_root, name);
  if (path_is_file(q)) { xpar_free(p);  return q; }
  xpar_free(q);
  return p;
}

/*  Trim the final component to leave room for the staging suffix.  */
char * xpar_stage_stem(const char * stem, sz suffix) {
  const char * base = xpar_path_base(stem);
  sz dirlen = (sz) (base - stem), blen = xpar_strlen(base);
  sz room = suffix < XPAR_COMPONENT_MAX ? XPAR_COMPONENT_MAX - suffix : 1;
  char * out;
  if (blen <= room) return xpar_strdup(stem);
  out = (char *) xpar_alloc_raw(dirlen + room + 1);
  if (dirlen) xpar_memcpy(out, stem, dirlen);
  xpar_memcpy(out + dirlen, base, room);
  out[dirlen + room] = 0;
  return out;
}

xpar_file * xpar_stage_open(const char * stem, int flags, int nofollow,
                            char ** out) {
  char * trimmed = xpar_stage_stem(stem, 2 * STAGE_RANDOM + 4);
  for (u32 attempt = 0; attempt < STAGE_TRIES; attempt++) {
    u8 rnd[STAGE_RANDOM];
    char hex[2 * STAGE_RANDOM + 1];
    char * path = NULL;
    xpar_file * f;
    xpar_random_bytes(rnd, sizeof rnd);
    xpar_hex(hex, rnd, sizeof rnd);
    xpar_asprintf(&path, "%s%s.tmp", trimmed, hex);
    f = xpar_open(path, flags | XPAR_O_CREAT | XPAR_O_EXCL |
                        XPAR_O_PRIVATE);
    if (f) {
      (void) xpar_set_mode(path, nofollow, 0600);
      *out = path;
      xpar_free(trimmed);
      return f;
    }
    /*  Only a name collision is worth another random name.  */
    { xpar_stat_t st;
      bool collided = xpar_lstat(path, &st) == 0;
      xpar_free(path);
      if (!collided) break; }
  }
  xpar_free(trimmed);
  return NULL;
}

char * xpar_stage_dir(const char * stem) {
  char * trimmed = xpar_stage_stem(stem, 2 * STAGE_RANDOM);
  for (u32 attempt = 0; attempt < STAGE_TRIES; attempt++) {
    u8 rnd[STAGE_RANDOM];
    char hex[2 * STAGE_RANDOM + 1];
    char * path = NULL;
    xpar_random_bytes(rnd, sizeof rnd);
    xpar_hex(hex, rnd, sizeof rnd);
    xpar_asprintf(&path, "%s%s", trimmed, hex);
    if (xpar_mkdir(path, 0700) == 0) { xpar_free(trimmed);  return path; }
    /*  Only a name collision is worth another random name.  */
    { xpar_stat_t st;
      bool collided = xpar_lstat(path, &st) == 0;
      xpar_free(path);
      if (!collided) break; }
  }
  xpar_free(trimmed);
  return NULL;
}
