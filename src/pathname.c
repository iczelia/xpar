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

/*  A stage name is unguessable rather than merely unused: a predictable
    one in a shared directory is a symlink waiting to be planted, and
    O_EXCL only helps when the attacker has to win the race first.  */
#define STAGE_TRIES  32
#define STAGE_RANDOM 8

bool xpar_path_sep(char c) { return c == '/' || c == '\\'; }

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
  char * p = (char *) xpar_malloc(d + sep + n + 1);
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

xpar_file * xpar_stage_open(const char * stem, int flags, int nofollow,
                            char ** out) {
  for (u32 attempt = 0; attempt < STAGE_TRIES; attempt++) {
    u8 rnd[STAGE_RANDOM];
    char hex[2 * STAGE_RANDOM + 1];
    char * path = NULL;
    xpar_file * f;
    xpar_random_bytes(rnd, sizeof rnd);
    xpar_hex(hex, rnd, sizeof rnd);
    xpar_asprintf(&path, "%s%s.tmp", stem, hex);
    f = xpar_open(path, flags | XPAR_O_CREAT | XPAR_O_EXCL);
    if (f) {
      (void) xpar_set_mode(path, nofollow, 0600);
      *out = path;
      return f;
    }
    xpar_free(path);
  }
  return NULL;
}

char * xpar_stage_dir(const char * stem) {
  for (u32 attempt = 0; attempt < STAGE_TRIES; attempt++) {
    u8 rnd[STAGE_RANDOM];
    char hex[2 * STAGE_RANDOM + 1];
    char * path = NULL;
    xpar_random_bytes(rnd, sizeof rnd);
    xpar_hex(hex, rnd, sizeof rnd);
    xpar_asprintf(&path, "%s%s", stem, hex);
    if (xpar_mkdir(path, 0700) == 0) return path;
    xpar_free(path);
  }
  return NULL;
}
