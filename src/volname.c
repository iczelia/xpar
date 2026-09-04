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

static char dos_fold(char c) {
  if (c >= 'a' && c <= 'z') return (char) (c - 'a' + 'A');
  if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')
    return c;
  return '_';
}

/*  Preserve the directory and map only the requested leaf.  Four stem bytes
    leave room for generation markers in the DOS 8.3 public namespace.  */
static char * dos_base4(const char * base) {
  const char * leaf = xpar_path_base(base);
  sz dir = (sz) (leaf - base), n = xpar_strlen(leaf), i;
  char * out = (char *) xpar_alloc_raw(dir + 5);
  if (dir) xpar_memcpy(out, base, dir);
  Fi(4, out[dir + i] = i < n ? dos_fold(leaf[i]) : '_');
  out[dir + 4] = 0;
  return out;
}

#if defined(XPAR_DOS) || defined(__MSDOS__)
static void dos_generation(u32 gen) {
  FATAL_UNLESS(gen <= 9,
               "DOS 8.3 names can represent generations 0 through 9");
}
#endif

char * xpar_vname_index(const char * base, u32 gen) {
  char * s;
#if defined(XPAR_DOS) || defined(__MSDOS__)
  char * b = dos_base4(base);
  dos_generation(gen);
  if (!gen) xpar_asprintf(&s, "%s.XPA", b);
  else      xpar_asprintf(&s, "%s.XG%" PRIu32, b, gen);
  xpar_free(b);
#else
  if (!gen) xpar_asprintf(&s, "%s" XPAR_EXT, base);
  else      xpar_asprintf(&s, "%s.g%03" PRIu32 XPAR_EXT, base, gen);
#endif
  return s;
}

char * xpar_vname_recovery(const char * base, u32 gen, u64 first, u64 count,
                           int wfirst, int wcount, u32 ordinal) {
  char * s;
#if defined(XPAR_DOS) || defined(__MSDOS__)
  char * b = dos_base4(base);
  sz n = xpar_strlen(b);
  (void) first;  (void) count;  (void) wfirst;  (void) wcount;
  dos_generation(gen);
  FATAL_UNLESS(ordinal < 100,
               "DOS 8.3 names can represent at most 100 recovery volumes");
  if (!gen) xpar_asprintf(&s, "%s.V%02" PRIu32, b, ordinal);
  else { b[n - 1] = 'G';  xpar_asprintf(&s, "%s%" PRIu32 ".V%02" PRIu32, b, gen, ordinal); }
  xpar_free(b);
#else
  (void) ordinal;
  if (!gen)
    xpar_asprintf(&s, "%s.v%0*" PRIu64 "+%0*" PRIu64 XPAR_EXT, base,
                  wfirst, first,
                  wcount, count);
  else
    xpar_asprintf(&s, "%s.g%03" PRIu32 ".v%0*" PRIu64 "+%0*" PRIu64 XPAR_EXT, base, gen,
                  wfirst, first,
                  wcount, count);
#endif
  return s;
}

char * xpar_vname_data(const char * base, u32 gen, u32 index, int width) {
  char * s;
#if defined(XPAR_DOS) || defined(__MSDOS__)
  char * b = dos_base4(base);
  sz n = xpar_strlen(b);
  (void) width;
  dos_generation(gen);
  FATAL_UNLESS(index < 100,
               "DOS 8.3 names can represent at most 100 data volumes");
  if (!gen) xpar_asprintf(&s, "%s.D%02" PRIu32, b, index);
  else { b[n - 1] = 'G';  xpar_asprintf(&s, "%s%" PRIu32 ".D%02" PRIu32, b, gen, index); }
  xpar_free(b);
#else
  if (!gen) xpar_asprintf(&s, "%s.d%0*" PRIu32, base, width, index);
  else      xpar_asprintf(&s, "%s.g%03" PRIu32 ".d%0*" PRIu32, base, gen, width, index);
#endif
  return s;
}

char * xpar_vname_label(const char * data_name) {
  char * s;
#if defined(XPAR_DOS) || defined(__MSDOS__)
  sz n = xpar_strlen(data_name);
  s = xpar_strdup(data_name);
  if (n >= 4 && s[n - 4] == '.' &&
      (s[n - 3] == 'D' || s[n - 3] == 'd')) s[n - 3] = 'L';
#else
  xpar_asprintf(&s, "%s" XPAR_EXT, data_name);
#endif
  return s;
}

char * xpar_vname_undo(const char * base, u32 generation) {
  char * s;
#if defined(XPAR_DOS) || defined(__MSDOS__)
  char * b = dos_base4(base);
  dos_generation(generation);
  if (generation) xpar_asprintf(&s, "%s.XU%" PRIu32, b, generation);
  else            xpar_asprintf(&s, "%s.XPU", b);
  xpar_free(b);
#else
  if (generation)
    xpar_asprintf(&s, "%s.g%03" PRIu32 ".xparundo", base, generation);
  else
    xpar_asprintf(&s, "%s.xparundo", base);
#endif
  return s;
}

char * xpar_vname_maint(const char * base) {
  char * s;
#if defined(XPAR_DOS) || defined(__MSDOS__)
  char * b = dos_base4(base);
  xpar_asprintf(&s, "%s.XPM", b);  xpar_free(b);
#else
  xpar_asprintf(&s, "%s.xparmaint", base);
#endif
  return s;
}

char * xpar_vname_cache(const char * base) {
  char * s;
#if defined(XPAR_DOS) || defined(__MSDOS__)
  char * b = dos_base4(base);
  xpar_asprintf(&s, "%s.XPI", b);  xpar_free(b);
#else
  xpar_asprintf(&s, "%s.xparidx", base);
#endif
  return s;
}

void xpar_vname_widths(u64 max_first, u64 max_count,
                       int * wfirst, int * wcount) {
  *wfirst = MAX(xpar_digits10(max_first), 2);
  *wcount = MAX(xpar_digits10(max_count), 2);
}

static char fold(char c) { return c >= 'A' && c <= 'Z' ? (char) (c + 32) : c; }

static bool dos_equal_n(const char * a, const char * b, sz n) {
  sz i;
  Fi(n, if (fold(a[i]) != fold(b[i])) return false);
  return true;
}

static void dos_stem4(const char * stem, char out[4]) {
  char * b = dos_base4(stem);
  const char * leaf = xpar_path_base(b);
  xpar_memcpy(out, leaf, 4);
  xpar_free(b);
}

static bool dos_2digits(const char * p) {
  return p[0] >= '0' && p[0] <= '9' && p[1] >= '0' && p[1] <= '9';
}

static bool vname_has_xpa_ext(const char * name) {
  sz n = xpar_strlen(name), i;
  static const char ext[] = XPAR_EXT;
  if (n <= XPAR_EXT_LEN) return false;
  Fi(XPAR_EXT_LEN, if (fold(name[n - XPAR_EXT_LEN + i]) != ext[i]) return false);
  return true;
}

static bool vname_is_dos_index(const char * name, const char * stem) {
  char b[4];
  sz n = xpar_strlen(name);
  dos_stem4(stem, b);
  if (n == 8 && dos_equal_n(name, b, 4) && name[4] == '.' &&
      fold(name[5]) == 'x' && fold(name[6]) == 'p' && fold(name[7]) == 'a')
    return true;
  return n == 8 && dos_equal_n(name, b, 4) && name[4] == '.' &&
         fold(name[5]) == 'x' && fold(name[6]) == 'g' &&
         name[7] >= '1' && name[7] <= '9';
}

static bool vname_is_dos_recovery(const char * name, const char * stem) {
  char b[4];
  sz n = xpar_strlen(name);
  dos_stem4(stem, b);
  if (n == 8 && dos_equal_n(name, b, 4) && name[4] == '.' &&
      fold(name[5]) == 'v' && dos_2digits(name + 6)) return true;
  return n == 9 && dos_equal_n(name, b, 3) && fold(name[3]) == 'g' &&
         name[4] >= '1' && name[4] <= '9' && name[5] == '.' &&
         fold(name[6]) == 'v' && dos_2digits(name + 7);
}

bool xpar_vname_has_ext(const char * name) {
  sz n = xpar_strlen(name);
#if defined(XPAR_DOS) || defined(__MSDOS__)
  if (n >= 4 && name[n - 4] == '.' && fold(name[n - 3]) == 'v' &&
      dos_2digits(name + n - 2)) return true;
#endif
  if (n >= 4 && name[n - 4] == '.' && fold(name[n - 3]) == 'x' &&
      fold(name[n - 2]) == 'g' && name[n - 1] >= '1' && name[n - 1] <= '9')
    return true;
  return vname_has_xpa_ext(name);
}

bool xpar_vname_is_undo(const char * name) {
  sz n = xpar_strlen(name);
  if (n >= sizeof ".xparundo" - 1 &&
      !xpar_strcmp(name + n - (sizeof ".xparundo" - 1), ".xparundo"))
    return true;
  if (n < 4 || name[n - 4] != '.' || fold(name[n - 3]) != 'x') return false;
  return (fold(name[n - 2]) == 'p' && fold(name[n - 1]) == 'u') ||
         (fold(name[n - 2]) == 'u' && name[n - 1] >= '1' &&
                                      name[n - 1] <= '9');
}

/*  Both recognisers work on the part before the extension and consume it
    left to right, so an unexpected byte anywhere fails the whole name
    rather than being read as part of the next field.  */

bool xpar_vname_is_index(const char * name, const char * stem) {
  sz n = xpar_strlen(name), p = xpar_strlen(stem), i;
  if (vname_is_dos_index(name, stem)) return true;
  if (!vname_has_xpa_ext(name) || n - XPAR_EXT_LEN < p ||
      xpar_strncmp(name, stem, p)) return false;
  n -= XPAR_EXT_LEN;
  if (p == n) return true;
  if (name[p++] != '.' || p == n || name[p] != 'g') return false;
  i = p + 1;
  return xpar_scan_digits(name, &i, n) && i == n;
}

bool xpar_vname_is_member(const char * name, const char * stem) {
  sz n = xpar_strlen(name), p = xpar_strlen(stem), i;
  if (xpar_vname_is_index(name, stem)) return true;
  if (vname_is_dos_recovery(name, stem)) return true;
  if (!vname_has_xpa_ext(name) || n - XPAR_EXT_LEN < p ||
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

i64 xpar_vname_gen_of(const char * name, const char * stem) {
  char b[4];
  sz n = xpar_strlen(name), p = xpar_strlen(stem), i;
  u64 g = 0;
  bool dos_name = vname_is_dos_index(name, stem) ||
                  vname_is_dos_recovery(name, stem);
  dos_stem4(stem, b);
  if (!xpar_vname_is_member(name, stem)) return -1;
  if (n == 8 && dos_equal_n(name, b, 4) && name[4] == '.' &&
      fold(name[5]) == 'x' && fold(name[6]) == 'g')
    return name[7] - '0';
  if (n == 9 && dos_equal_n(name, b, 3) && fold(name[3]) == 'g')
    return name[4] - '0';
  if (dos_name) return 0;
  n -= XPAR_EXT_LEN;
  if (p == n) return 0;
  if (name[p + 1] != 'g') return 0;   /*  stem.vAA+BB: generation zero.  */
  for (i = p + 2; i < n && name[i] >= '0' && name[i] <= '9'; i++) {
    g = g * 10 + (u64) (name[i] - '0');
    if (g > 0xFFFFFFFEu) return -1;
  }
  return (i64) g;
}

/*  A split data volume, with or without the label's extension.  */
static bool vname_is_data(const char * name, const char * stem) {
  sz n = xpar_strlen(name), p = xpar_strlen(stem), i;
  char b[4];
  dos_stem4(stem, b);
  if ((n == 8 && dos_equal_n(name, b, 4) && name[4] == '.' &&
       (fold(name[5]) == 'd' || fold(name[5]) == 'l') &&
       dos_2digits(name + 6)) ||
      (n == 9 && dos_equal_n(name, b, 3) && fold(name[3]) == 'g' &&
       name[4] >= '1' && name[4] <= '9' && name[5] == '.' &&
       (fold(name[6]) == 'd' || fold(name[6]) == 'l') &&
       dos_2digits(name + 7))) return true;
  if (xpar_strncmp(name, stem, p)) return false;
  if (xpar_vname_has_ext(name)) n -= XPAR_EXT_LEN;
  if (n <= p) return false;
  i = p;
  if (name[i++] != '.') return false;
  if (i < n && name[i] == 'g') {
    i++;
    if (!xpar_scan_digits(name, &i, n)) return false;
    if (i == n || name[i++] != '.') return false;
  }
  if (i == n || name[i++] != 'd') return false;
  return xpar_scan_digits(name, &i, n) && i == n;
}

bool xpar_vname_is_output(const char * path, const char * base) {
  char * pdir, * bdir;
  const char * leaf, * stem;
  sz p;
  bool same;
  if (!path || !base || !base[0]) return false;
  pdir = xpar_path_dir(path);  bdir = xpar_path_dir(base);
  same = xpar_path_same(pdir, bdir);
  xpar_free(pdir);  xpar_free(bdir);
  if (!same) return false;
  leaf = xpar_path_base(path);  stem = xpar_path_base(base);
  {
    char b[4];
    sz n = xpar_strlen(leaf);
    dos_stem4(stem, b);
    if (xpar_vname_is_member(leaf, stem)) return true;
    if (n == 8 && dos_equal_n(leaf, b, 4) && leaf[4] == '.' &&
        (fold(leaf[5]) == 'd' || fold(leaf[5]) == 'l') &&
        dos_2digits(leaf + 6)) return true;
    if (n == 8 && dos_equal_n(leaf, b, 4) && leaf[4] == '.' &&
        fold(leaf[5]) == 'x' && fold(leaf[6]) == 'p' &&
        fold(leaf[7]) == 'i') return true;
    if (n == 9 && dos_equal_n(leaf, b, 3) && fold(leaf[3]) == 'g' &&
        leaf[4] >= '1' && leaf[4] <= '9' && leaf[5] == '.' &&
        (fold(leaf[6]) == 'd' || fold(leaf[6]) == 'l') &&
        dos_2digits(leaf + 7)) return true;
  }
  /*  The output staging directory, but not a staged pipe input.  */
  if (!xpar_strncmp(leaf, ".xpar-create-", 13)) return true;
  if (xpar_vname_is_index(leaf, stem) || xpar_vname_is_member(leaf, stem) ||
      vname_is_data(leaf, stem)) return true;
  p = xpar_strlen(stem);
  if (xpar_strncmp(leaf, stem, p)) return false;
  /*  The chunk cache and the recovery spill hang off the same stem.  */
  return !xpar_strncmp(leaf + p, ".xparidx", 8) ||
         !xpar_strncmp(leaf + p, ".xpar-tmp", 9);
}
