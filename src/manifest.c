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

/*  Manifest construction, validation, and occurrence indexing.  */

#include "manifest.h"

#include "blake3.h"
#include "chunk.h"
#include "cli.h"
#include "common.h"
#include "port-fs.h"

static void * dup_bytes(const void * p, u32 n) {
  u8 * q = (u8 *) xpar_alloc_raw((sz) n + 1);
  if (n) xpar_memcpy(q, p, n);
  q[n] = 0;
  return q;
}

/*  Grow to at least `need` elements, doubling.  */
static void * grow_to(void * p, u32 * cap, u32 need, sz elem) {
  u64 c = *cap ? *cap : 16;
  if (need <= *cap) return p;
  while (c < need) c *= 2;
  if (c > 0x7FFFFFFFu) FATAL("Too many manifest entries.");
  *cap = (u32) c;
  return xpar_realloc(p, (sz) c * elem);
}

/*  Power-of-two open-addressing table size for `n` live keys, kept at
    or under half load. Refuses past 2^31 rather than wrapping a mask.  */
static u32 table_mask(u32 n) {
  u64 size = 16;
  while (size < (u64) n * 2 + 2) size *= 2;
  if (size > 0x80000000u) FATAL("Too many manifest entries.");
  return (u32) (size - 1);
}

int xpar_name_cmp(const char * a, u32 alen, const char * b, u32 blen) {
  u32 n = alen < blen ? alen : blen;
  int c = n ? xpar_memcmp(a, b, n) : 0;
  if (c) return c;
  return alen < blen ? -1 : (alen > blen ? 1 : 0);
}

static u8 fold_ascii(u8 c) {
  return (u8) (c >= 'A' && c <= 'Z' ? c + 32 : c);
}

/*  Case folding is ASCII-only.  */
static int name_cmp_fold(const char * a, u32 alen, const char * b,
                         u32 blen) {
  u32 n = alen < blen ? alen : blen, i;
  for (i = 0; i < n; i++) {
    u8 x = fold_ascii((u8) a[i]), y = fold_ascii((u8) b[i]);
    if (x != y) return x < y ? -1 : 1;
  }
  return alen < blen ? -1 : (alen > blen ? 1 : 0);
}

/*  UTF-8..  */

bool xpar_utf8_valid(const u8 * p, u32 n) {
  u32 i = 0;
  while (i < n) {
    u32 c = p[i], need, lo, v, j;
    if (c < 0x80) { i++;  continue; }
    if (c >= 0xC2 && c <= 0xDF)      { need = 1; lo = 0x80;    v = c & 0x1F; }
    else if (c >= 0xE0 && c <= 0xEF) { need = 2; lo = 0x800;   v = c & 0x0F; }
    else if (c >= 0xF0 && c <= 0xF4) { need = 3; lo = 0x10000; v = c & 0x07; }
    else return false;
    if (need > n - i - 1) return false;
    for (j = 0; j < need; j++) {
      u8 b = p[i + 1 + j];
      if (b < 0x80 || b > 0xBF) return false;
      v = (v << 6) | (b & 0x3F);
    }
    if (v < lo || v > 0x10FFFF) return false;
    if (v >= 0xD800 && v <= 0xDFFF) return false;
    i += need + 1;
  }
  return true;
}

/*  Path rules.  */

static bool stem_is(const char * c, const char * want, u32 n) {
  u32 i;
  for (i = 0; i < n; i++)
    if (fold_ascii((u8) c[i]) != (u8) want[i]) return false;
  return true;
}

static bool is_win_device(const char * c, u32 n) {
  u32 stem = 0;
  while (stem < n && c[stem] != '.') stem++;
  if (stem == 3)
    return stem_is(c, "con", 3) || stem_is(c, "prn", 3) ||
           stem_is(c, "aux", 3) || stem_is(c, "nul", 3);
  if (stem == 4 && c[3] >= '1' && c[3] <= '9')
    return stem_is(c, "com", 3) || stem_is(c, "lpt", 3);
  return false;
}

static xpar_path_status check_component(const char * c, u32 n, u32 flags) {
  u32 i;
  if (n == 0) return XPAR_PATH_EMPTY_COMPONENT;
  if (n == 1 && c[0] == '.') return XPAR_PATH_DOT;
  if (n == 2 && c[0] == '.' && c[1] == '.') return XPAR_PATH_DOTDOT;
  for (i = 0; i < n; i++) {
    u8 b = (u8) c[i];
    if (b <= 0x1F) return XPAR_PATH_CONTROL;
    if (flags & XPAR_PATH_WIN) {
      if (b == '\\' || b == ':' || b == '*' || b == '?' || b == '"' ||
          b == '<' || b == '>' || b == '|') return XPAR_PATH_WINCHAR;
    }
  }
  if (flags & XPAR_PATH_WIN) {
    if (c[n - 1] == '.' || c[n - 1] == ' ') return XPAR_PATH_WINTRAIL;
    if (is_win_device(c, n)) return XPAR_PATH_DEVICE;
  }
  return XPAR_PATH_OK;
}

xpar_path_status xpar_path_check(const char * name, u32 len, u32 flags) {
  u32 i, start = 0;
  if (len == 0) return XPAR_PATH_EMPTY;
  if (len > XPAR_NAME_MAX) return XPAR_PATH_TOO_LONG;
  if (name[0] == '/') return XPAR_PATH_ABSOLUTE;
  if (len >= 2 && name[1] == ':') return XPAR_PATH_DRIVE;
  if (len >= 2 && name[0] == '\\' && name[1] == '\\') return XPAR_PATH_UNC;
  if (name[len - 1] == '/') return XPAR_PATH_TRAILING_SLASH;
  for (i = 0; i <= len; i++) {
    if (i == len || name[i] == '/') {
      xpar_path_status s = check_component(name + start, i - start, flags);
      if (s != XPAR_PATH_OK) return s;
      start = i + 1;
    }
  }
  return XPAR_PATH_OK;
}

xpar_path_status xpar_symlink_target_check(const u8 * target, u32 len) {
  u32 i;
  if (len == 0) return XPAR_PATH_EMPTY;
  if (len > XPAR_EXTRA_MAX) return XPAR_PATH_TOO_LONG;
  for (i = 0; i < len; i++) if (target[i] == 0) return XPAR_PATH_CONTROL;
  return XPAR_PATH_OK;
}

const char * xpar_path_reason(xpar_path_status s) {
  switch (s) {
    case XPAR_PATH_OK:              return "conforming";
    case XPAR_PATH_EMPTY:           return "empty path";
    case XPAR_PATH_EMPTY_COMPONENT: return "empty path component";
    case XPAR_PATH_ABSOLUTE:        return "absolute path";
    case XPAR_PATH_DRIVE:           return "drive letter";
    case XPAR_PATH_UNC:             return "UNC prefix";
    case XPAR_PATH_DOT:             return "'.' component";
    case XPAR_PATH_DOTDOT:          return "'..' component";
    case XPAR_PATH_TRAILING_SLASH:  return "trailing '/'";
    case XPAR_PATH_CONTROL:         return "control byte in 0x00..0x1F";
    case XPAR_PATH_WINCHAR:         return "byte Windows cannot store";
    case XPAR_PATH_DEVICE:          return "Windows device name";
    case XPAR_PATH_WINTRAIL:        return "component ends in '.' or ' '";
    case XPAR_PATH_TOO_LONG:        return "path too long";
    case XPAR_PATH_SYMLINK:         return "path crosses a symbolic link";
  }
  return "malformed path";
}

/*  Refusing to *follow* a symlink rather than refusing to overwrite one
    is the whole of the check: the danger is an entry earlier in the same
    set planting `d -> /etc` and a later entry writing `d/passwd`, which
    a test on every component catches and a test on the final component
    alone does not.  */
char * xpar_path_resolve(const char * dir, const char * name, u32 len,
                         u32 flags, xpar_path_status * why) {
  sz dlen = dir ? xpar_strlen(dir) : 0, off = dlen ? dlen + 1 : 0;
  char * out;
  u32 i;
  xpar_path_status s = xpar_path_check(name, len, flags);
  if (s != XPAR_PATH_OK) { *why = s;  return NULL; }
  out = (char *) xpar_alloc_raw(off + len + 1);
  if (dlen) { xpar_memcpy(out, dir, dlen);  out[dlen] = '/'; }
  xpar_memcpy(out + off, name, len);
  out[off + len] = 0;
  for (i = 0; i <= len; i++) {
    if (i == len || name[i] == '/') {
      xpar_stat_t st;
      sz cut = off + i;
      char keep = out[cut];
      out[cut] = 0;
      if (xpar_lstat(out, &st) == 0 && st.is_symlink) {
        xpar_free(out);
        *why = XPAR_PATH_SYMLINK;
        return NULL;
      }
      out[cut] = keep;
    }
  }
  *why = XPAR_PATH_OK;
  return out;
}

/*  Manifest storage.  */

xpar_entry * xpar_manifest_append(xpar_manifest * m) {
  xpar_entry * e;
  if (m->count + 1 > m->cap) {
    u32 c1 = m->cap, c2 = m->cap;
    m->entry  = (xpar_entry *) grow_to(m->entry, &c1, m->count + 1,
                                       sizeof(xpar_entry));
    m->source = (char **) grow_to(m->source, &c2, m->count + 1,
                                  sizeof(char *));
    m->cap = c1;
  }
  e = &m->entry[m->count];
  m->source[m->count] = NULL;
  m->count++;
  xpar_memset(e, 0, sizeof(*e));
  e->mode        = XPAR_ABSENT_U32;
  e->posix_index = XPAR_ABSENT_U32;
  e->mtime_ns = e->atime_ns = XPAR_ABSENT_TIME;
  e->ctime_ns = e->btime_ns = XPAR_ABSENT_TIME;
  return e;
}

static void posix_rec_free(xpar_posix_rec * r) {
  u32 i;
  for (i = 0; i < r->xattr_count; i++) {
    xpar_free(r->xattrs[i].name);  xpar_free(r->xattrs[i].value);
  }
  xpar_free(r->xattrs);  xpar_free(r->owner);  xpar_free(r->group);
  xpar_memset(r, 0, sizeof(*r));
}

void xpar_manifest_free(xpar_manifest * m) {
  u32 i;
  for (i = 0; i < m->count; i++) {
    xpar_entry_free(&m->entry[i]);
    if (m->source) xpar_free(m->source[i]);
  }
  for (i = 0; i < m->posix_count; i++) posix_rec_free(&m->posix[i]);
  xpar_free(m->entry);  xpar_free(m->source);  xpar_free(m->posix);
  xpar_memset(m, 0, sizeof(*m));
}

/*  Heap-sort an index permutation without recursion or host qsort.  */

static int idx_cmp(const xpar_manifest * m, u32 a, u32 b) {
  return xpar_name_cmp(m->entry[a].name, m->entry[a].name_len,
                       m->entry[b].name, m->entry[b].name_len);
}

static void sift(const xpar_manifest * m, u32 * a, u32 root, u32 n) {
  while (1) {
    u32 c = 2 * root + 1, big;
    if (c >= n) return;
    big = c;
    if (c + 1 < n && idx_cmp(m, a[c], a[c + 1]) < 0) big = c + 1;
    if (idx_cmp(m, a[root], a[big]) >= 0) return;
    { u32 t = a[root];  a[root] = a[big];  a[big] = t; }
    root = big;
  }
}

static void sort_names(const xpar_manifest * m, u32 * a, u32 n) {
  u32 i;
  if (n < 2) return;
  for (i = n / 2; i-- > 0;) sift(m, a, i, n);
  for (i = n; i-- > 1;) {
    u32 t = a[0];  a[0] = a[i];  a[i] = t;
    sift(m, a, 0, i);
  }
}

void xpar_nameidx_build(const xpar_manifest * m, xpar_nameidx * ix) {
  u32 i;
  ix->count = m->count;
  ix->order = (u32 *) xpar_alloc_raw((m->count ? m->count : 1) *
                                     sizeof(u32));
  for (i = 0; i < m->count; i++) ix->order[i] = i;
  sort_names(m, ix->order, m->count);
}

void xpar_nameidx_free(xpar_nameidx * ix) {
  xpar_free(ix->order);  ix->order = NULL;  ix->count = 0;
}

i64 xpar_nameidx_find(const xpar_manifest * m, const xpar_nameidx * ix,
                      const char * name, u32 len) {
  u32 lo = 0, hi = ix->count;
  if (!name) return -1;
  while (lo < hi) {
    u32 mid = lo + (hi - lo) / 2, e = ix->order[mid];
    int c = xpar_name_cmp(m->entry[e].name, m->entry[e].name_len, name, len);
    if (c == 0) return (i64) e;
    if (c < 0) lo = mid + 1; else hi = mid;
  }
  return -1;
}

/*  Identities.  */

void xpar_file_id(const xpar_entry * e, const u8 * key,
                  u8 out[XPAR_SET_ID_LEN]) {
  xpar_blake3_t h;
  u8 le[8];
  if (key) xpar_blake3_init_keyed(&h, key); else xpar_blake3_init(&h);
  xpar_blake3_update(&h, "xpar2 file id v1", 16);
  xpar_wr64(le, e->length);
  xpar_blake3_update(&h, le, 8);
  xpar_blake3_update(&h, e->prefix_hash, 16);
  if (e->name_len) xpar_blake3_update(&h, e->name, e->name_len);
  xpar_blake3_final(&h, out, XPAR_SET_ID_LEN);
}

void xpar_set_id_begin(xpar_set_id_ctx * c, const u8 * key,
                       const u8 * setd_body, sz n) {
  if (key) xpar_blake3_init_keyed(&c->h, key); else xpar_blake3_init(&c->h);
  xpar_blake3_update(&c->h, "xpar2 set id v1", 15);
  xpar_blake3_update(&c->h, setd_body, n);
}

void xpar_set_id_update(xpar_set_id_ctx * c, const u8 * file_body, sz n) {
  xpar_blake3_update(&c->h, file_body, n);
}

void xpar_set_id_final(const xpar_set_id_ctx * c,
                       u8 out[XPAR_SET_ID_LEN]) {
  xpar_blake3_final(&c->h, out, XPAR_SET_ID_LEN);
}

/*  POSX packets.  */

bool xpar_posix_equal(const xpar_posix_rec * a, const xpar_posix_rec * b) {
  u32 i;
  if (a->uid != b->uid || a->gid != b->gid) return false;
  if (a->xattr_count != b->xattr_count) return false;
  if (!!a->owner != !!b->owner || !!a->group != !!b->group) return false;
  if (a->owner && xpar_strcmp(a->owner, b->owner)) return false;
  if (a->group && xpar_strcmp(a->group, b->group)) return false;
  for (i = 0; i < a->xattr_count; i++) {
    if (xpar_strcmp(a->xattrs[i].name, b->xattrs[i].name)) return false;
    if (a->xattrs[i].value_len != b->xattrs[i].value_len) return false;
    if (a->xattrs[i].value_len &&
        xpar_memcmp(a->xattrs[i].value, b->xattrs[i].value,
                    a->xattrs[i].value_len)) return false;
  }
  return true;
}

u32 xpar_posix_intern(xpar_manifest * m, const xpar_posix_rec * r) {
  xpar_posix_rec * d;
  u32 i, cap = m->posix_cap;
  for (i = 0; i < m->posix_count; i++)
    if (xpar_posix_equal(&m->posix[i], r)) return i;
  m->posix = (xpar_posix_rec *) grow_to(m->posix, &cap, m->posix_count + 1,
                                        sizeof(xpar_posix_rec));
  m->posix_cap = cap;
  d = &m->posix[m->posix_count];
  xpar_memset(d, 0, sizeof(*d));
  d->uid = r->uid;  d->gid = r->gid;
  d->owner = r->owner ? xpar_strdup(r->owner) : NULL;
  d->group = r->group ? xpar_strdup(r->group) : NULL;
  d->xattr_count = r->xattr_count;
  if (r->xattr_count) {
    d->xattrs = (xpar_xattr *) xpar_alloc_raw(r->xattr_count *
                                              sizeof(xpar_xattr));
    for (i = 0; i < r->xattr_count; i++) {
      d->xattrs[i].name      = xpar_strdup(r->xattrs[i].name);
      d->xattrs[i].value_len = r->xattrs[i].value_len;
      d->xattrs[i].value     = (u8 *) dup_bytes(r->xattrs[i].value,
                                                r->xattrs[i].value_len);
    }
  }
  return m->posix_count++;
}

/*  Walking a tree (writer side).  */

void xpar_walk_opts_default(xpar_walk_opts * o) {
  xpar_memset(o, 0, sizeof(*o));
  o->dedup     = XPAR_DEDUP_FILE;   /*  Free: content_hash exists.  */
  o->align     = XPAR_ALIGN_PACKED;
  o->dedup_chunk  = (u64) 1 << 20;
  o->dedup_memory = (u64) 64 << 20;
  o->preserve  = XPAR_PRES_DEFAULT;
  o->caps_mask = 0xFFFFFFFFu;
}

static u32 keep_mask(const xpar_walk_opts * o) {
  return o->reproducible ? (o->preserve & o->preserve_explicit)
                         : o->preserve;
}

typedef struct { u64 dev, ino, nlink; } wstat;

typedef struct { char * p; sz len, cap; } pathbuf;

typedef struct {
  xpar_manifest * m;
  const xpar_walk_opts * o;
  wstat * st;
  u32 st_cap;
  u32 caps;
  u32 caps_all;
} walker;

static void pb_reserve(pathbuf * b, sz n) {
  if (n + 1 <= b->cap) return;
  while (b->cap < n + 1) b->cap = b->cap ? b->cap * 2 : 256;
  b->p = (char *) xpar_realloc(b->p, b->cap);
}

static void pb_set(pathbuf * b, const char * s, sz n) {
  pb_reserve(b, n);
  xpar_memcpy(b->p, s, n);
  b->p[n] = 0;
  b->len = n;
}

static void pb_push(pathbuf * b, const char * comp, sz n) {
  pb_reserve(b, b->len + 1 + n);
  if (b->len) b->p[b->len++] = '/';
  xpar_memcpy(b->p + b->len, comp, n);
  b->len += n;
  b->p[b->len] = 0;
}

/*  Portable manifest-name globbing for --include and --exclude.  */
static bool glob_class(const char ** pp, u8 c) {
  const char * p = *pp;
  bool invert = false, hit = false;
  if (*p == '!' || *p == '^') { invert = true;  p++; }
  while (*p && *p != ']') {
    u8 lo, hi;
    if (*p == '\\' && p[1]) p++;
    lo = (u8) *p++;
    hi = lo;
    if (*p == '-' && p[1] && p[1] != ']') {
      p++;
      if (*p == '\\' && p[1]) p++;
      hi = (u8) *p++;
      if (hi < lo) { u8 t = lo;  lo = hi;  hi = t; }
    }
    if (c >= lo && c <= hi) hit = true;
  }
  if (*p == ']') p++;
  *pp = p;
  return invert ? !hit : hit;
}

static bool glob_match(const char * pat, const char * text) {
  const char * star = NULL, * retry = NULL;
  while (*text) {
    if (*pat == '*') {
      while (*pat == '*') pat++;
      star = pat;  retry = text;
      if (!*pat) return true;
      continue;
    }
    if (*pat == '?' || *pat == *text) { pat++;  text++;  continue; }
    if (*pat == '\\' && pat[1] && pat[1] == *text) {
      pat += 2;  text++;  continue;
    }
    if (*pat == '[') {
      const char * p = pat + 1;
      if (glob_class(&p, (u8) *text)) { pat = p;  text++;  continue; }
    }
    if (!star) return false;
    pat = star;  text = ++retry;
  }
  while (*pat == '*') pat++;
  return !*pat;
}

static bool any_glob(char * const * pat, u32 count, const char * name) {
  u32 i;
  for (i = 0; i < count; i++) if (glob_match(pat[i], name)) return true;
  return false;
}

static bool selected(const xpar_walk_opts * o, const char * name,
                     bool * excluded) {
  bool inc = any_glob(o->include, o->include_count, name);
  bool exc = any_glob(o->exclude, o->exclude_count, name);
  *excluded = exc && !inc;
  if (inc) return true;
  return !o->include_count && !exc;
}

bool xpar_manifest_name_selected(const xpar_walk_opts * o,
                                 const char * name) {
  bool excluded;
  return selected(o, name, &excluded);
}

/*  Sub-second times are dropped rather than rounded where the host does
    not carry them: a fabricated nanosecond field compares unequal
    against its own source on the next verify.  */
static void record_times(xpar_entry * e, const xpar_stat_t * st,
                         const xpar_walk_opts * o) {
  u32 keep = keep_mask(o);
  if (keep & XPAR_PRES_MTIME) e->mtime_ns = st->mtime_ns;
  if (keep & XPAR_PRES_ATIME) e->atime_ns = st->atime_ns;
  if (keep & XPAR_PRES_CTIME) e->ctime_ns = st->ctime_ns;
  if (keep & XPAR_PRES_BTIME) e->btime_ns = st->btime_ns;
}

static u16 derive_attrs(const xpar_entry * e, const xpar_stat_t * st,
                        u32 caps) {
  u16 a = (u16) ((caps & XPAR_FS_FATATTR) ? st->attrs : 0);
  if (st->mode != XPAR_MODE_NONE) {
    if (st->mode & 0111u) a |= XPAR_ATTR_EXEC;
    if (st->mode & (XPAR_MODE_SETUID | XPAR_MODE_SETGID | XPAR_MODE_STICKY))
      a |= XPAR_ATTR_SETID;
  }
  if (!xpar_utf8_valid((const u8 *) e->name, e->name_len))
    a |= XPAR_ATTR_RAWNAME;
  return a;
}

/*  One POSX record, or none at all. Ownership and xattrs are two
    --preserve tokens and one record, so a tree preserving only ownership
    still interns a record with no xattrs. */
static void record_posix(walker * w, xpar_entry * e, const char * path,
                         const xpar_stat_t * st) {
  xpar_posix_rec r;
  u32 keep = keep_mask(w->o);
  bool want_owner = (keep & XPAR_PRES_OWNER) && (w->caps & XPAR_FS_OWNER);
  bool want_xattr = (keep & (XPAR_PRES_XATTR | XPAR_PRES_XATTR_ALL)) &&
                    (w->caps & XPAR_FS_XATTR);
  char * names = NULL;
  if (!want_owner && !want_xattr) return;

  xpar_memset(&r, 0, sizeof(r));
  r.uid = r.gid = XPAR_ID_NONE;
  if (want_owner) {
    char buf[256];
    r.uid = st->uid;  r.gid = st->gid;
    if (st->uid != XPAR_ID_NONE &&
        xpar_name_of(st->uid, buf, sizeof(buf)) == 0)
      r.owner = xpar_strdup(buf);
    if (st->gid != XPAR_ID_NONE &&
        xpar_group_of(st->gid, buf, sizeof(buf)) == 0)
      r.group = xpar_strdup(buf);
  }
  if (want_xattr) {
    sz n = xpar_listxattr(path, 1, NULL, 0);
    if (n != XPAR_FS_NOSIZE && n > 0) {
      sz i = 0, got;
      names = (char *) xpar_alloc_raw(n + 1);
      got = xpar_listxattr(path, 1, names, n);
      if (got == XPAR_FS_NOSIZE || got > n) got = 0;
      names[got] = 0;
      while (i < got) {
        sz nl = xpar_strlen(names + i);
        sz vl = nl ? xpar_getxattr(path, 1, names + i, NULL, 0) : 0;
        if (nl && vl != XPAR_FS_NOSIZE && vl <= XPAR_EXTRA_MAX) {
          xpar_xattr * x;
          r.xattrs = (xpar_xattr *) xpar_realloc(r.xattrs,
                                                 (r.xattr_count + 1) *
                                                 sizeof(xpar_xattr));
          x = &r.xattrs[r.xattr_count++];
          x->name      = xpar_strdup(names + i);
          x->value_len = (u32) vl;
          x->value     = (u8 *) xpar_alloc_raw(vl + 1);
          if (vl) xpar_getxattr(path, 1, x->name, x->value, vl);
          x->value[vl] = 0;
        }
        i += nl + 1;
      }
    }
  }
  if (r.xattr_count > 1) {
    u32 i, j;
    for (i = 1; i < r.xattr_count; i++) {
      xpar_xattr t = r.xattrs[i];
      for (j = i; j && xpar_strcmp(r.xattrs[j - 1].name, t.name) > 0; j--)
        r.xattrs[j] = r.xattrs[j - 1];
      r.xattrs[j] = t;
    }
  }
  e->posix_index = xpar_posix_intern(w->m, &r);
  posix_rec_free(&r);
  xpar_free(names);
}

static bool follow_link(pathbuf * b, xpar_stat_t * st) {
  int hop;
  for (hop = 0; hop < 8 && st->is_symlink; hop++) {
    u32 n;
    char * buf = xpar_read_symlink(b->p, &n);
    if (!buf) return false;
    if (buf[0] == '/') pb_set(b, buf, (sz) n);
    else {
      sz cut = b->len;
      while (cut && b->p[cut - 1] != '/') cut--;
      b->len = cut ? cut - 1 : 0;
      b->p[b->len] = 0;
      pb_push(b, buf, (sz) n);
    }
    xpar_free(buf);
    if (xpar_lstat(b->p, st) != 0) return false;
  }
  return !st->is_symlink;
}

char * xpar_read_symlink(const char * path, u32 * length) {
  sz cap = 256;
  for (;;) {
    char * buf = (char *) xpar_malloc(cap);
    i64 n = xpar_readlink(path, buf, cap);
    if (n >= 0) {
      *length = (u32) n;
      return buf;
    }
    xpar_free(buf);
    if (cap == (sz) XPAR_EXTRA_MAX + 2) return NULL;
    cap = MIN(cap * 2, (sz) XPAR_EXTRA_MAX + 2);
  }
}

static void walk_path(walker * w, pathbuf * disk, const char * name,
                      u32 name_len, int depth);

static void walk_dir(walker * w, pathbuf * disk, const char * name,
                     u32 name_len, int depth) {
  xpar_dir * d = xpar_opendir(disk->p);
  const xpar_dirent * de;
  char ** kids = NULL;
  u32 nk = 0, cap = 0, i;
  if (!d) FATAL_PERROR(disk->p);
  while ((de = xpar_readdir(d)) != NULL) {
    kids = (char **) grow_to(kids, &cap, nk + 1, sizeof(char *));
    kids[nk++] = xpar_strdup(de->name);
  }
  xpar_closedir(d);
  for (i = 0; i < nk; i++) {
    sz keep = disk->len, kl = xpar_strlen(kids[i]);
    char * sub;
    u32 sub_len;
    if (name_len) {
      sub_len = (u32) (name_len + 1 + kl);
      sub = (char *) xpar_alloc_raw((sz) sub_len + 1);
      xpar_memcpy(sub, name, name_len);
      sub[name_len] = '/';
      xpar_memcpy(sub + name_len + 1, kids[i], kl + 1);
    } else {
      sub_len = (u32) kl;
      sub = xpar_strdup(kids[i]);
    }
    pb_push(disk, kids[i], kl);
    walk_path(w, disk, sub, sub_len, depth + 1);
    disk->len = keep;  disk->p[keep] = 0;
    xpar_free(sub);
  }
  for (i = 0; i < nk; i++) xpar_free(kids[i]);
  xpar_free(kids);
}

/*  A name the host produced but the format may not carry is a hard
    error and not a silent skip: dropping one file out of a backup set
    without saying so is the failure this program exists to prevent.  */
static void check_emit_name(const walker * w, const char * name, u32 len,
                            const char * disk) {
  xpar_path_status s = xpar_path_check(name, len, w->o->path_flags);
  if (s != XPAR_PATH_OK)
    FATAL("Cannot store '%s': %s.", disk, xpar_path_reason(s));
}

static void note_stat(walker * w, u32 idx, const xpar_stat_t * st) {
  w->st = (wstat *) grow_to(w->st, &w->st_cap, idx + 1, sizeof(wstat));
  w->st[idx].dev   = st->dev;
  w->st[idx].ino   = st->ino;
  w->st[idx].nlink = st->is_regular ? st->nlink : 0;
}

static xpar_entry * emit_entry(walker * w, const char * name, u32 name_len,
                               const char * path, const xpar_stat_t * st) {
  xpar_entry * e;
  u32 idx;
  check_emit_name(w, name, name_len, path);
  e = xpar_manifest_append(w->m);
  idx = w->m->count - 1;
  e->name     = (char *) dup_bytes(name, name_len);
  e->name_len = name_len;
  w->m->source[idx] = xpar_strdup(path);
  if (st->mode != XPAR_MODE_NONE && (keep_mask(w->o) & XPAR_PRES_MODE))
    e->mode = st->mode & XPAR_MODE_PERM;
  record_times(e, st, w->o);
  e->attrs = derive_attrs(e, st, w->caps);
  record_posix(w, e, path, st);
  note_stat(w, idx, st);
  return e;
}

static void emit_symlink(xpar_entry * e, const char * path) {
  u32 n;
  char * buf = xpar_read_symlink(path, &n);
  xpar_path_status ts;
  if (!buf) FATAL_PERROR(path);
  ts = xpar_symlink_target_check((const u8 *) buf, n);
  if (ts != XPAR_PATH_OK)
    FATAL("Cannot store the target of '%s': %s.", path,
          xpar_path_reason(ts));
  e->entry_type = XPAR_ENTRY_SYMLINK;
  e->extra      = (u8 *) buf;
  e->extra_len  = n;
}

static void walk_path(walker * w, pathbuf * disk, const char * name,
                      u32 name_len, int depth) {
  xpar_stat_t st;
  pathbuf real;
  pathbuf * path = disk;
  bool excluded = false, keep;
  if (depth > 256)
    FATAL("Directory nesting past 256 levels at '%s'.", disk->p);
  if (xpar_lstat(disk->p, &st) != 0) FATAL_PERROR(disk->p);
  xpar_memset(&real, 0, sizeof(real));
  if (st.is_symlink && w->o->follow_symlinks) {
    pb_set(&real, disk->p, disk->len);
    if (!follow_link(&real, &st)) {
      xpar_fprintf(xpar_stderr, "xpar: skipping '%s': dangling link.\n",
                   disk->p);
      xpar_free(real.p);
      return;
    }
    path = &real;
  }

  keep = !name_len || selected(w->o, name, &excluded);

  if (st.is_dir) {
    /*  A nameless root is one the path rules cannot express, such as
        ".", so its children are stored and it is not.  */
    if (name_len && keep)
      emit_entry(w, name, name_len, path->p, &st)->entry_type =
        (u16) XPAR_ENTRY_DIR;
    /*  An excluded directory prunes its subtree unless an include list may
        name a descendant explicitly. An unmatched include directory still
        has to be traversed for the same reason.  */
    if (w->o->recurse && (!excluded || w->o->include_count))
      walk_dir(w, path, name, name_len, depth);
  } else if (!name_len) {
    FATAL("Input '%s' resolves to no storable name.", path->p);
  } else if (!keep) {
    /*  A filtered special file is silent just like a filtered regular one.  */
  } else if (!st.is_regular && !st.is_symlink) {
    xpar_fprintf(xpar_stderr,
                 "xpar: skipping '%s': not a file, directory or link.\n",
                 path->p);
  } else {
    xpar_entry * e = emit_entry(w, name, name_len, path->p, &st);
    if (st.is_symlink) emit_symlink(e, path->p);
    else { e->entry_type = XPAR_ENTRY_REGULAR;  e->length = st.size; }
  }
  xpar_free(real.p);
}

static void detect_links(walker * w) {
  xpar_manifest * m = w->m;
  u32 i, mask, * table;
  if (!(w->caps_all & XPAR_FS_LINKID) || !m->count) return;
  mask  = table_mask(m->count);
  table = (u32 *) xpar_alloc_raw(((sz) mask + 1) * sizeof(u32));
  for (i = 0; i <= mask; i++) table[i] = 0xFFFFFFFFu;
  for (i = 0; i < m->count; i++) {
    xpar_entry * e = &m->entry[i];
    u64 h;
    u32 slot;
    if (e->entry_type != XPAR_ENTRY_REGULAR) continue;
    if (w->st[i].nlink <= 1) continue;   /*  The cheap pre-filter.  */
    h = w->st[i].dev * 0x9E3779B97F4A7C15ull + w->st[i].ino;
    slot = (u32) ((h ^ (h >> 32)) & mask);
    while (table[slot] != 0xFFFFFFFFu) {
      u32 c = table[slot];
      if (w->st[c].dev == w->st[i].dev && w->st[c].ino == w->st[i].ino) {
        e->entry_type   = XPAR_ENTRY_HARDLINK;
        e->extra        = (u8 *) dup_bytes(m->entry[c].name,
                                           m->entry[c].name_len);
        e->extra_len    = m->entry[c].name_len;
        e->length       = 0;   /*  The canonical's, copied in pack.  */
        e->extent_count = 0;
        m->link_count++;
        break;
      }
      slot = (slot + 1) & mask;
    }
    if (e->entry_type == XPAR_ENTRY_REGULAR) table[slot] = i;
  }
  xpar_free(table);
}

void xpar_manifest_walk(xpar_manifest * m, char * const * roots,
                        u32 root_count, const xpar_walk_opts * o) {
  walker w;
  pathbuf disk;
  u32 r, i, * order;
  xpar_entry * sorted;
  char ** src;
  wstat * ws;

  xpar_memset(&w, 0, sizeof(w));
  xpar_memset(&disk, 0, sizeof(disk));
  w.m = m;  w.o = o;  w.caps_all = 0xFFFFFFFFu;
  m->align       = o->align;
  m->slice_size  = o->align == XPAR_ALIGN_SLICE ? o->slice_size : 0;
  m->stream_base = o->stream_base;

  for (r = 0; r < root_count; r++) {
    const char * root = roots[r];
    sz rl = xpar_strlen(root), cut;
    const char * base;
    u32 blen;
    while (rl > 1 && root[rl - 1] == '/') rl--;
    cut = rl;
    while (cut && root[cut - 1] != '/') cut--;
    base = root + cut;
    blen = (u32) (rl - cut);
    /*  A root named ".", ".." or "/" cannot be emitted as a component,
        so its children are stored under their own names instead.  */
    if (blen == 0 || (blen == 1 && base[0] == '.') ||
        (blen == 2 && base[0] == '.' && base[1] == '.')) blen = 0;
    if (o->base_dir) {
      sz bl = xpar_strlen(o->base_dir);
      if (rl > bl + 1 && !xpar_strncmp(root, o->base_dir, bl) &&
          root[bl] == '/') {
        base = root + bl + 1;  blen = (u32) (rl - bl - 1);
      }
    }
    w.caps = xpar_fs_caps(root) & o->caps_mask;
    w.caps_all &= w.caps;
    pb_set(&disk, root, rl);
    walk_path(&w, &disk, base, blen, 0);
  }
  xpar_free(disk.p);
  if (!m->count) { xpar_free(w.st);  return; }

  order = (u32 *) xpar_alloc_raw(m->count * sizeof(u32));
  for (i = 0; i < m->count; i++) order[i] = i;
  sort_names(m, order, m->count);
  sorted = (xpar_entry *) xpar_alloc_raw(m->count * sizeof(xpar_entry));
  src    = (char **) xpar_alloc_raw(m->count * sizeof(char *));
  ws     = (wstat *) xpar_alloc_raw(m->count * sizeof(wstat));
  for (i = 0; i < m->count; i++) {
    sorted[i] = m->entry[order[i]];
    src[i]    = m->source[order[i]];
    ws[i]     = w.st[order[i]];
  }
  xpar_free(m->entry);  xpar_free(m->source);  xpar_free(w.st);
  xpar_free(order);
  m->entry = sorted;  m->source = src;
  w.st = ws;  w.st_cap = m->count;
  detect_links(&w);
  xpar_free(w.st);
}

/*  Packing: hashes, deduplication and extents.  */

typedef struct {
  u32 * slot;      /*  Entry index, or 0xFFFFFFFF for empty.  */
  u64 * refs;      /*  References to that canonical (--dedup-max-refs).  */
  u32   mask;
} dedup_tab;

static void dedup_init(dedup_tab * t, u32 n) {
  u32 i;
  t->mask = table_mask(n);
  t->slot = (u32 *) xpar_alloc_raw(((sz) t->mask + 1) * sizeof(u32));
  t->refs = (u64 *) xpar_calloc((sz) t->mask + 1, sizeof(u64));
  for (i = 0; i <= t->mask; i++) t->slot[i] = 0xFFFFFFFFu;
}

static void dedup_free(dedup_tab * t) {
  xpar_free(t->slot);  xpar_free(t->refs);
}

static u32 dedup_probe(const dedup_tab * t, const xpar_manifest * m,
                       const xpar_entry * e, u32 * slot_out) {
  u32 slot = (u32) (xpar_rd64(e->content_hash) & t->mask);
  while (t->slot[slot] != 0xFFFFFFFFu) {
    const xpar_entry * c = &m->entry[t->slot[slot]];
    if (c->length == e->length &&
        !xpar_memcmp(c->content_hash, e->content_hash, 32)) {
      *slot_out = slot;
      return t->slot[slot];
    }
    slot = (slot + 1) & t->mask;
  }
  *slot_out = slot;
  return 0xFFFFFFFFu;
}

static void hash_entry_file(xpar_manifest * m, u32 idx, u8 * cache,
                            u64 expected, xpar_progress_t * prog) {
  xpar_entry * e = &m->entry[idx];
  xpar_blake3_t h;
  xpar_file * f;
  u8 * buf;
  u64 total;
  sz got;
  const sz chunk = 1u << 16;
  xpar_assert(m->source[idx] != NULL);
  f = xpar_open(m->source[idx], XPAR_O_RDONLY);
  if (!f) FATAL_PERROR(m->source[idx]);
  buf = (u8 *) xpar_alloc_raw(chunk);
  xpar_blake3_init(&h);
  got = xpar_xread(f, buf, 16384);
  if (cache && (u64) got > expected)
    FATAL("'%s' changed while it was being cached; nothing was written.",
          m->source[idx]);
  if (cache && got) xpar_memcpy(cache, buf, got);
  if (got) xpar_blake3_update(&h, buf, got);
  total = got;
  if (prog) xpar_progress_tick(prog, got);
  xpar_blake3_final(&h, e->prefix_hash, 16);
  while ((got = xpar_xread(f, buf, chunk)) > 0) {
    if (cache && (u64) got > expected - total)
      FATAL("'%s' changed while it was being cached; nothing was written.",
            m->source[idx]);
    if (cache) xpar_memcpy(cache + total, buf, got);
    xpar_blake3_update(&h, buf, got);
    total += got;
    if (prog) xpar_progress_tick(prog, got);
  }
  xpar_blake3_final(&h, e->content_hash, 32);
  xpar_free(buf);
  xpar_xclose(f);
  if (cache && total != expected)
    FATAL("'%s' changed while it was being cached; nothing was written.",
          m->source[idx]);
  e->length = total;
}

static void hash_bytes(const u8 * p, u32 n, u8 out32[32], u8 out16[16]) {
  xpar_blake3_t h;
  xpar_blake3_init(&h);
  if (n) xpar_blake3_update(&h, p, n);
  xpar_blake3_final(&h, out32, 32);
  xpar_blake3_init(&h);
  xpar_blake3_final(&h, out16, 16);
}

static void set_extents(xpar_entry * e, const xpar_extent * src, u32 n) {
  xpar_free(e->extents);
  e->extent_count = n;
  e->extents = n ? (xpar_extent *) xpar_alloc_raw(n * sizeof(xpar_extent))
                 : NULL;
  if (n) xpar_memcpy(e->extents, src, n * sizeof(xpar_extent));
}

typedef struct {
  xpar_manifest * m;
  xpar_entry * e;
  const xpar_walk_opts * o;
  xpar_chunk_index * ix;
  xpar_extent * ext;
  u32 count, capacity;
  u64 * high;
  u64 bytes;
  xpar_progress_t * prog;
  bool aligned;
  bool full;
} chunk_pack;

static u64 pack_alignment(const xpar_walk_opts * o) {
  if (o->align == XPAR_ALIGN_SLICE) return o->slice_size;
  if (o->align == XPAR_ALIGN_1K) return XPAR_BLAKE3_CHUNK_LEN;
  return 0;
}

static bool cache_layout(const xpar_manifest * m,
                         const xpar_walk_opts * o, u64 * offset,
                         u64 * size) {
  u64 high = o->stream_base, q = pack_alignment(o);
  u32 i;
  for (i = 0; i < m->count; i++) {
    const xpar_entry * e = &m->entry[i];
    u64 pad = 0;
    offset[i] = 0;
    if (e->entry_type != XPAR_ENTRY_REGULAR || !e->length) continue;
    if (q) {
      pad = (high - o->stream_base) % q;
      if (pad) pad = q - pad;
    }
    if (pad > UINT64_MAX - high || e->length > UINT64_MAX - high - pad)
      return false;
    high += pad;
    offset[i] = high - o->stream_base;
    high += e->length;
  }
  *size = high - o->stream_base;
  return true;
}

static void chunk_extent(chunk_pack * c, u64 off, u32 len) {
  if (c->count && c->ext[c->count - 1].stream_offset +
                  c->ext[c->count - 1].length == off) {
    c->ext[c->count - 1].length += len;
    return;
  }
  if (c->count == c->capacity) {
    c->capacity = c->capacity ? c->capacity * 2 : 8;
    c->ext = (xpar_extent *) xpar_realloc(
      c->ext, (sz) c->capacity * sizeof(xpar_extent));
  }
  c->ext[c->count].stream_offset = off;
  c->ext[c->count].length = len;
  c->count++;
}

static bool pack_chunk(void * user, u64 file_offset, u32 len,
                       const u8 hash[16]) {
  chunk_pack * c = (chunk_pack *) user;
  xpar_chunk_slot * hit = xpar_chunk_index_find(c->ix, hash, len);
  u64 off;
  (void) file_offset;
  c->bytes += len;
  if (c->prog) xpar_progress_tick(c->prog, len);
  if (hit && c->o->dedup_max_refs &&
      hit->refs + 1 > c->o->dedup_max_refs) hit = NULL;
  if (hit) {
    off = hit->stream_offset;
    hit->refs++;
    c->m->shared_bytes += len;
    c->m->alias_extents++;
  } else {
    u64 q = pack_alignment(c->o);
    if (q && (!c->aligned || c->o->align == XPAR_ALIGN_1K)) {
      u64 pad = (*c->high - c->o->stream_base) % q;
      if (pad) *c->high += q - pad;
    }
    c->aligned = true;
    off = *c->high;
    *c->high += len;
    if (!xpar_chunk_index_put(c->ix, hash, len, off)) {
      c->full = true;
      return false;
    }
  }
  chunk_extent(c, off, len);
  return true;
}

static void pack_regular_chunks(xpar_manifest * m,
                                const xpar_walk_opts * o,
                                xpar_chunk_index * chunks, u32 i, u64 * H,
                                xpar_progress_t * prog) {
  chunk_pack c;
  u64 expected = m->entry[i].length;
  xpar_memset(&c, 0, sizeof c);
  c.m = m;  c.e = &m->entry[i];  c.o = o;  c.ix = chunks;  c.high = H;
  c.prog = prog;
  if (!xpar_chunk_file(m->source[i], o->dedup_chunk, pack_chunk, &c,
                       c.e->content_hash, c.e->prefix_hash)) {
    xpar_free(c.ext);
    if (c.full)
      FATAL_CODE(XPAR_EXIT_NOPLAN,
                 "The chunk fingerprint index exceeded --dedup-memory "
                 "(%llu bytes); raise it or --dedup-chunk.",
                 (unsigned long long) o->dedup_memory);
    FATAL_PERROR(m->source[i]);
  }
  if (c.bytes != expected) {
    xpar_free(c.ext);
    FATAL("'%s' changed while it was being chunked; nothing was written.",
          m->source[i]);
  }
  c.e->length = c.bytes;
  set_extents(c.e, c.ext, c.count);
  xpar_free(c.ext);
}

static void pack_regular(xpar_manifest * m, const xpar_walk_opts * o,
                         dedup_tab * tab, xpar_chunk_index * chunks,
                         u32 i, u64 * H, u8 * cache, u64 cache_offset,
                         xpar_progress_t * prog) {
  xpar_entry * e = &m->entry[i];
  u32 slot = 0, hit = 0xFFFFFFFFu;
  if (o->dedup == XPAR_DEDUP_CHUNK) {
    pack_regular_chunks(m, o, chunks, i, H, prog);
    m->entry_bytes += e->length;
    return;
  }
  hash_entry_file(m, i, cache ? cache + cache_offset : NULL, e->length,
                  prog);
  m->entry_bytes += e->length;
  if (e->length == 0) { set_extents(e, NULL, 0);  return; }
  if (o->dedup != XPAR_DEDUP_NONE) hit = dedup_probe(tab, m, e, &slot);
  if (hit != 0xFFFFFFFFu && o->dedup_max_refs &&
      tab->refs[slot] + 1 > o->dedup_max_refs) hit = 0xFFFFFFFFu;
  if (hit != 0xFFFFFFFFu) {
    const xpar_entry * c = &m->entry[hit];
    set_extents(e, c->extents, c->extent_count);
    m->shared_bytes  += e->length;
    m->alias_extents += c->extent_count;
    tab->refs[slot]++;
    return;
  }
  {
    xpar_extent x;
    u64 q = pack_alignment(o);
    if (q) {
      u64 pad = (*H - o->stream_base) % q;
      if (pad) *H += q - pad;
    }
    x.stream_offset = *H;  x.length = e->length;
    if (cache && e->length && x.stream_offset - o->stream_base != cache_offset)
      xpar_memmove(cache + x.stream_offset - o->stream_base,
                   cache + cache_offset, (sz) e->length);
    set_extents(e, &x, 1);
    *H += e->length;
  }
  if (o->dedup != XPAR_DEDUP_NONE) {
    u32 s2;
    dedup_probe(tab, m, e, &s2);
    tab->slot[s2] = i;  tab->refs[s2] = 1;
  }
}

static void pack_links(xpar_manifest * m, const xpar_nameidx * ix) {
  u32 i;
  for (i = 0; i < m->count; i++) {
    xpar_entry * e = &m->entry[i];
    const xpar_entry * t;
    i64 c;
    if (e->entry_type != XPAR_ENTRY_HARDLINK) continue;
    c = xpar_nameidx_find(m, ix, (const char *) e->extra, e->extra_len);
    xpar_assert(c >= 0 && (u32) c != i);
    t = &m->entry[c];
    xpar_assert(t->entry_type == XPAR_ENTRY_REGULAR);
    e->length = t->length;
    xpar_memcpy(e->content_hash, t->content_hash, 32);
    xpar_memcpy(e->prefix_hash, t->prefix_hash, 16);
    e->mode        = t->mode;      e->attrs    = t->attrs;
    e->mtime_ns    = t->mtime_ns;  e->atime_ns = t->atime_ns;
    e->ctime_ns    = t->ctime_ns;  e->btime_ns = t->btime_ns;
    e->posix_index = t->posix_index;
    set_extents(e, NULL, 0);
    m->entry_bytes += e->length;
  }
}

void xpar_manifest_pack(xpar_manifest * m, const xpar_walk_opts * o,
                        xpar_progress_t * prog) {
  xpar_nameidx ix;
  dedup_tab tab;
  xpar_chunk_index chunks;
  u8 * cache = NULL;
  u64 * cache_offset = NULL;
  u64 cache_size = 0;
  u64 H = o->stream_base;
  u32 i;
  m->stream_base   = o->stream_base;
  m->align         = o->align;
  m->slice_size    = o->align == XPAR_ALIGN_SLICE ? o->slice_size : 0;
  m->entry_bytes   = 0;
  m->shared_bytes  = 0;
  m->alias_extents = 0;
  xpar_nameidx_build(m, &ix);
  dedup_init(&tab, m->count);
  xpar_memset(&chunks, 0, sizeof chunks);
  if (o->stream_cache_out) *o->stream_cache_out = NULL;
  if (o->stream_cache_length_out) *o->stream_cache_length_out = 0;
  if (o->dedup != XPAR_DEDUP_CHUNK && o->stream_cache_out &&
      o->stream_cache_length_out && o->stream_cache_limit) {
    cache_offset = (u64 *) xpar_calloc(m->count ? m->count : 1,
                                       sizeof(u64));
    if (!cache_layout(m, o, cache_offset, &cache_size) ||
        cache_size > o->stream_cache_limit ||
        cache_size > (u64) (sz) -1) {
      xpar_free(cache_offset);  cache_offset = NULL;
    } else if (cache_size) {
      cache = (u8 *) xpar_calloc((sz) cache_size, 1);
    }
  }
  if (o->dedup == XPAR_DEDUP_CHUNK &&
      !xpar_chunk_index_init(&chunks, o->dedup_memory))
    FATAL_CODE(XPAR_EXIT_NOPLAN,
               "--dedup-memory=%llu is too small for a chunk index.",
               (unsigned long long) o->dedup_memory);

  for (i = 0; i < m->count; i++) {
    xpar_entry * e = &m->entry[i];
    switch (e->entry_type) {
      case XPAR_ENTRY_DIR:
        e->length = 0;
        hash_bytes(NULL, 0, e->content_hash, e->prefix_hash);
        break;
      case XPAR_ENTRY_SYMLINK:
        e->length = 0;
        hash_bytes(e->extra, e->extra_len, e->content_hash, e->prefix_hash);
        break;
      case XPAR_ENTRY_HARDLINK:
        break;
      default:
        pack_regular(m, o, &tab, &chunks, i, &H, cache,
                     cache_offset ? cache_offset[i] : 0, prog);
        break;
    }
  }
  m->stream_length = H - o->stream_base;
  pack_links(m, &ix);

  /*  8.4: the level written is the highest actually used, and a writer
      must not claim a level at which it shared nothing.  */
  m->dedup_level = m->alias_extents ? o->dedup : XPAR_DEDUP_NONE;
  for (i = 0; i < m->count; i++)
    xpar_file_id(&m->entry[i], NULL, m->entry[i].file_id);
  dedup_free(&tab);
  if (o->chunk_cache_out) {
    xpar_chunk_index_free(o->chunk_cache_out);
    *o->chunk_cache_out = chunks;
    xpar_memset(&chunks, 0, sizeof chunks);
  }
  if (cache) {
    u64 used = H - o->stream_base;
    cache = (u8 *) xpar_realloc(cache, (sz) (used ? used : 1));
    *o->stream_cache_out = cache;
    *o->stream_cache_length_out = used;
  }
  xpar_free(cache_offset);
  xpar_chunk_index_free(&chunks);
  xpar_nameidx_free(&ix);
}

/*  Validation (reader side).  */

const char * xpar_mf_reason(xpar_mf_status s) {
  switch (s) {
    case XPAR_MF_OK:            return "conforming";
    case XPAR_MF_PATH:          return "a path violates the path rules";
    case XPAR_MF_DUP_NAME:      return "two entries share a name";
    case XPAR_MF_TYPE:          return "unknown entry type";
    case XPAR_MF_TYPE_LENGTH:   return "non-zero length on this type";
    case XPAR_MF_TYPE_EXTENTS:  return "extents on a type that has none";
    case XPAR_MF_TYPE_EXTRA:    return "wrong extra field for this type";
    case XPAR_MF_EXTENT_LEN:    return "extent of length zero";
    case XPAR_MF_EXTENT_OVF:    return "extent range overflows 64 bits";
    case XPAR_MF_EXTENT_SUM:    return "extents do not sum to the length";
    case XPAR_MF_EXTENT_RANGE:  return "extent outside the stream range";
    case XPAR_MF_EXTENT_FWD:    return "extent names undefined bytes";
    case XPAR_MF_EXTENT_SPLIT:  return "extent straddles the defined stream";
    case XPAR_MF_STREAM_GAP:    return "the stream has a gap";
    case XPAR_MF_LINK_MISSING:  return "hard link names no entry of the set";
    case XPAR_MF_LINK_CHAIN:    return "hard link names a non-regular entry";
    case XPAR_MF_LINK_SELF:     return "hard link names itself";
    case XPAR_MF_LINK_CONTENT:  return "hard link content identity disagrees";
    case XPAR_MF_POSIX_INDEX:   return "posix_index out of range";
  }
  return "malformed manifest";
}

i64 xpar_link_target(const xpar_manifest * m, const xpar_nameidx * ix,
                     u32 entry) {
  const xpar_entry * e = &m->entry[entry];
  if (e->entry_type != XPAR_ENTRY_HARDLINK) return -1;
  return xpar_nameidx_find(m, ix, (const char *) e->extra, e->extra_len);
}

static bool alias_content_matches(const xpar_entry * a,
                                  const xpar_entry * t) {
  return a->length == t->length &&
         !xpar_memcmp(a->content_hash, t->content_hash, 32) &&
         !xpar_memcmp(a->prefix_hash, t->prefix_hash, 16);
}

static bool alias_meta_matches(const xpar_entry * a, const xpar_entry * t) {
  return a->mode == t->mode && a->attrs == t->attrs &&
         a->mtime_ns == t->mtime_ns && a->atime_ns == t->atime_ns &&
         a->ctime_ns == t->ctime_ns && a->btime_ns == t->btime_ns;
}

static xpar_mf_status check_entry(const xpar_manifest * m,
                                  const xpar_nameidx * ix,
                                  const xpar_mf_limits * lim, u32 i,
                                  xpar_mf_result * out) {
  const xpar_entry * e = &m->entry[i];
  if (xpar_path_check(e->name, e->name_len, lim->path_flags) != XPAR_PATH_OK)
    return XPAR_MF_PATH;
  if (e->posix_index != XPAR_ABSENT_U32 &&
      e->posix_index >= lim->posix_record_count) return XPAR_MF_POSIX_INDEX;
  switch (e->entry_type) {
    case XPAR_ENTRY_REGULAR:
      if (e->extra_len) return XPAR_MF_TYPE_EXTRA;
      return XPAR_MF_OK;
    case XPAR_ENTRY_DIR:
      if (e->length) return XPAR_MF_TYPE_LENGTH;
      if (e->extent_count) return XPAR_MF_TYPE_EXTENTS;
      if (e->extra_len) return XPAR_MF_TYPE_EXTRA;
      return XPAR_MF_OK;
    case XPAR_ENTRY_SYMLINK:
      if (e->length) return XPAR_MF_TYPE_LENGTH;
      if (e->extent_count) return XPAR_MF_TYPE_EXTENTS;
      if (xpar_symlink_target_check(e->extra, e->extra_len) != XPAR_PATH_OK)
        return XPAR_MF_PATH;
      return XPAR_MF_OK;
    case XPAR_ENTRY_HARDLINK: {
      i64 t;
      if (e->extent_count) return XPAR_MF_TYPE_EXTENTS;
      if (xpar_path_check((const char *) e->extra, e->extra_len,
                          lim->path_flags) != XPAR_PATH_OK)
        return XPAR_MF_PATH;
      t = xpar_nameidx_find(m, ix, (const char *) e->extra, e->extra_len);
      if (t < 0) return XPAR_MF_LINK_MISSING;
      if ((u32) t == i) return XPAR_MF_LINK_SELF;
      if (m->entry[t].entry_type != XPAR_ENTRY_REGULAR)
        return XPAR_MF_LINK_CHAIN;
      if (!alias_content_matches(e, &m->entry[t]))
        return XPAR_MF_LINK_CONTENT;
      if (!alias_meta_matches(e, &m->entry[t])) out->link_meta_mismatch++;
      return XPAR_MF_OK;
    }
    default: return XPAR_MF_TYPE;
  }
}

static bool ancestor_ok(const xpar_mf_limits * lim, u64 off, u64 len) {
  u32 i;
  u64 end;
  if (len > UINT64_MAX - off) return false;
  end = off + len;
  if (!lim->ancestor_count) return end <= lim->stream_base;
  for (i = 0; i < lim->ancestor_count; i++) {
    u64 hi;
    if (lim->ancestor[i].length > UINT64_MAX - lim->ancestor[i].base)
      return false;
    hi = lim->ancestor[i].base + lim->ancestor[i].length;
    if (off >= lim->ancestor[i].base && end <= hi)
      return true;
  }
  return false;
}

xpar_mf_status xpar_manifest_validate(const xpar_manifest * m,
                                      const xpar_mf_limits * lim,
                                      xpar_mf_result * out) {
  xpar_nameidx ix;
  xpar_mf_status s = XPAR_MF_OK;
  u64 H = lim->stream_base, own_end;
  u32 i, k;
  xpar_memset(out, 0, sizeof(*out));
  if (lim->stream_length > UINT64_MAX - lim->stream_base) {
    out->status = XPAR_MF_EXTENT_OVF;
    return out->status;
  }
  own_end = lim->stream_base + lim->stream_length;
  xpar_nameidx_build(m, &ix);

  for (i = 0; i < m->count && s == XPAR_MF_OK; i++) {
    s = check_entry(m, &ix, lim, i, out);
    if (s != XPAR_MF_OK) out->entry = i;
  }
  for (i = 1; i < ix.count && s == XPAR_MF_OK; i++) {
    const xpar_entry * a = &m->entry[ix.order[i - 1]];
    const xpar_entry * b = &m->entry[ix.order[i]];
    bool dup = xpar_name_cmp(a->name, a->name_len, b->name,
                             b->name_len) == 0;
    if (!dup && (lim->path_flags & XPAR_PATH_NOCASE))
      dup = name_cmp_fold(a->name, a->name_len, b->name, b->name_len) == 0;
    if (dup) {
      s = XPAR_MF_DUP_NAME;
      out->entry = ix.order[i - 1];  out->other = ix.order[i];
    }
  }
  xpar_nameidx_free(&ix);
  if (s != XPAR_MF_OK) { out->status = s;  return s; }

  for (i = 0; i < m->count; i++) {
    const xpar_entry * e = &m->entry[i];
    u64 sum = 0;
    for (k = 0; k < e->extent_count; k++) {
      u64 off = e->extents[k].stream_offset, len = e->extents[k].length;
      out->entry = i;  out->extent = k;
      if (len == 0) { s = XPAR_MF_EXTENT_LEN;  goto done; }
      if (off + len < off) { s = XPAR_MF_EXTENT_OVF;  goto done; }
      if (sum + len < sum) { s = XPAR_MF_EXTENT_SUM;  goto done; }
      sum += len;
      if (off < lim->stream_base) {
        if (!ancestor_ok(lim, off, len)) {
          s = XPAR_MF_EXTENT_RANGE;  goto done;
        }
        continue;
      }
      if (off + len > own_end) { s = XPAR_MF_EXTENT_RANGE;  goto done; }
      if (off == H) { H = off + len;  continue; }
      if (off < H) {
        if (off + len <= H) continue;
        s = XPAR_MF_EXTENT_SPLIT;  goto done;
      }
      {
        u64 q = lim->align == XPAR_ALIGN_SLICE ? lim->slice_size
              : lim->align == XPAR_ALIGN_1K ? XPAR_BLAKE3_CHUNK_LEN : 0;
        u64 pad = q ? (H - lim->stream_base) % q : 0;
        if (pad && q - pad <= UINT64_MAX - H && off == H + (q - pad)) {
          H = off + len;  continue;
        }
      }
      s = XPAR_MF_EXTENT_FWD;  goto done;
    }
    if (e->entry_type == XPAR_ENTRY_REGULAR && sum != e->length) {
      out->entry = i;  out->extent = 0;
      s = XPAR_MF_EXTENT_SUM;  goto done;
    }
  }
  out->entry = 0;  out->extent = 0;
  if (H != own_end) s = XPAR_MF_STREAM_GAP;
done:
  out->high_water = H;
  out->status = s;
  return s;
}

static bool occ_less(const xpar_occurrence * a, const xpar_occurrence * b) {
  if (a->stream_offset != b->stream_offset)
    return a->stream_offset < b->stream_offset;
  if (a->entry != b->entry) return a->entry < b->entry;
  return a->file_offset < b->file_offset;
}

static void occ_sift(xpar_occurrence * a, u32 root, u32 n) {
  while (1) {
    u32 c = 2 * root + 1, big;
    if (c >= n) return;
    big = c;
    if (c + 1 < n && occ_less(&a[c], &a[c + 1])) big = c + 1;
    if (!occ_less(&a[root], &a[big])) return;
    { xpar_occurrence t = a[root];  a[root] = a[big];  a[big] = t; }
    root = big;
  }
}

void xpar_occindex_build(const xpar_manifest * m, xpar_occindex * ix) {
  u64 total = 0, run = 0;
  u32 i, k, n = 0;
  for (i = 0; i < m->count; i++) total += m->entry[i].extent_count;
  if (total > 0x7FFFFFFFu) FATAL("Too many extents to index.");
  ix->count   = (u32) total;
  ix->occ     = (xpar_occurrence *) xpar_alloc_raw(
                  (total ? (sz) total : 1) * sizeof(xpar_occurrence));
  ix->max_end = (u64 *) xpar_alloc_raw((total ? (sz) total : 1) *
                                       sizeof(u64));
  for (i = 0; i < m->count; i++) {
    u64 fo = 0;
    for (k = 0; k < m->entry[i].extent_count; k++) {
      ix->occ[n].stream_offset = m->entry[i].extents[k].stream_offset;
      ix->occ[n].length        = m->entry[i].extents[k].length;
      ix->occ[n].file_offset   = fo;
      ix->occ[n].entry         = i;
      ix->occ[n].extent        = k;
      fo += m->entry[i].extents[k].length;
      n++;
    }
  }
  if (n > 1) {
    for (i = n / 2; i-- > 0;) occ_sift(ix->occ, i, n);
    for (i = n; i-- > 1;) {
      xpar_occurrence t = ix->occ[0];
      ix->occ[0] = ix->occ[i];  ix->occ[i] = t;
      occ_sift(ix->occ, 0, i);
    }
  }
  for (i = 0; i < n; i++) {
    u64 end = ix->occ[i].stream_offset + ix->occ[i].length;
    if (end > run) run = end;
    ix->max_end[i] = run;
  }
}

void xpar_occindex_free(xpar_occindex * ix) {
  xpar_free(ix->occ);  xpar_free(ix->max_end);
  ix->occ = NULL;  ix->max_end = NULL;  ix->count = 0;
}

/*  max_end is a prefix maximum and therefore non-decreasing, so the
    first occurrence that can reach `off` is found by binary search on
    it. Without that, one long extent early in the stream would drag
    every query into a scan of everything after it.  */
static u32 occ_lower(const xpar_occindex * ix, u64 off) {
  u32 lo = 0, hi = ix->count;
  while (lo < hi) {
    u32 mid = lo + (hi - lo) / 2;
    if (ix->max_end[mid] > off) hi = mid; else lo = mid + 1;
  }
  return lo;
}

static u32 occ_upper(const xpar_occindex * ix, u64 end) {
  u32 lo = 0, hi = ix->count;
  while (lo < hi) {
    u32 mid = lo + (hi - lo) / 2;
    if (ix->occ[mid].stream_offset < end) lo = mid + 1; else hi = mid;
  }
  return lo;
}

u32 xpar_occindex_overlaps(const xpar_occindex * ix, u64 off, u64 len,
                           xpar_occ_fn fn, void * user) {
  u32 i, lo, hi, hits = 0;
  if (!len) return 0;
  lo = occ_lower(ix, off);
  hi = occ_upper(ix, off + len);
  for (i = lo; i < hi; i++) {
    if (ix->occ[i].stream_offset + ix->occ[i].length <= off) continue;
    hits++;
    if (fn) fn(&ix->occ[i], user);
  }
  return hits;
}

bool xpar_occindex_canonical(const xpar_occindex * ix, u64 off,
                             xpar_occurrence * out, u64 * run) {
  u32 i, lo = occ_lower(ix, off), hi = occ_upper(ix, off + 1);
  bool got = false;
  for (i = lo; i < hi; i++) {
    const xpar_occurrence * o = &ix->occ[i];
    if (o->stream_offset > off) continue;
    if (o->stream_offset + o->length <= off) continue;
    if (!got || o->entry < out->entry ||
        (o->entry == out->entry && o->file_offset < out->file_offset)) {
      *out = *o;  got = true;
    }
  }
  if (!got) return false;
  *run = out->stream_offset + out->length - off;
  return true;
}

bool xpar_occindex_repair_source(const xpar_occindex * ix, u64 off,
                                 u64 len,
                                 bool (* intact)(const xpar_occurrence *,
                                                 void *),
                                 void * user, xpar_occurrence * out) {
  u32 i, lo, hi;
  bool got = false;
  if (!len) return false;
  lo = occ_lower(ix, off);
  hi = occ_upper(ix, off + len);
  for (i = lo; i < hi; i++) {
    const xpar_occurrence * o = &ix->occ[i];
    if (o->stream_offset > off) continue;
    if (o->stream_offset + o->length < off + len) continue;
    if (!intact(o, user)) continue;
    /*  Prefer the canonical occurrence: it is the copy the stream and
        therefore the cell tables describe, so a repair sourced from it
        needs no second argument to explain why it was allowed.  */
    if (!got || o->entry < out->entry ||
        (o->entry == out->entry && o->file_offset < out->file_offset)) {
      *out = *o;  got = true;
    }
  }
  return got;
}
