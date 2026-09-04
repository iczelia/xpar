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

/*  Manifest construction, validation, and occurrence lookup.  */

#ifndef XPAR_MANIFEST_H
#define XPAR_MANIFEST_H

#include "blake3.h"
#include "xpar2.h"

/*  C99 forbids repeating a typedef, so every declarer shares a guard.  */
#ifndef XPAR_CHUNK_INDEX_TYPEDEF
#define XPAR_CHUNK_INDEX_TYPEDEF
typedef struct xpar_chunk_index xpar_chunk_index;
#endif

#define XPAR_PATH_WIN    (1u << 0)  /*  Also the Windows-target rules.  */
#define XPAR_PATH_NOCASE (1u << 1)  /*  Target folds case; duplicates.  */
/*  Allow only the final component to be a symbolic link.  */
#define XPAR_PATH_LEAF_LINK (1u << 2)

typedef enum {
  XPAR_PATH_OK = 0,
  XPAR_PATH_EMPTY,
  XPAR_PATH_EMPTY_COMPONENT,
  XPAR_PATH_ABSOLUTE,
  XPAR_PATH_DRIVE,
  XPAR_PATH_UNC,
  XPAR_PATH_DOT,
  XPAR_PATH_DOTDOT,
  XPAR_PATH_TRAILING_SLASH,
  XPAR_PATH_CONTROL,          /*  Any byte in 0x00..0x1F, NUL included.  */
  XPAR_PATH_WINCHAR,
  XPAR_PATH_DEVICE,           /*  CON, PRN, AUX, NUL, COM1..9, LPT1..9.  */
  XPAR_PATH_WINTRAIL,         /*  Component ends in '.' or ' '.  */
  XPAR_PATH_TOO_LONG,
  XPAR_PATH_SYMLINK           /*  A component of the path exists as one.  */
} xpar_path_status;

xpar_path_status xpar_path_check(const char * name, u32 len, u32 flags);
u32              xpar_host_path_flags(void);
const char *     xpar_path_reason(xpar_path_status s);

xpar_path_status xpar_symlink_target_check(const u8 * target, u32 len);
char * xpar_read_symlink(const char * path, u32 * length);

char * xpar_path_resolve(const char * dir, const char * name, u32 len,
                         u32 flags, xpar_path_status * why);

/*  Lexically normalize PATH as an absolute path.  */
char * xpar_path_lex_abs(const char * path);

bool xpar_utf8_valid(const u8 * p, u32 n);

/*  The manifest.  */

typedef struct xpar_manifest {
  xpar_entry * entry;
  char **      source;      /*  On-disk path per entry; walk side only.  */
  u32          count, cap;

  xpar_posix_rec * posix;
  u32              posix_count, posix_cap;

  u64 stream_base;
  u64 stream_length;        /*  L this generation appends.  */
  u64 entry_bytes;          /*  Sum of entry lengths, aliases included.  */
  u64 shared_bytes;         /*  Bytes deduplication kept out of L.  */
  u32 link_count;           /*  entry_type == 3 entries.  */
  u32 alias_extents;        /*  Extents naming already-defined bytes.  */
  u8  dedup_level;
  u8  align;                /*  XPAR_ALIGN_*  */
  u64 slice_size;           /*  Z, under XPAR_ALIGN_SLICE only.  */
} xpar_manifest;
/*  First entry missing under dir, excluding exempt.  */
const xpar_entry * xpar_manifest_unreachable(const xpar_manifest *,
                                            const char * dir,
                                            const char * exempt);

/*  Append an entry with absent-value sentinels. Invalidates earlier pointers.  */
xpar_entry * xpar_manifest_append(xpar_manifest * m);

void xpar_manifest_free(xpar_manifest * m);

int xpar_name_cmp(const char * a, u32 alen, const char * b, u32 blen);

/*  Reader manifests follow stream order, so hard-link lookup uses a shared
    name index.  */

typedef struct {
  u32 * order;    /*  Entry indices, sorted by name.  */
  u32   count;
} xpar_nameidx;

void xpar_nameidx_build(const xpar_manifest * m, xpar_nameidx * ix);
void xpar_nameidx_free (xpar_nameidx * ix);
i64 xpar_nameidx_find(const xpar_manifest * m, const xpar_nameidx * ix,
                      const char * name, u32 len);

void xpar_file_id(const xpar_entry * e, const u8 * key,
                  u8 out[XPAR_SET_ID_LEN]);

typedef struct { xpar_blake3_t h; } xpar_set_id_ctx;

void xpar_set_id_begin (xpar_set_id_ctx * c, const u8 * key,
                        const u8 * setd_body, sz n);
void xpar_set_id_update(xpar_set_id_ctx * c, const u8 * file_body, sz n);
void xpar_set_id_final (const xpar_set_id_ctx * c,
                        u8 out[XPAR_SET_ID_LEN]);

/*  Append a chunk, coalescing adjacent stream ranges.  */
void xpar_extents_append(xpar_extent ** list, u32 * count, u32 * capacity,
                         u64 stream_offset, u32 length);

u32 xpar_posix_intern(xpar_manifest * m, const xpar_posix_rec * r);
bool xpar_posix_equal(const xpar_posix_rec *, const xpar_posix_rec *);
void xpar_posix_rec_free(xpar_posix_rec *);

/*  Walking a tree into entries (writer side).  */

typedef struct {
  u8  dedup;              /*  XPAR_DEDUP_*.  */
  u8  align;              /*  XPAR_ALIGN_*  */
  u64 slice_size;
  u64 stream_base;
  u64 dedup_max_refs;     /*  0 is unlimited.  */
  u64 dedup_chunk;        /*  Target mean; zero selects 1 MiB.  */
  u64 dedup_memory;       /*  Maximum transient chunk index bytes.  */
  xpar_chunk_index * chunk_cache_out; /*  Optional ownership transfer.  */
  u8 ** stream_cache_out; /*  Optional canonical bytes from this pass.  */
  u64 * stream_cache_length_out;
  u64 stream_cache_limit;
  u32 preserve;           /*  XPAR_PRES_* mask, cli.h.  */
  u32 preserve_explicit;
  u32 caps_mask;          /*  xpar_fs_caps mask.  */
  u32 path_flags;         /*  XPAR_PATH_* for the names emitted.  */
  const char * base_dir;  /*  Names are relative to this when set.  */
  char * const * exclude; /*  Manifest-name globs, matched bytewise.  */
  u32 exclude_count;
  char * const * include; /*  Explicit includes override excludes.  */
  u32 include_count;
  const char * self_base; /*  Output base to skip.  */
  bool strict;            /*  Reject unstorable names.  */
  bool recurse;
  bool follow_symlinks;
  bool reproducible;
} xpar_walk_opts;

void xpar_walk_opts_default(xpar_walk_opts * o);
bool xpar_manifest_name_selected(const xpar_walk_opts *, const char *);

/*  Enumerate, sort, and record metadata and hard links.  */
void xpar_manifest_walk(xpar_manifest * m, char * const * roots,
                        u32 root_count, const xpar_walk_opts * o);

/*  Hash, deduplicate, and lay out regular entries.  */
void xpar_manifest_pack(xpar_manifest * m, const xpar_walk_opts * o,
                        xpar_progress_t * prog);

/*  Validation (reader side).  */

typedef struct { u64 base, length; } xpar_gen_range;

typedef struct {
  u64 stream_base;             /*  Of the generation that owns these.  */
  u64 stream_length;           /*  Its own L.  */
  u64 slice_size;              /*  Z; only read under ALIGN_SLICE.  */
  const xpar_gen_range * ancestor;  /*  Oldest first; may be NULL.  */
  u32 ancestor_count;
  u32 posix_record_count;
  u32 path_flags;              /*  XPAR_PATH_*  */
  u8  align;                   /*  XPAR_ALIGN_*  */
} xpar_mf_limits;

typedef enum {
  XPAR_MF_OK = 0,
  XPAR_MF_PATH,           /*  A path-shaped field violates rules.  */
  XPAR_MF_DUP_NAME,
  XPAR_MF_TYPE,           /*  Unknown entry_type.  */
  XPAR_MF_TYPE_LENGTH,    /*  length non-zero on a dir or symlink.  */
  XPAR_MF_TYPE_EXTENTS,   /*  extent_count non-zero on types 1, 2, 3.  */
  XPAR_MF_TYPE_EXTRA,     /*  extra present or absent wrongly.  */
  XPAR_MF_EXTENT_LEN,     /*  An extent of length zero.  */
  XPAR_MF_EXTENT_OVF,     /*  offset + length wraps u64.  */
  XPAR_MF_EXTENT_SUM,     /*  sum(length) != FILE.length.  */
  XPAR_MF_EXTENT_RANGE,   /*  Outside every generation's range.  */
  XPAR_MF_EXTENT_FWD,     /*  Rule 5: names undefined bytes.  */
  XPAR_MF_EXTENT_SPLIT,   /*  Rule 5: half definition, half reference.  */
  XPAR_MF_STREAM_GAP,     /*  The walk did not define all of L.  */
  XPAR_MF_LINK_MISSING,   /*  Alias target is not in this manifest.  */
  XPAR_MF_LINK_CHAIN,     /*  Alias names a non-regular entry.  */
  XPAR_MF_LINK_SELF,
  XPAR_MF_LINK_CONTENT,   /*  Length or content certificates disagree.  */
  XPAR_MF_POSIX_INDEX,
  XPAR_MF_LINK_META,      /*  mode, times or attrs differ from canonical.  */
  XPAR_MF_EXTENT_SHARE    /*  Shared extent is not contained in C_g.  */
} xpar_mf_status;

const char * xpar_mf_reason(xpar_mf_status s);

typedef struct {
  xpar_mf_status status;
  u32 entry;               /*  The entry at fault.  */
  u32 extent;              /*  Its extent at fault, when relevant.  */
  u32 other;               /*  The second entry of a duplicate pair.  */
  u32 link_meta_mismatch;
  u64 high_water;
} xpar_mf_result;

xpar_mf_status xpar_manifest_validate(const xpar_manifest * m,
                                      const xpar_mf_limits * lim,
                                      xpar_mf_result * out);

i64 xpar_link_target(const xpar_manifest * m, const xpar_nameidx * ix,
                     u32 entry);

typedef struct {
  u64 stream_offset;
  u64 length;
  u64 file_offset;   /*  Offset of these bytes within the entry.  */
  u32 entry;         /*  Manifest index.  */
  u32 extent;        /*  Index within that entry's extent list.  */
} xpar_occurrence;

typedef struct {
  xpar_occurrence * occ;
  u64 * max_end;     /*  Prefix maximum of occ[i].stream_offset+length.  */
  u32   count;
} xpar_occindex;

void xpar_occindex_build(const xpar_manifest * m, xpar_occindex * ix);
void xpar_occindex_free (xpar_occindex * ix);

/*  Visit overlaps in stream order; return the hit count.  */
typedef void (* xpar_occ_fn)(const xpar_occurrence * o, void * user);
u32 xpar_occindex_overlaps(const xpar_occindex * ix, u64 off, u64 len,
                           xpar_occ_fn fn, void * user);
bool xpar_occindex_canonical(const xpar_occindex * ix, u64 off,
                             xpar_occurrence * out, u64 * run);

/*  Return `off` if covered, otherwise the next extent start or `limit`.  */
u64 xpar_occindex_next(const xpar_occindex * ix, u64 off, u64 limit);
bool xpar_occindex_repair_source(const xpar_occindex * ix, u64 off,
                                 u64 len,
                                 bool (* intact)(const xpar_occurrence *,
                                                 void *),
                                 void * user, xpar_occurrence * out);

#endif
