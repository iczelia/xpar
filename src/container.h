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

#ifndef XPAR_CONTAINER_H
#define XPAR_CONTAINER_H

#include "blake3.h"
#include "xpar2.h"

typedef enum {
  XPAR_OK = 0,
  XPAR_E_SHORT,        /*  Fewer bytes present than the structure needs.  */
  XPAR_E_MAGIC,        /*  No "XPAR2PKT" at the candidate offset.  */
  XPAR_E_LENGTH,
  XPAR_E_CHECKSUM,
  XPAR_E_MALFORMED,
  XPAR_E_UNSUPPORTED,  /*  required_features, or format_version_major.  */
  XPAR_E_NEEDKEY
} xpar_status;

const char * xpar_status_str(xpar_status);

#define XPAR_PKT_LEN_MAX  ((u64) 1 << 48)

typedef struct {
  u8 k_pkt[XPAR_BLAKE3_KEY_LEN];
  u8 k_slice[XPAR_BLAKE3_KEY_LEN];
  u8 k_set[XPAR_BLAKE3_KEY_LEN];
  u8 k_file[XPAR_BLAKE3_KEY_LEN];
} xpar_key;

bool xpar_key_master(u8 * out, const void * key_file, sz n);
void xpar_key_derive(xpar_key * out, const u8 * master);
void xpar_key_check(u8 * out16, const u8 * master);

typedef struct {
  u64  length;                  /*  Header plus body, a multiple of 8.  */
  u32  flags;                   /*  XPAR_PF_*  */
  u8   set_id[XPAR_SET_ID_LEN];
  char type[4];                 /*  Four ASCII bytes; not NUL terminated.  */
  u8   checksum[8];
} xpar_pkt;

static inline bool xpar_pkt_is(const xpar_pkt * p, const char * t) {
  return p->type[0] == t[0] && p->type[1] == t[1] &&
         p->type[2] == t[2] && p->type[3] == t[3];
}

xpar_status xpar_pkt_read(const u8 * p, u64 avail, const xpar_key * key,
                          xpar_pkt * out);

typedef struct {
  const u8 * buf;
  u64 size;
  u64 pos;
  u64 emitted;
  u64 skip_length;      /*  Candidates rejected by a length constraint.  */
  u64 skip_checksum;
  u64 skip_keyed;       /*  Keyed packets no key was supplied for.  */
  u64 skip_unsupported; /*  Unknown critical packet types.  */
  u32 step;
  bool accept_unverified_keyed;
  const xpar_key * key;
} xpar_scan;

void xpar_scan_init(xpar_scan *, const u8 * buf, u64 size,
                    const xpar_key * key, bool resync);

bool xpar_scan_next(xpar_scan *, xpar_pkt * hdr, const u8 ** body, u64 * off);

typedef struct { u8 * data;  sz len;  sz cap; } xpar_buf;

void xpar_buf_init(xpar_buf *);
void xpar_buf_free(xpar_buf *);
u8 * xpar_buf_grow(xpar_buf *, sz n);   /*  Zeroed; the region is returned.  */
void xpar_buf_put (xpar_buf *, const void * data, sz n);

typedef struct { const void * p;  sz n; } xpar_part;

void xpar_pkt_writev(xpar_buf * out, const char * type, u32 flags,
                     const u8 * set_id, const xpar_part * parts, u32 nparts,
                     const xpar_key * key);
void xpar_pkt_write (xpar_buf * out, const char * type, u32 flags,
                     const u8 * set_id, const void * body, sz body_len,
                     const xpar_key * key);

typedef struct {
  bool reproducible;
  bool keep_mtime, keep_atime, keep_ctime, keep_btime;
  bool keep_posix;      /*  --preserve=owner or --preserve=xattr, explicit.  */
} xpar_wropt;

#define XPAR_FORMAT_MAJOR 2
#define XPAR_FORMAT_MINOR 0
#define XPAR_VOL_STANDALONE 0xFFFFFFFFu   /*  VOLH.volume_index, LAYT.this.  */

typedef struct {
  u32 volume_index;     /*  XPAR_VOL_STANDALONE for a lone index volume.  */
  u32 volume_kind;      /*  XPAR_VOL_*  */
  u32 version_major, version_minor;
} xpar_volh;

xpar_status xpar_volh_read (const u8 * body, sz n, xpar_volh * out);
void        xpar_volh_write(xpar_buf *, const xpar_volh *, const u8 * set_id,
                            const xpar_key *);

/*  XPAR_E_UNSUPPORTED populates out; xpar_setd_free accepts all results.  */
xpar_status xpar_setd_read (const u8 * body, sz n, xpar_setd * out);
void        xpar_setd_write(xpar_buf *, const xpar_setd *, const u8 * set_id,
                            const xpar_key *);
void        xpar_setd_free (xpar_setd *);

xpar_status xpar_setd_check_parent(const xpar_setd * child,
                                   const u8 * parent_set_id,
                                   const xpar_setd * parent);

xpar_status xpar_entry_read (const u8 * body, sz n, u32 posix_record_count,
                             xpar_entry * out);
void        xpar_entry_write(xpar_buf *, const xpar_entry *,
                             const u8 * set_id, const xpar_key *,
                             const xpar_wropt *);

typedef struct {
  u32 first_record;
  u32 count;
  xpar_posix_rec * rec;
} xpar_posx;

xpar_status xpar_posx_read (const u8 * body, sz n, xpar_posx * out);
void        xpar_posx_write(xpar_buf *, u32 first_record, u32 count,
                            const xpar_posix_rec *, const u8 * set_id,
                            const xpar_key *);
void        xpar_posx_free (xpar_posx *);

#define XPAR_POSX_SPLIT ((sz) 1 << 20)
void xpar_posx_write_all(xpar_buf *, const xpar_posix_rec *, u32 count,
                         const u8 * set_id, const xpar_key *);

#define XPAR_TABLE_SPLIT 65536

typedef struct { u64 first_slice, count;  u32 * crc; } xpar_slcr;
typedef struct { u64 first_slice, count;  u8 tag_len;  u8 * tag; } xpar_sltg;
typedef struct {
  u64 first_slice, count;
  u32 cell_bytes;             /*  Y; must equal SETD.cell_bytes.  */
  u32 cells_per_slice;        /*  K = ceil(Z/Y), derived, not stored.  */
  u32 * crc;                  /*  count * cells_per_slice, row major.  */
} xpar_slcl;

xpar_status xpar_slcr_read (const u8 * body, sz n, xpar_slcr * out);
xpar_status xpar_sltg_read (const u8 * body, sz n, xpar_sltg * out);
xpar_status xpar_slcl_read (const u8 * body, sz n, u64 slice_size,
                            xpar_slcl * out);
void        xpar_slcr_free (xpar_slcr *);
void        xpar_sltg_free (xpar_sltg *);
void        xpar_slcl_free (xpar_slcl *);

/* Writers accept const table data without reader-struct casts. */
void xpar_slcr_write(xpar_buf *, u64 first_slice, u64 count,
                     const u32 * crc, const u8 * set_id,
                     const xpar_key *);
void xpar_sltg_write(xpar_buf *, u64 first_slice, u64 count, u8 tag_len,
                     const u8 * tag, const u8 * set_id,
                     const xpar_key *);
void xpar_slcl_write(xpar_buf *, u64 first_slice, u64 count,
                     u32 cell_bytes, u32 cells_per_slice, const u32 * crc,
                     const u8 * set_id,
                     const xpar_key *);

void xpar_slcr_write_all(xpar_buf *, const u32 * crc, u64 slices,
                         const u8 * set_id, const xpar_key *);
void xpar_sltg_write_all(xpar_buf *, const u8 * tag, u64 slices, u8 tag_len,
                         const u8 * set_id, const xpar_key *);
void xpar_slcl_write_all(xpar_buf *, const u32 * crc, u64 slices,
                         u32 cell_bytes, u32 cells_per_slice,
                         const u8 * set_id, const xpar_key *);

#define XPAR_TAGS_CRC   1u
#define XPAR_TAGS_TAG   2u
#define XPAR_TAGS_CELL  4u

typedef struct {
  xpar_tags t;
  u8 * seen_crc;
  u8 * seen_tag;
  u8 * seen_cell;
} xpar_tagset;

/*  `have_crc` is false for auth-only sets, which store no SLCR table.  */
bool        xpar_tagset_init(xpar_tagset *, u64 slice_count, u8 tag_len,
                             u32 cells_per_slice, bool have_crc,
                             u64 input_bytes);
void        xpar_tagset_free(xpar_tagset *);
xpar_status xpar_tagset_slcr(xpar_tagset *, const xpar_slcr *);
xpar_status xpar_tagset_sltg(xpar_tagset *, const xpar_sltg *);
xpar_status xpar_tagset_slcl(xpar_tagset *, const xpar_slcl *);
u32         xpar_tagset_complete(const xpar_tagset *);

typedef struct {
  u64 exponent;
  const u8 * data;
  u64 length;
} xpar_rcvs;

/*  `slice_size` is SETD.Z, or 0 to take Z from the body length.  */
xpar_status xpar_rcvs_read (const u8 * body, sz n, u64 slice_size,
                            xpar_rcvs * out);
void        xpar_rcvs_write(xpar_buf *, u64 exponent, const void * data,
                            sz len, const u8 * set_id, const xpar_key *);
u32         xpar_rcvs_stream_header(u8 out[XPAR_PKT_HDR + 16], u64 exponent,
                                    const void * data, sz len,
                                    const u8 * set_id, const xpar_key *);

typedef struct {
  u32 this_volume;            /*  XPAR_VOL_STANDALONE if a lone index.  */
  u32 count;
  xpar_vol * vol;
} xpar_layt;

xpar_status xpar_layt_read (const u8 * body, sz n, xpar_layt * out);
void        xpar_layt_write(xpar_buf *, const xpar_layt *, const u8 * set_id,
                            const xpar_key *);
void        xpar_layt_free (xpar_layt *);

xpar_status xpar_layt_tiles(const xpar_layt *, u64 stream_length);

/*  A DATA volume's BLAKE3-64 tag is its identity; its name is only a hint.  */
void xpar_vol_tag_begin(xpar_blake3_t *);
u64  xpar_vol_tag_final(xpar_blake3_t *);

/*  Whether the file at `path` is that volume: the right size, and, where
    the layout records a tag, the right bytes. A missing file is not.  */
bool xpar_vol_tag_match(const char * path, const xpar_vol *);

typedef struct {
  u32 kdf_id;
  u8  slice_tag_keyed, packet_tag_keyed, unkeyed_retained;
  u8  key_check[16];
} xpar_auth;

xpar_status xpar_auth_read (const u8 * body, sz n, xpar_auth * out);
void        xpar_auth_write(xpar_buf *, const xpar_auth *, const u8 * set_id,
                            const xpar_key *);

bool xpar_auth_key_ok(const xpar_auth *, const u8 * master);

/*  Body text with its zero padding removed. `*out` is NUL terminated for
    convenience and `*out_len` carries the true length.  */
void xpar_text_write(xpar_buf *, const char * type, const char * text,
                     const u8 * set_id, const xpar_key *);

xpar_status xpar_text_read(const u8 * body, sz n, sz * out_len);

void xpar_crtr_write(xpar_buf *, const char * creator, const u8 * set_id,
                     const xpar_key *, const xpar_wropt *);

#define XPAR_ARMG_DEPTH_MAX ((u64) 1 << 24)

typedef struct {
  u8  symbol_bits;            /*  8 or 16.  */
  u32 poly, n, k, fcr, prim;
  u64 depth;
  u64 plain_length;
  u64 armoured_length;
  const u8 * data;            /*  Points into the packet body.  */
} xpar_armg;

/*  ceil(plain / (D*k*W)) * D*n*W with W = symbol_bits/8. Zero when the
    parameters are out of range or the product would overflow, which is
    what the reader tests the stored field against.  */
u64 xpar_armg_length(u8 symbol_bits, u32 n, u32 k, u64 depth, u64 plain);

xpar_status xpar_armg_read (const u8 * body, sz n, xpar_armg * out);
void        xpar_armg_write(xpar_buf *, const xpar_armg *,
                            const void * armoured, const u8 * set_id,
                            const xpar_key *);

typedef struct {
  u64 stream_offset;
  const u8 * data;
  u64 length;
} xpar_strm;

xpar_status xpar_strm_read (const u8 * body, sz n, xpar_strm * out);
void        xpar_strm_write(xpar_buf *, u64 stream_offset, const void * data,
                            sz len, const u8 * set_id, const xpar_key *);

void xpar_strm_write_header(xpar_buf *, u64 stream_length,
                            const u8 * set_id, const xpar_key *);

typedef struct {
  const xpar_setd * setd;
  const xpar_entry * file;    /*  Owned entries only, manifest order.  */
  u32 file_count;
  const xpar_posix_rec * posix;
  u32 posix_count;
  const u32 * slice_crc;      /*  NULL in a stream-empty generation.  */
  u64 slice_count;
  const xpar_auth * auth;     /*  NULL unless --auth-key.  */
  const xpar_layt * layt;
} xpar_crit;

void xpar_crit_write(xpar_buf * out, const xpar_crit *, const u8 * set_id,
                     const xpar_key *, const xpar_wropt *);

bool xpar_replicate_here(u64 crit, u64 payload, u32 i, u32 count);

typedef struct {
  xpar_pkt hdr;
  const u8 * body;
  u64 body_len;
  u32 copies;                 /*  Verifying copies seen, this one included.  */
  u32 conflicts;              /*  Verifying copies that disagreed.  */
} xpar_crit_pkt;

typedef struct {
  xpar_crit_pkt * pkt;
  u32 count, cap;
  u32 * idx;                  /*  Open addressed; 0 empty, slot+1 otherwise.  */
  u32 mask;
  u32 copies, conflicts;
} xpar_critset;

void xpar_critset_init(xpar_critset *);
void xpar_critset_free(xpar_critset *);

/*  True when the packet was new. A repeat bumps `copies`, and `conflicts`
    when its bytes differ from the copy already held.

    The set keeps `body` rather than copying it, and reads it again on
    every later add and find. It shall point into storage that outlives
    the set and does not move: a volume image, or an ARMG plaintext the
    caller owns. A growable buffer that is written to again after the add
    is not such storage.  */
bool xpar_critset_add(xpar_critset *, const xpar_pkt *, const u8 * body);

const xpar_crit_pkt * xpar_critset_find(const xpar_critset *,
                                        const u8 * set_id, const char * type,
                                        u64 disc);
const xpar_crit_pkt * xpar_critset_find_file(const xpar_critset *,
                                             const u8 * set_id,
                                             const u8 * file_id);

xpar_status xpar_posx_collect(const xpar_critset *, const u8 * set_id,
                              u32 count, xpar_posix_rec ** out);
void xpar_posix_records_free(xpar_posix_rec *, u32 count);

#endif
