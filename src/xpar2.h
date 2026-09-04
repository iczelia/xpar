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

/*  XPAR 2.0 constants and decoded structures.  */

#ifndef XPAR_XPAR2_H
#define XPAR_XPAR2_H

#include "common.h"

#define XPAR_PKT_MAGIC   "XPAR2PKT"
#define XPAR_PKT_HDR     48       /*  Bytes before the body.  */
#define XPAR_SET_ID_LEN  16
#define XPAR_PKT_ALIGN   8        /*  Every packet length is a multiple.  */

/*  Extension for packet-bearing volumes.  */
#define XPAR_EXT         ".xpa"
#define XPAR_EXT_LEN     4

#define XPAR_PF_CRITICAL       (1u << 0)
#define XPAR_PF_KEYED          (1u << 1)
#define XPAR_PF_BODY_UNCHECKED (1u << 2)  /*  Legal only on STRM.  */
/*  Bits 3..31 are reserved and shall be zero; a reader rejects them.  */
#define XPAR_PF_KNOWN          (XPAR_PF_CRITICAL | XPAR_PF_KEYED |            \
                                XPAR_PF_BODY_UNCHECKED)

/*  Four-byte packet tags.  */
#define XPAR_T_VOLH "VOLH"
#define XPAR_T_SETD "SETD"
#define XPAR_T_FILE "FILE"
#define XPAR_T_POSX "POSX"
#define XPAR_T_SLCR "SLCR"
#define XPAR_T_SLTG "SLTG"
#define XPAR_T_SLCL "SLCL"
#define XPAR_T_RCVS "RCVS"
#define XPAR_T_LAYT "LAYT"
#define XPAR_T_AUTH "AUTH"
#define XPAR_T_ARMG "ARMG"
#define XPAR_T_STRM "STRM"
#define XPAR_T_CRTR "CRTR"
#define XPAR_T_CMNT "CMNT"

enum {
  XPAR_CODEC_FFT = 0,
  XPAR_CODEC_MATRIX = 1,
  XPAR_CODEC_FFT_LOW = 2
};
#define XPAR_CODEC_IS_FFT(c) \
  ((c) == XPAR_CODEC_FFT || (c) == XPAR_CODEC_FFT_LOW)
enum {
  XPAR_LAYOUT_SIDECAR = 0, XPAR_LAYOUT_SPLIT = 1, XPAR_LAYOUT_ARMOURED = 2
};

/*  Output name; unknown layouts default to sidecar.  */
static inline const char * xpar_layout_name(u8 layout) {
  switch (layout) {
    case XPAR_LAYOUT_SPLIT:    return "split";
    case XPAR_LAYOUT_ARMOURED: return "armoured";
    default:                   return "sidecar";
  }
}

static inline const char * xpar_codec_name(u8 codec) {
  switch (codec) {
    case XPAR_CODEC_FFT:     return "fft";
    case XPAR_CODEC_FFT_LOW: return "fft-low";
    default:                 return "matrix";
  }
}
enum {
  XPAR_ALIGN_PACKED = 0,
  XPAR_ALIGN_SLICE = 1,
  XPAR_ALIGN_1K = 2
};
enum { XPAR_DEDUP_NONE = 0, XPAR_DEDUP_FILE = 1, XPAR_DEDUP_CHUNK = 2 };

#define XPAR_FEAT_B3_SUBTREE (1u << 0)
#define XPAR_REQUIRED_KNOWN  XPAR_FEAT_B3_SUBTREE
#define XPAR_OPTIONAL_KNOWN  0u

#define XPAR_CELL_MIN      4096
/*  Format limit for cells per slice.  */
#define XPAR_CELLS_MAX     65536
#define XPAR_SLICE_MIN     64
#define XPAR_SLICE_MAX     ((u64) 1 << 40)
#define XPAR_NAME_MAX      65535
#define XPAR_EXTRA_MAX     65535
#define XPAR_ABSENT_U32    0xFFFFFFFFu   /*  mode, posix_index.  */
#define XPAR_ABSENT_TIME   INT64_MIN     /*  Any of the four timestamps.  */

typedef struct {
  u64 slice_size;          /*  Z  */
  u64 data_slice_count;    /*  S  */
  u64 stream_length;       /*  L  */
  u32 file_count;          /*  F; zero only in a child generation.  */
  u8  field_log2;          /*  8 or 16  */
  u8  codec;               /*  XPAR_CODEC_*  */
  u8  recovery_axis_log2;
  u8  layout;              /*  XPAR_LAYOUT_*  */
  u8  align;               /*  XPAR_ALIGN_*  */
  u8  slice_tag_len;       /*  0, 8 or 16  */
  u8  dedup_level;         /*  XPAR_DEDUP_*  */
  u32 required_features;
  u32 optional_features;
  u32 cell_bytes;          /*  Y; 0 means no SLCL and slice-granular erasure  */
  u8  parent_set_id[XPAR_SET_ID_LEN];
  u32 generation;
  u32 posix_record_count;
  u64 stream_base;         /*  This generation's origin in the chain space.  */
  u8  (* file_id)[XPAR_SET_ID_LEN];   /*  file_count entries, manifest order  */
} xpar_setd;

/*  High-rate FFT records the power-of-two recovery bracket. Low-rate FFT
    records the rounded data bracket instead, leaving the rest of the field
    for recovery exponents. Matrix exponents span the field.  */
static inline u64 xpar_setd_recovery_limit(const xpar_setd * s) {
  u64 axis = (u64) 1 << s->recovery_axis_log2;
  if (s->codec == XPAR_CODEC_FFT_LOW)
    return ((u64) 1 << s->field_log2) - axis;
  if (s->codec == XPAR_CODEC_MATRIX)
    return ((u64) 1 << s->field_log2) - s->data_slice_count;
  return axis;
}

enum {
  XPAR_ENTRY_REGULAR  = 0,
  XPAR_ENTRY_DIR      = 1,
  XPAR_ENTRY_SYMLINK  = 2,
  XPAR_ENTRY_HARDLINK = 3
};

/*  attrs bits. Bits 7, 8 and 9 are advisory and are never applied on
    extract; bit 10 exists so setid can be reported before extraction starts
    rather than discovered during it.  */
#define XPAR_ATTR_READONLY   (1u << 0)
#define XPAR_ATTR_HIDDEN     (1u << 1)
#define XPAR_ATTR_SYSTEM     (1u << 2)
#define XPAR_ATTR_EXEC       (1u << 3)
#define XPAR_ATTR_RAWNAME    (1u << 4)   /*  name is not valid UTF-8.  */
#define XPAR_ATTR_ARCHIVE    (1u << 5)
#define XPAR_ATTR_NOINDEX    (1u << 6)
#define XPAR_ATTR_SPARSE     (1u << 7)   /*  Advisory.  */
#define XPAR_ATTR_FSCOMPRESS (1u << 8)   /*  Advisory.  */
#define XPAR_ATTR_FSENCRYPT  (1u << 9)   /*  Advisory.  */
#define XPAR_ATTR_SETID      (1u << 10)

/* Bits 11-15 are reserved. */
#define XPAR_ATTR_KNOWN      0x07FFu

typedef struct {
  u64 stream_offset;
  u64 length;
} xpar_extent;

typedef struct {
  u8  file_id[XPAR_SET_ID_LEN];
  u64 length;                 /*  Content length; a hard-link alias carries
                                  its target's, not zero.  */
  u8  content_hash[32];       /*  BLAKE3-256.  */
  u8  prefix_hash[16];        /*  BLAKE3-128 of min(16384, length) bytes.  */
  i64 mtime_ns, atime_ns, ctime_ns, btime_ns;
  u32 mode;                   /*  XPAR_ABSENT_U32 when not recorded.  */
  u32 posix_index;            /*  XPAR_ABSENT_U32 when not recorded.  */
  u16 entry_type;             /*  XPAR_ENTRY_*  */
  u16 attrs;
  u32 extent_count;           /*  0 for every type but REGULAR.  */
  xpar_extent * extents;
  char * name;                /*  Relative, '/' separated, NUL terminated.  */
  u32 name_len;
  u8 * extra;                 /*  Symlink target, or hard-link target path.  */
  u32 extra_len;
} xpar_entry;

static inline void xpar_entry_free(xpar_entry * e) {
  xpar_free(e->extents);  xpar_free(e->name);  xpar_free(e->extra);
  xpar_memset(e, 0, sizeof *e);
}

typedef struct {
  char * name;
  u8 *   value;
  u32    value_len;
} xpar_xattr;

typedef struct {
  u32 uid, gid;
  char * owner;               /*  May be NULL when the host had no name.  */
  char * group;
  xpar_xattr * xattrs;        /*  Sorted by name, strictly ascending.  */
  u32 xattr_count;
} xpar_posix_rec;

typedef struct {
  u32 * slice_crc;            /*  SLCR: one per slice, always present.  */
  u8  * slice_tag;            /*  SLTG: slice_tag_len bytes each, or NULL.  */
  u32 * cell_crc;             /*  SLCL: cells_per_slice per slice, or NULL.  */
  u64   slice_count;
  u32   cells_per_slice;      /*  ceil(Z / Y); 0 when cell_bytes == 0.  */
  u8    tag_len;
} xpar_tags;

enum { XPAR_VOL_INDEX = 0, XPAR_VOL_DATA = 1, XPAR_VOL_RECOVERY = 2 };

typedef struct {
  u8  kind;                   /*  XPAR_VOL_*  */
  u8  vflags;                 /*  Bit 0: armoured.  */
  u32 recovery_first;         /*  RECOVERY: first exponent.  */
  u64 stream_offset;          /*  DATA: offset into the set stream.  */
  u64 byte_length;            /*  DATA: bytes here. RECOVERY: slice count.  */
  u64 vol_tag;                /*  DATA only: BLAKE3-64 of the volume.  */
  char * name;                /*  Basename; never a path.  */
} xpar_vol;

typedef struct {
  u8 * bad;                   /*  slice_count * cells_per_slice, 1 = erased.  */
  u64  slice_count;
  u32  cells_per_slice;
  u64  bad_count;
} xpar_erasures;

static inline bool xpar_cell_bad(const xpar_erasures * e, u64 slice, u32 col) {
  return e->bad[slice * e->cells_per_slice + col] != 0;
}

static inline void xpar_cell_mark(xpar_erasures * e, u64 slice, u32 col) {
  u8 * p = &e->bad[slice * e->cells_per_slice + col];
  if (!*p) { *p = 1;  e->bad_count++; }
}

#endif
