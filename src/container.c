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

#include "container.h"
#include "manifest.h"

/*  Checked arithmetic for untrusted lengths and offsets.  */

static bool add64(u64 a, u64 b, u64 * out) {
  if (a > (u64) -1 - b) return false;
  *out = a + b;  return true;
}

static bool mul64(u64 a, u64 b, u64 * out) {
  if (a && b > (u64) -1 / a) return false;
  *out = a * b;  return true;
}

/*  Bytewise ordering is unsigned; plain char may be signed.  */
static int bytes_cmp(const u8 * a, sz na, const u8 * b, sz nb) {
  sz n = na < nb ? na : nb;
  int r = n ? xpar_memcmp(a, b, n) : 0;
  if (r) return r;
  return na < nb ? -1 : (na > nb);
}

/*  Callers have already rejected embedded NUL bytes.  */
static char * dup_str(const u8 * p, sz n) {
  char * s = (char *) xpar_malloc(n + 1);
  if (n) xpar_memcpy(s, p, n);
  s[n] = 0;
  return s;
}

const char * xpar_status_str(xpar_status s) {
  switch (s) {
    case XPAR_OK:            return "ok";
    case XPAR_E_SHORT:       return "truncated";
    case XPAR_E_MAGIC:       return "not a packet";
    case XPAR_E_LENGTH:      return "bad packet length";
    case XPAR_E_CHECKSUM:    return "checksum mismatch";
    case XPAR_E_MALFORMED:   return "malformed packet";
    case XPAR_E_UNSUPPORTED: return "unsupported format feature";
    case XPAR_E_NEEDKEY:     return "keyed packet, no key";
  }
  return "unknown";
}

/*  The context strings are ASCII, carry no NUL, and are part of the
    normative format: changing one changes every tag the key produces.  */

bool xpar_key_master(u8 * out, const void * key_file, sz n) {
  if (!n) return false;
  xpar_blake3_derive_key("xpar2 auth master v1", key_file, n, out);
  return true;
}

void xpar_key_derive(xpar_key * out, const u8 * master) {
  xpar_blake3_derive_key("xpar2 packet v1", master, XPAR_BLAKE3_KEY_LEN,
                         out->k_pkt);
  xpar_blake3_derive_key("xpar2 slice v1", master, XPAR_BLAKE3_KEY_LEN,
                         out->k_slice);
  xpar_blake3_derive_key("xpar2 set v1", master, XPAR_BLAKE3_KEY_LEN,
                         out->k_set);
  xpar_blake3_derive_key("xpar2 file v1", master, XPAR_BLAKE3_KEY_LEN,
                         out->k_file);
}

void xpar_key_check(u8 * out16, const u8 * master) {
  xpar_blake3_hash_keyed(master, "xpar2 key check v1", 18, out16, 16);
}

/*  The tag covers packet bytes [0, 40) and [48, length), i.e. the whole
    packet with the eight checksum bytes elided.  */
static void pkt_tag(const u8 * p, u64 len, bool body, const xpar_key * key,
                    u8 * out8) {
  xpar_blake3_t h;
  if (key) xpar_blake3_init_keyed(&h, key->k_pkt);
  else     xpar_blake3_init(&h);
  xpar_blake3_update(&h, p, 40);
  if (body && len > XPAR_PKT_HDR)
    xpar_blake3_update(&h, p + XPAR_PKT_HDR, (sz) (len - XPAR_PKT_HDR));
  xpar_blake3_final(&h, out8, 8);
}

/*  Require at most seven zero padding bytes.  */
static bool pad_ok(const u8 * body, u64 used, u64 n) {
  if (used > n || n - used >= XPAR_PKT_ALIGN) return false;
  for (; used < n; used++) if (body[used]) return false;
  return true;
}

xpar_status xpar_pkt_read(const u8 * p, u64 avail, const xpar_key * key,
                          xpar_pkt * out) {
  u64 len;  u8 want[8];  bool body = true;
  xpar_memset(out, 0, sizeof *out);
  if (avail < XPAR_PKT_HDR)                    return XPAR_E_SHORT;
  if (xpar_memcmp(p, XPAR_PKT_MAGIC, 8) != 0)  return XPAR_E_MAGIC;

  len = xpar_rd64(p + 8);
  if (len < XPAR_PKT_HDR)                      return XPAR_E_LENGTH;
  if (len % XPAR_PKT_ALIGN)                    return XPAR_E_LENGTH;
  if (len > XPAR_PKT_LEN_MAX)                  return XPAR_E_LENGTH;
  if (len > avail)                             return XPAR_E_LENGTH;
  /*  sz may be 32-bit. A packet longer than the host can address cannot
      have been mapped, and narrowing the length to sz would hash the wrong
      range and accept a packet whose declared size was never present.  */
  if (len > (u64) (sz) -1)                     return XPAR_E_LENGTH;

  out->length = len;
  xpar_memcpy(out->set_id, p + 16, XPAR_SET_ID_LEN);
  xpar_memcpy(out->type, p + 32, 4);
  out->flags = xpar_rd32(p + 36);
  xpar_memcpy(out->checksum, p + 40, 8);

  /*  Reject reserved flag bits.  */
  if (out->flags & ~(u32) XPAR_PF_KNOWN)       return XPAR_E_MALFORMED;
  if (out->flags & XPAR_PF_BODY_UNCHECKED) {
    if (!xpar_pkt_is(out, XPAR_T_STRM))        return XPAR_E_MALFORMED;
    body = false;
  }
  if ((out->flags & XPAR_PF_KEYED) && !key)    return XPAR_E_NEEDKEY;

  pkt_tag(p, len, body, (out->flags & XPAR_PF_KEYED) ? key : NULL, want);
  if (!xpar_ct_equal(want, out->checksum, 8))  return XPAR_E_CHECKSUM;
  return XPAR_OK;
}

void xpar_scan_init(xpar_scan * s, const u8 * buf, u64 size,
                    const xpar_key * key, bool resync) {
  xpar_memset(s, 0, sizeof *s);
  s->buf = buf;  s->size = size;  s->key = key;
  s->step = resync ? 1 : XPAR_PKT_ALIGN;
}

bool xpar_scan_next(xpar_scan * s, xpar_pkt * hdr, const u8 ** body,
                    u64 * off) {
  if (s->size < XPAR_PKT_HDR) return false;
  while (s->pos <= s->size - XPAR_PKT_HDR) {
    xpar_status st;
    if (xpar_memcmp(s->buf + s->pos, XPAR_PKT_MAGIC, 8) != 0) {
      s->pos += s->step;  continue;
    }
    st = xpar_pkt_read(s->buf + s->pos, s->size - s->pos, s->key, hdr);
    if (st == XPAR_E_NEEDKEY && s->accept_unverified_keyed) st = XPAR_OK;
    if (st != XPAR_OK) {
      if      (st == XPAR_E_NEEDKEY)  s->skip_keyed++;
      else if (st == XPAR_E_CHECKSUM) s->skip_checksum++;
      else                            s->skip_length++;
      /*  "XPAR2PKT" has no proper border, so no second magic can begin
          inside the eight bytes just matched. Advancing 8 therefore loses
          no candidate even while resyncing at STEP 1.  */
      s->pos += XPAR_PKT_ALIGN;
      continue;
    }
    *off = s->pos;
    *body = s->buf + s->pos + XPAR_PKT_HDR;
    s->pos += hdr->length;
    s->emitted++;
    if (s->emitted > s->size / XPAR_PKT_HDR) {
      s->implausible = true;  return false;
    }
    return true;
  }
  return false;
}

/*  Writing.  */

void xpar_buf_init(xpar_buf * b) { b->data = NULL;  b->len = 0;  b->cap = 0; }

void xpar_buf_free(xpar_buf * b) { xpar_free(b->data);  xpar_buf_init(b); }

u8 * xpar_buf_grow(xpar_buf * b, sz n) {
  sz need;
  xpar_assert(n <= (sz) -1 - b->len);
  need = b->len + n;
  if (need > b->cap) {
    sz cap = b->cap ? b->cap : 256;
    while (cap < need) {
      xpar_assert(cap <= ((sz) -1) / 2);
      cap *= 2;
    }
    b->data = (u8 *) xpar_realloc(b->data, cap);
    b->cap  = cap;
  }
  xpar_memset(b->data + b->len, 0, n);
  b->len = need;
  return b->data + need - n;
}

void xpar_buf_put(xpar_buf * b, const void * data, sz n) {
  if (n) xpar_memcpy(xpar_buf_grow(b, n), data, n);
}

void xpar_pkt_writev(xpar_buf * out, const char * type, u32 flags,
                     const u8 * set_id, const xpar_part * parts, u32 nparts,
                     const xpar_key * key) {
  u64 body = 0, len;  sz at;  u8 * p;  u32 i;
  for (i = 0; i < nparts; i++) {
    xpar_assert(add64(body, parts[i].n, &body));
  }
  /*  Bound the body before the header and the pad are added, so that the
      alignment cannot wrap a huge length into a small one.  */
  xpar_assert(body <= XPAR_PKT_LEN_MAX - XPAR_PKT_HDR);
  len = xpar_align_up(body + XPAR_PKT_HDR, XPAR_PKT_ALIGN);
  xpar_assert(len <= XPAR_PKT_LEN_MAX && len <= (u64) (sz) -1);

  p = xpar_buf_grow(out, (sz) len);
  xpar_memcpy(p, XPAR_PKT_MAGIC, 8);
  xpar_wr64(p + 8, len);
  xpar_memcpy(p + 16, set_id, XPAR_SET_ID_LEN);
  xpar_memcpy(p + 32, type, 4);
  xpar_wr32(p + 36, flags);
  at = XPAR_PKT_HDR;
  for (i = 0; i < nparts; i++) {
    if (parts[i].n) xpar_memcpy(p + at, parts[i].p, parts[i].n);
    at += parts[i].n;
  }
  pkt_tag(p, len, !(flags & XPAR_PF_BODY_UNCHECKED),
          (flags & XPAR_PF_KEYED) ? key : NULL, p + 40);
}

void xpar_pkt_write(xpar_buf * out, const char * type, u32 flags,
                    const u8 * set_id, const void * body, sz body_len,
                    const xpar_key * key) {
  xpar_part part;
  part.p = body;  part.n = body_len;
  xpar_pkt_writev(out, type, flags, set_id, &part, 1, key);
}

static u32 pkt_flags(u32 base, const xpar_key * key) {
  return key ? (base | XPAR_PF_KEYED) : base;
}

xpar_status xpar_volh_read(const u8 * body, sz n, xpar_volh * out) {
  xpar_memset(out, 0, sizeof *out);
  if (n != 24) return n < 24 ? XPAR_E_SHORT : XPAR_E_MALFORMED;
  out->volume_index  = xpar_rd32(body);
  out->volume_kind   = xpar_rd32(body + 4);
  out->version_major = xpar_rd32(body + 8);
  out->version_minor = xpar_rd32(body + 12);
  if (out->version_major != XPAR_FORMAT_MAJOR) return XPAR_E_UNSUPPORTED;
  /*  Reserved field must be zero.  */
  if (xpar_rd64(body + 16))                    return XPAR_E_MALFORMED;
  return XPAR_OK;
}

void xpar_volh_write(xpar_buf * out, const xpar_volh * v, const u8 * set_id,
                     const xpar_key * key) {
  u8 b[24];
  xpar_memset(b, 0, sizeof b);
  xpar_wr32(b,      v->volume_index);
  xpar_wr32(b + 4,  v->volume_kind);
  /*  The version written is this implementation's, never the struct's:
      claiming a version one does not implement has no correct use.  */
  xpar_wr32(b + 8,  XPAR_FORMAT_MAJOR);
  xpar_wr32(b + 12, XPAR_FORMAT_MINOR);
  xpar_pkt_write(out, XPAR_T_VOLH, pkt_flags(0, key), set_id, b, sizeof b,
                 key);
}

static xpar_status setd_body(const u8 * body, sz n, xpar_setd * out) {
  u64 want, hi;  u32 f;  bool zero_parent = true;

  if (n < 80) return XPAR_E_SHORT;
  f = xpar_rd32(body + 24);
  if (!f && !xpar_rd32(body + 64)) return XPAR_E_MALFORMED;
  /*  80 + 16F is already a multiple of 8, so the body carries no padding
      and F is pinned by the body length before it sizes anything.  */
  want = (u64) 80 + (u64) f * 16;
  if (want != (u64) n) return XPAR_E_MALFORMED;

  out->slice_size         = xpar_rd64(body);
  out->data_slice_count   = xpar_rd64(body + 8);
  out->stream_length      = xpar_rd64(body + 16);
  out->file_count         = f;
  out->field_log2         = body[28];
  out->codec              = body[29];
  out->recovery_axis_log2 = body[30];
  out->layout             = body[31];
  out->align              = body[32];
  out->slice_tag_len      = body[33];
  out->dedup_level        = body[34];
  out->required_features  = xpar_rd32(body + 36);
  out->optional_features  = xpar_rd32(body + 40);
  out->cell_bytes         = xpar_rd32(body + 44);
  out->generation         = xpar_rd32(body + 64);
  out->posix_record_count = xpar_rd32(body + 68);
  out->stream_base        = xpar_rd64(body + 72);
  xpar_memcpy(out->parent_set_id, body + 48, XPAR_SET_ID_LEN);

  if (out->field_log2 != 8 && out->field_log2 != 16) return XPAR_E_MALFORMED;
  if (out->codec > XPAR_CODEC_FFT_LOW)               return XPAR_E_MALFORMED;
  if (out->layout > XPAR_LAYOUT_ARMOURED)            return XPAR_E_MALFORMED;
  if (out->slice_tag_len != 0 && out->slice_tag_len != 8 &&
      out->slice_tag_len != 16)                      return XPAR_E_MALFORMED;
  /*  Reserved field must be zero.  */
  if (body[35])                                      return XPAR_E_MALFORMED;
  if (out->align > XPAR_ALIGN_1K)                    return XPAR_E_MALFORMED;
  if (out->slice_size % 64)                          return XPAR_E_MALFORMED;
  if (out->recovery_axis_log2 > out->field_log2)     return XPAR_E_MALFORMED;
  if (out->dedup_level > XPAR_DEDUP_CHUNK)           return XPAR_E_MALFORMED;

  if (out->required_features & XPAR_FEAT_B3_SUBTREE) {
    if (out->align != XPAR_ALIGN_1K || !out->slice_tag_len ||
        out->slice_size < XPAR_BLAKE3_CHUNK_LEN ||
        (out->slice_size & (out->slice_size - 1)) != 0 ||
        out->stream_base % XPAR_BLAKE3_CHUNK_LEN)
      return XPAR_E_MALFORMED;
  }

  if (out->slice_size < XPAR_SLICE_MIN ||
      out->slice_size > XPAR_SLICE_MAX)              return XPAR_E_MALFORMED;
  if (out->data_slice_count > (((u64) 1 << out->field_log2) - 1))
    return XPAR_E_MALFORMED;
  if (out->data_slice_count) {
    u64 order = (u64) 1 << out->field_log2;
    u64 axis = (u64) 1 << out->recovery_axis_log2;
    if (out->codec == XPAR_CODEC_MATRIX &&
        out->recovery_axis_log2 != out->field_log2)
      return XPAR_E_MALFORMED;
    if (out->codec == XPAR_CODEC_FFT &&
        (axis > order - out->data_slice_count)) return XPAR_E_MALFORMED;
    if (out->codec == XPAR_CODEC_FFT_LOW &&
        axis != xpar_next_pow2(out->data_slice_count))
      return XPAR_E_MALFORMED;
  }

  if (out->data_slice_count == 0) {
    if (out->stream_length != 0)                     return XPAR_E_MALFORMED;
  } else {
    hi = out->data_slice_count * out->slice_size;
    if (out->stream_length > hi ||
        out->stream_length <= hi - out->slice_size)   return XPAR_E_MALFORMED;
  }

  if (out->cell_bytes != 0 &&
      (out->cell_bytes < XPAR_CELL_MIN || out->cell_bytes % 64 ||
       (u64) out->cell_bytes > out->slice_size))     return XPAR_E_MALFORMED;
  /*  K = ceil(Z/Y) is capped, and every entry point reads SETD, so the
      bound belongs here rather than only in the geometry builder.  */
  if (out->cell_bytes != 0 &&
      xpar_ceil_div(out->slice_size, out->cell_bytes) > XPAR_CELLS_MAX)
    return XPAR_E_MALFORMED;

  Fi(XPAR_SET_ID_LEN, if (out->parent_set_id[i]) zero_parent = false)
  if (out->generation == 0) {
    if (!zero_parent || out->stream_base != 0)       return XPAR_E_MALFORMED;
  } else if (zero_parent)                            return XPAR_E_MALFORMED;

  out->file_id = (u8 (*)[XPAR_SET_ID_LEN])
                   xpar_calloc(f ? f : 1, XPAR_SET_ID_LEN);
  xpar_memcpy(out->file_id, body + 80, n - 80);

  if (out->required_features & ~XPAR_REQUIRED_KNOWN)
    return XPAR_E_UNSUPPORTED;
  return XPAR_OK;
}

xpar_status xpar_setd_read(const u8 * body, sz n, xpar_setd * out) {
  xpar_status st;
  xpar_memset(out, 0, sizeof *out);
  st = setd_body(body, n, out);
  if (st != XPAR_OK && st != XPAR_E_UNSUPPORTED) xpar_setd_free(out);
  return st;
}

void xpar_setd_free(xpar_setd * s) {
  xpar_free(s->file_id);
  xpar_memset(s, 0, sizeof *s);
}

void xpar_setd_write(xpar_buf * out, const xpar_setd * s, const u8 * set_id,
                     const xpar_key * key) {
  u8 b[80];  xpar_part part[2];
  xpar_memset(b, 0, sizeof b);
  xpar_wr64(b,      s->slice_size);
  xpar_wr64(b + 8,  s->data_slice_count);
  xpar_wr64(b + 16, s->stream_length);
  xpar_wr32(b + 24, s->file_count);
  b[28] = s->field_log2;          b[29] = s->codec;
  b[30] = s->recovery_axis_log2;  b[31] = s->layout;
  b[32] = s->align;               b[33] = s->slice_tag_len;
  b[34] = s->dedup_level;
  xpar_wr32(b + 36, s->required_features);
  xpar_wr32(b + 40, s->optional_features);
  xpar_wr32(b + 44, s->cell_bytes);
  xpar_memcpy(b + 48, s->parent_set_id, XPAR_SET_ID_LEN);
  xpar_wr32(b + 64, s->generation);
  xpar_wr32(b + 68, s->posix_record_count);
  xpar_wr64(b + 72, s->stream_base);

  part[0].p = b;             part[0].n = sizeof b;
  part[1].p = s->file_id;    part[1].n = (sz) s->file_count * XPAR_SET_ID_LEN;
  xpar_pkt_writev(out, XPAR_T_SETD, pkt_flags(XPAR_PF_CRITICAL, key), set_id,
                  part, 2, key);
}

xpar_status xpar_setd_check_parent(const xpar_setd * c,
                                   const u8 * parent_set_id,
                                   const xpar_setd * parent) {
  u64 base;
  if (c->generation == 0)
    return (parent || parent_set_id) ? XPAR_E_MALFORMED : XPAR_OK;
  if (!parent || !parent_set_id) return XPAR_E_MALFORMED;
  if (xpar_memcmp(c->parent_set_id, parent_set_id, XPAR_SET_ID_LEN) != 0)
    return XPAR_E_MALFORMED;
  if (c->generation != parent->generation + 1) return XPAR_E_MALFORMED;
  if (!add64(parent->stream_base, parent->stream_length, &base))
    return XPAR_E_MALFORMED;
  if (c->stream_base != base) return XPAR_E_MALFORMED;
  return XPAR_OK;
}

static xpar_status entry_body(const u8 * body, sz n, u32 prc,
                              xpar_entry * out) {
  u64 need, sum = 0;  u32 ec, nl, xl, i;  const u8 * ep;

  if (n < 128) return XPAR_E_SHORT;
  ec = xpar_rd32(body + 112);
  nl = xpar_rd32(body + 120);
  xl = xpar_rd32(body + 124);
  if (nl > XPAR_NAME_MAX || xl > XPAR_EXTRA_MAX) return XPAR_E_MALFORMED;
  /*  ec is a u32 so 16*ec is at most 2^36 and the sum cannot wrap. This is
      the line that stops extent_count = 0xFFFFFFFF from ever reaching an
      allocator: 16*ec is compared against the bytes present first.  */
  need = (u64) 128 + (u64) ec * 16 + nl + xl;
  if (need > (u64) n) return XPAR_E_MALFORMED;
  /*  At most seven bytes of packet padding may follow the last field.  */
  if (!pad_ok(body, need, n)) return XPAR_E_MALFORMED;

  xpar_memcpy(out->file_id, body, XPAR_SET_ID_LEN);
  out->length     = xpar_rd64(body + 16);
  xpar_memcpy(out->content_hash, body + 24, 32);
  xpar_memcpy(out->prefix_hash,  body + 56, 16);
  out->mtime_ns   = (i64) xpar_rd64(body + 72);
  out->atime_ns   = (i64) xpar_rd64(body + 80);
  out->ctime_ns   = (i64) xpar_rd64(body + 88);
  out->btime_ns   = (i64) xpar_rd64(body + 96);
  out->mode       = xpar_rd32(body + 104);
  out->posix_index= xpar_rd32(body + 108);
  out->extent_count = ec;
  out->entry_type = xpar_rd16(body + 116);
  out->attrs      = xpar_rd16(body + 118);
  if (out->attrs & ~(u32) XPAR_ATTR_KNOWN) return XPAR_E_MALFORMED;
  out->name_len   = nl;
  out->extra_len  = xl;

  if (out->entry_type > XPAR_ENTRY_HARDLINK) return XPAR_E_MALFORMED;
  if (out->entry_type != XPAR_ENTRY_REGULAR && ec != 0)
    return XPAR_E_MALFORMED;
  if (out->posix_index != XPAR_ABSENT_U32 && prc != XPAR_ABSENT_U32 &&
      out->posix_index >= prc) return XPAR_E_MALFORMED;

  if (ec) {
    out->extents = (xpar_extent *) xpar_calloc(ec, sizeof(xpar_extent));
    ep = body + 128;
    for (i = 0; i < ec; i++) {
      u64 off = xpar_rd64(ep), len = xpar_rd64(ep + 8), end;
      ep += 16;
      if (len == 0)                 return XPAR_E_MALFORMED;  /*  Rule 1.  */
      if (!add64(off, len, &end))   return XPAR_E_MALFORMED;  /*  Rule 3.  */
      if (!add64(sum, len, &sum))   return XPAR_E_MALFORMED;
      out->extents[i].stream_offset = off;
      out->extents[i].length        = len;
    }
  }
  /*  Rule 2, which also covers the zero-extent case: a regular entry with
      no extents describes no bytes and its length must say so.  */
  if (out->entry_type == XPAR_ENTRY_REGULAR && sum != out->length)
    return XPAR_E_MALFORMED;

  if (nl) out->name  = dup_str(body + 128 + (sz) ec * 16, nl);
  if (xl) {
    out->extra = (u8 *) xpar_malloc(xl + 1);
    xpar_memcpy(out->extra, body + 128 + (sz) ec * 16 + nl, xl);
    out->extra[xl] = 0;   /* Match dup_str's terminator. */
  }
  return XPAR_OK;
}

xpar_status xpar_entry_read(const u8 * body, sz n, u32 prc, xpar_entry * out) {
  xpar_status st;
  xpar_memset(out, 0, sizeof *out);
  st = entry_body(body, n, prc, out);
  if (st != XPAR_OK) xpar_entry_free(out);
  return st;
}

void xpar_entry_write(xpar_buf * out, const xpar_entry * e,
                      const u8 * set_id, const xpar_key * key,
                      const xpar_wropt * o) {
  u8 fixed[128];  xpar_part part[4];  u8 * ex = NULL;  u32 i;
  i64 mt = e->mtime_ns, at = e->atime_ns;
  i64 ct = e->ctime_ns, bt = e->btime_ns;
  u32 pi = e->posix_index;

  if (o && o->reproducible) {
    if (!o->keep_mtime) mt = XPAR_ABSENT_TIME;
    if (!o->keep_atime) at = XPAR_ABSENT_TIME;
    if (!o->keep_ctime) ct = XPAR_ABSENT_TIME;
    if (!o->keep_btime) bt = XPAR_ABSENT_TIME;
    if (!o->keep_posix) pi = XPAR_ABSENT_U32;
  }

  xpar_assert(e->name_len <= XPAR_NAME_MAX && e->extra_len <= XPAR_EXTRA_MAX);

  xpar_memset(fixed, 0, sizeof fixed);
  xpar_memcpy(fixed, e->file_id, XPAR_SET_ID_LEN);
  xpar_wr64(fixed + 16, e->length);
  xpar_memcpy(fixed + 24, e->content_hash, 32);
  xpar_memcpy(fixed + 56, e->prefix_hash,  16);
  xpar_wr64(fixed + 72, (u64) mt);
  xpar_wr64(fixed + 80, (u64) at);
  xpar_wr64(fixed + 88, (u64) ct);
  xpar_wr64(fixed + 96, (u64) bt);
  xpar_wr32(fixed + 104, e->mode);
  xpar_wr32(fixed + 108, pi);
  xpar_wr32(fixed + 112, e->extent_count);
  xpar_wr16(fixed + 116, e->entry_type);
  xpar_wr16(fixed + 118, e->attrs);
  xpar_wr32(fixed + 120, e->name_len);
  xpar_wr32(fixed + 124, e->extra_len);

  if (e->extent_count) {
    ex = (u8 *) xpar_calloc(e->extent_count, 16);
    for (i = 0; i < e->extent_count; i++) {
      xpar_wr64(ex + (sz) i * 16,     e->extents[i].stream_offset);
      xpar_wr64(ex + (sz) i * 16 + 8, e->extents[i].length);
    }
  }
  part[0].p = fixed;     part[0].n = sizeof fixed;
  part[1].p = ex;        part[1].n = (sz) e->extent_count * 16;
  part[2].p = e->name;   part[2].n = e->name_len;
  part[3].p = e->extra;  part[3].n = e->extra_len;
  xpar_pkt_writev(out, XPAR_T_FILE, pkt_flags(XPAR_PF_CRITICAL, key), set_id,
                  part, 4, key);
  xpar_free(ex);
}

static xpar_status posx_rec(const u8 * b, sz avail, xpar_posix_rec * r,
                            sz * used) {
  u32 ol, gl, xc, j;  sz p = 16;
  const u8 * prev = NULL;  sz prev_n = 0;

  if (avail < 16) return XPAR_E_SHORT;
  r->uid = xpar_rd32(b);
  r->gid = xpar_rd32(b + 4);
  ol = xpar_rd16(b + 8);
  gl = xpar_rd16(b + 10);
  xc = xpar_rd16(b + 12);
  if (ol > 255 || gl > 255) return XPAR_E_MALFORMED;
  /*  Reserved field must be zero.  */
  if (xpar_rd16(b + 14)) return XPAR_E_MALFORMED;

  if (ol > avail - p) return XPAR_E_MALFORMED;
  if (ol) {
    if (xpar_has_nul(b + p, ol)) return XPAR_E_MALFORMED;
    r->owner = dup_str(b + p, ol);
    p += ol;
  }
  if (gl > avail - p) return XPAR_E_MALFORMED;
  if (gl) {
    if (xpar_has_nul(b + p, gl)) return XPAR_E_MALFORMED;
    r->group = dup_str(b + p, gl);
    p += gl;
  }

  if (xc) {
    /*  Four bytes of header per xattr, checked before the array is sized
        so that xattr_count cannot drive an allocation on its own.  */
    if ((u64) xc * 4 > (u64) (avail - p)) return XPAR_E_MALFORMED;
    r->xattrs = (xpar_xattr *) xpar_calloc(xc, sizeof(xpar_xattr));
    r->xattr_count = xc;
    for (j = 0; j < xc; j++) {
      u32 nl, vl;
      if (avail - p < 4) return XPAR_E_MALFORMED;
      nl = xpar_rd16(b + p);  vl = xpar_rd16(b + p + 2);  p += 4;
      if (nl < 1) return XPAR_E_MALFORMED;
      if (nl > avail - p) return XPAR_E_MALFORMED;
      if (xpar_has_nul(b + p, nl)) return XPAR_E_MALFORMED;
      if (prev && bytes_cmp(prev, prev_n, b + p, nl) >= 0)
        return XPAR_E_MALFORMED;
      prev = b + p;  prev_n = nl;
      r->xattrs[j].name = dup_str(b + p, nl);
      p += nl;
      if (vl > avail - p) return XPAR_E_MALFORMED;
      r->xattrs[j].value_len = vl;
      if (vl) {
        r->xattrs[j].value = (u8 *) xpar_malloc(vl);
        xpar_memcpy(r->xattrs[j].value, b + p, vl);
        p += vl;
      }
    }
  }

  *used = (sz) xpar_align_up(p, XPAR_PKT_ALIGN);
  if (*used > avail) return XPAR_E_MALFORMED;
  return XPAR_OK;
}

static xpar_status posx_body(const u8 * body, sz n, xpar_posx * out) {
  u32 cnt, i;  sz p = 8;
  if (n < 8) return XPAR_E_SHORT;
  out->first_record = xpar_rd32(body);
  cnt = xpar_rd32(body + 4);
  if (cnt < 1) return XPAR_E_MALFORMED;
  /*  A record is at least 16 bytes, so a count that could not fit even at
      the minimum size is refused before the array is allocated.  */
  if ((u64) cnt * 16 > (u64) n - 8) return XPAR_E_MALFORMED;
  if ((u64) out->first_record + cnt > 0xFFFFFFFFu) return XPAR_E_MALFORMED;
  out->rec = (xpar_posix_rec *) xpar_calloc(cnt, sizeof(xpar_posix_rec));
  out->count = cnt;
  for (i = 0; i < cnt; i++) {
    sz used = 0;
    xpar_status st = posx_rec(body + p, n - p, &out->rec[i], &used);
    if (st != XPAR_OK) return st;
    p += used;
  }
  if (!pad_ok(body, p, n)) return XPAR_E_MALFORMED;
  return XPAR_OK;
}

xpar_status xpar_posx_read(const u8 * body, sz n, xpar_posx * out) {
  xpar_status st;
  xpar_memset(out, 0, sizeof *out);
  st = posx_body(body, n, out);
  if (st != XPAR_OK) xpar_posx_free(out);
  return st;
}

void xpar_posx_free(xpar_posx * t) {
  For(u32, i, t->count, xpar_posix_rec_free(&t->rec[i]))
  xpar_free(t->rec);
  xpar_memset(t, 0, sizeof *t);
}

static u64 posx_rec_size(const xpar_posix_rec * r) {
  u64 n = 16;  u32 j;
  n += r->owner ? xpar_strlen(r->owner) : 0;
  n += r->group ? xpar_strlen(r->group) : 0;
  for (j = 0; j < r->xattr_count; j++)
    n += 4 + (r->xattrs[j].name ? xpar_strlen(r->xattrs[j].name) : 0) +
         r->xattrs[j].value_len;
  return xpar_align_up(n, XPAR_PKT_ALIGN);
}

static void posx_rec_put(xpar_buf * b, const xpar_posix_rec * r) {
  sz ol = r->owner ? xpar_strlen(r->owner) : 0;
  sz gl = r->group ? xpar_strlen(r->group) : 0;
  sz at = b->len;  u32 j;
  u8 * h;
  xpar_assert(ol <= 255 && gl <= 255 && r->xattr_count <= 0xFFFF);
  for (j = 0; j < r->xattr_count; j++) {
    sz nl = r->xattrs[j].name ? xpar_strlen(r->xattrs[j].name) : 0;
    xpar_assert(nl >= 1 && nl <= 0xFFFF);
    xpar_assert(r->xattrs[j].value_len <= 0xFFFF);
  }
  h = xpar_buf_grow(b, 16);
  xpar_wr32(h,     r->uid);
  xpar_wr32(h + 4, r->gid);
  xpar_wr16(h + 8,  (u16) ol);
  xpar_wr16(h + 10, (u16) gl);
  xpar_wr16(h + 12, (u16) r->xattr_count);
  xpar_buf_put(b, r->owner, ol);
  xpar_buf_put(b, r->group, gl);
  for (j = 0; j < r->xattr_count; j++) {
    sz nl = r->xattrs[j].name ? xpar_strlen(r->xattrs[j].name) : 0;
    u8 * e = xpar_buf_grow(b, 4);
    xpar_wr16(e,     (u16) nl);
    xpar_wr16(e + 2, (u16) r->xattrs[j].value_len);
    xpar_buf_put(b, r->xattrs[j].name,  nl);
    xpar_buf_put(b, r->xattrs[j].value, r->xattrs[j].value_len);
  }
  xpar_buf_grow(b, (sz) (xpar_align_up(b->len - at, XPAR_PKT_ALIGN) -
                         (b->len - at)));
}

void xpar_posx_write(xpar_buf * out, u32 first_record, u32 count,
                     const xpar_posix_rec * rec, const u8 * set_id,
                     const xpar_key * key) {
  xpar_buf body;  u8 * h;  u32 i;
  xpar_buf_init(&body);
  h = xpar_buf_grow(&body, 8);
  xpar_wr32(h,     first_record);
  xpar_wr32(h + 4, count);
  for (i = 0; i < count; i++) posx_rec_put(&body, &rec[i]);
  xpar_pkt_write(out, XPAR_T_POSX, pkt_flags(XPAR_PF_CRITICAL, key), set_id,
                 body.data, body.len, key);
  xpar_buf_free(&body);
}

void xpar_posx_write_all(xpar_buf * out, const xpar_posix_rec * rec,
                         u32 count, const u8 * set_id, const xpar_key * key) {
  u32 i = 0;
  while (i < count) {
    u64 bytes = 8;  u32 j = i;
    while (j < count) {
      u64 rs = posx_rec_size(&rec[j]);
      if (j > i && bytes + rs > XPAR_POSX_SPLIT) break;
      bytes += rs;  j++;
    }
    xpar_posx_write(out, i, j - i, &rec[i], set_id, key);
    i = j;
  }
}

static xpar_status slcr_body(const u8 * body, sz n, xpar_slcr * out) {
  u64 need, last;
  if (n < 16) return XPAR_E_SHORT;
  out->first_slice = xpar_rd64(body);
  out->count       = xpar_rd64(body + 8);
  if (out->count < 1) return XPAR_E_MALFORMED;
  if (out->count > XPAR_TABLE_SPLIT) return XPAR_E_MALFORMED;
  if (out->count > ((u64) n - 16) / 4) return XPAR_E_MALFORMED;
  if (!add64(out->first_slice, out->count, &last)) return XPAR_E_MALFORMED;
  need = 16 + out->count * 4;
  if (!pad_ok(body, need, n)) return XPAR_E_MALFORMED;
  { u64 i;
    out->crc = (u32 *) xpar_calloc((sz) out->count, 4);
    for (i = 0; i < out->count; i++)
      out->crc[i] = xpar_rd32(body + 16 + (sz) (i * 4)); }
  return XPAR_OK;
}

xpar_status xpar_slcr_read(const u8 * body, sz n, xpar_slcr * out) {
  xpar_status st;
  xpar_memset(out, 0, sizeof *out);
  st = slcr_body(body, n, out);
  if (st != XPAR_OK) xpar_slcr_free(out);
  return st;
}

void xpar_slcr_free(xpar_slcr * t) {
  xpar_free(t->crc);  xpar_memset(t, 0, sizeof *t);
}

static xpar_status sltg_body(const u8 * body, sz n, xpar_sltg * out) {
  u64 need, last;
  if (n < 24) return XPAR_E_SHORT;
  out->first_slice = xpar_rd64(body);
  out->count       = xpar_rd64(body + 8);
  out->tag_len     = body[16];
  if (out->count < 1) return XPAR_E_MALFORMED;
  if (out->count > XPAR_TABLE_SPLIT) return XPAR_E_MALFORMED;
  if (out->tag_len != 8 && out->tag_len != 16) return XPAR_E_MALFORMED;
  /*  Bytes 17..23 are reserved and shall be zero.  */
  { sz q;  for (q = 17; q < 24; q++) if (body[q]) return XPAR_E_MALFORMED; }
  if (out->count > ((u64) n - 24) / out->tag_len) return XPAR_E_MALFORMED;
  if (!add64(out->first_slice, out->count, &last)) return XPAR_E_MALFORMED;
  need = 24 + out->count * out->tag_len;
  if (!pad_ok(body, need, n)) return XPAR_E_MALFORMED;
  out->tag = (u8 *) xpar_calloc((sz) out->count, out->tag_len);
  xpar_memcpy(out->tag, body + 24, (sz) (out->count * out->tag_len));
  return XPAR_OK;
}

xpar_status xpar_sltg_read(const u8 * body, sz n, xpar_sltg * out) {
  xpar_status st;
  xpar_memset(out, 0, sizeof *out);
  st = sltg_body(body, n, out);
  if (st != XPAR_OK) xpar_sltg_free(out);
  return st;
}

void xpar_sltg_free(xpar_sltg * t) {
  xpar_free(t->tag);  xpar_memset(t, 0, sizeof *t);
}

static xpar_status slcl_body(const u8 * body, sz n, u64 slice_size,
                             xpar_slcl * out) {
  u64 need, cells, k, last;
  if (n < 24) return XPAR_E_SHORT;
  if (!slice_size) return XPAR_E_MALFORMED;
  out->first_slice = xpar_rd64(body);
  out->count       = xpar_rd64(body + 8);
  out->cell_bytes  = xpar_rd32(body + 16);
  if (out->count < 1) return XPAR_E_MALFORMED;
  /*  Bytes 20..23 are reserved and shall be zero.  */
  if (xpar_rd32(body + 20)) return XPAR_E_MALFORMED;
  if (!out->cell_bytes || (u64) out->cell_bytes > slice_size)
    return XPAR_E_MALFORMED;
  k = xpar_ceil_div(slice_size, out->cell_bytes);
  if (k > 0xFFFFFFFFu) return XPAR_E_MALFORMED;
  out->cells_per_slice = (u32) k;
  if (k > XPAR_TABLE_SPLIT || out->count > XPAR_TABLE_SPLIT / k)
    return XPAR_E_MALFORMED;
  if (out->count > (((u64) n - 24) / 4) / k) return XPAR_E_MALFORMED;
  if (!add64(out->first_slice, out->count, &last)) return XPAR_E_MALFORMED;
  if (!mul64(out->count, k, &cells)) return XPAR_E_MALFORMED;
  need = 24 + cells * 4;
  if (!pad_ok(body, need, n)) return XPAR_E_MALFORMED;
  { u64 i;
    out->crc = (u32 *) xpar_calloc((sz) cells, 4);
    for (i = 0; i < cells; i++)
      out->crc[i] = xpar_rd32(body + 24 + (sz) (i * 4)); }
  return XPAR_OK;
}

xpar_status xpar_slcl_read(const u8 * body, sz n, u64 slice_size,
                           xpar_slcl * out) {
  xpar_status st;
  xpar_memset(out, 0, sizeof *out);
  st = slcl_body(body, n, slice_size, out);
  if (st != XPAR_OK) xpar_slcl_free(out);
  return st;
}

void xpar_slcl_free(xpar_slcl * t) {
  xpar_free(t->crc);  xpar_memset(t, 0, sizeof *t);
}

void xpar_slcr_write(xpar_buf * out, u64 first_slice, u64 count,
                     const u32 * crc, const u8 * set_id,
                     const xpar_key * key) {
  xpar_buf body;  u8 * h;  u64 i;
  xpar_buf_init(&body);
  h = xpar_buf_grow(&body, 16);
  xpar_wr64(h,     first_slice);
  xpar_wr64(h + 8, count);
  h = xpar_buf_grow(&body, (sz) (count * 4));
  for (i = 0; i < count; i++) xpar_wr32(h + (sz) (i * 4), crc[i]);
  xpar_pkt_write(out, XPAR_T_SLCR, pkt_flags(XPAR_PF_CRITICAL, key), set_id,
                 body.data, body.len, key);
  xpar_buf_free(&body);
}

void xpar_sltg_write(xpar_buf * out, u64 first_slice, u64 count,
                     u8 tag_len, const u8 * tag, const u8 * set_id,
                     const xpar_key * key) {
  xpar_buf body;  u8 * h;
  xpar_buf_init(&body);
  h = xpar_buf_grow(&body, 24);
  xpar_wr64(h,     first_slice);
  xpar_wr64(h + 8, count);
  h[16] = tag_len;
  xpar_buf_put(&body, tag, (sz) (count * tag_len));
  xpar_pkt_write(out, XPAR_T_SLTG, pkt_flags(0, key), set_id, body.data,
                 body.len, key);
  xpar_buf_free(&body);
}

void xpar_slcl_write(xpar_buf * out, u64 first_slice, u64 count,
                     u32 cell_bytes, u32 cells_per_slice, const u32 * crc,
                     const u8 * set_id, const xpar_key * key) {
  xpar_buf body;  u8 * h;  u64 i, cells = count * cells_per_slice;
  xpar_buf_init(&body);
  h = xpar_buf_grow(&body, 24);
  xpar_wr64(h,      first_slice);
  xpar_wr64(h + 8,  count);
  xpar_wr32(h + 16, cell_bytes);
  h = xpar_buf_grow(&body, (sz) (cells * 4));
  for (i = 0; i < cells; i++) xpar_wr32(h + (sz) (i * 4), crc[i]);
  xpar_pkt_write(out, XPAR_T_SLCL, pkt_flags(0, key), set_id, body.data,
                 body.len, key);
  xpar_buf_free(&body);
}

void xpar_slcr_write_all(xpar_buf * out, const u32 * crc, u64 slices,
                         const u8 * set_id, const xpar_key * key) {
  u64 i;
  for (i = 0; i < slices; i += XPAR_TABLE_SPLIT) {
    xpar_slcr_write(out, i, MIN(slices - i, (u64) XPAR_TABLE_SPLIT),
                    crc + i, set_id, key);
  }
}

void xpar_sltg_write_all(xpar_buf * out, const u8 * tag, u64 slices,
                         u8 tag_len, const u8 * set_id,
                         const xpar_key * key) {
  u64 i;
  for (i = 0; i < slices; i += XPAR_TABLE_SPLIT) {
    xpar_sltg_write(out, i, MIN(slices - i, (u64) XPAR_TABLE_SPLIT),
                    tag_len, tag + i * tag_len, set_id, key);
  }
}

void xpar_slcl_write_all(xpar_buf * out, const u32 * crc, u64 slices,
                         u32 cell_bytes, u32 cells_per_slice,
                         const u8 * set_id, const xpar_key * key) {
  u64 per = XPAR_TABLE_SPLIT / (cells_per_slice ? cells_per_slice : 1), i;
  if (!per) per = 1;
  for (i = 0; i < slices; i += per) {
    xpar_slcl_write(out, i, MIN(slices - i, per), cell_bytes,
                    cells_per_slice, crc + i * cells_per_slice,
                    set_id, key);
  }
}

bool xpar_tagset_init(xpar_tagset * s, u64 slices, u8 tag_len, u32 cps,
                      u64 input_bytes) {
  u64 per, total;
  xpar_memset(s, 0, sizeof *s);
  s->t.slice_count     = slices;
  s->t.tag_len         = tag_len;
  s->t.cells_per_slice = cps;
  if (!slices) return true;
  if (!add64(4 + tag_len, (u64) cps * 4, &per)) return false;
  if (!mul64(slices, per, &total)) return false;
  if (total > input_bytes) return false;
  if (total > (u64) (sz) -1 / 2) return false;

  s->t.slice_crc = (u32 *) xpar_calloc((sz) slices, 4);
  s->seen_crc    = (u8 *)  xpar_calloc((sz) slices, 1);
  if (tag_len) {
    s->t.slice_tag = (u8 *) xpar_calloc((sz) slices, tag_len);
    s->seen_tag    = (u8 *) xpar_calloc((sz) slices, 1);
  }
  if (cps) {
    s->t.cell_crc = (u32 *) xpar_calloc((sz) (slices * cps), 4);
    s->seen_cell  = (u8 *)  xpar_calloc((sz) slices, 1);
  }
  return true;
}

void xpar_tagset_free(xpar_tagset * s) {
  xpar_free(s->t.slice_crc);  xpar_free(s->t.slice_tag);
  xpar_free(s->t.cell_crc);   xpar_free(s->seen_crc);
  xpar_free(s->seen_tag);     xpar_free(s->seen_cell);
  xpar_memset(s, 0, sizeof *s);
}

static xpar_status tagset_range(const xpar_tagset * s, const u8 * seen,
                                u64 first, u64 count) {
  u64 end, i;
  if (!seen) return XPAR_E_MALFORMED;
  if (!add64(first, count, &end)) return XPAR_E_MALFORMED;
  if (end > s->t.slice_count) return XPAR_E_MALFORMED;
  for (i = first; i < end; i++) if (seen[i]) return XPAR_E_MALFORMED;
  return XPAR_OK;
}

xpar_status xpar_tagset_slcr(xpar_tagset * s, const xpar_slcr * t) {
  u64 i;
  xpar_status st = tagset_range(s, s->seen_crc, t->first_slice, t->count);
  if (st != XPAR_OK) return st;
  for (i = 0; i < t->count; i++) {
    s->t.slice_crc[t->first_slice + i] = t->crc[i];
    s->seen_crc[t->first_slice + i] = 1;
  }
  return XPAR_OK;
}

xpar_status xpar_tagset_sltg(xpar_tagset * s, const xpar_sltg * t) {
  u64 i;
  xpar_status st;
  if (t->tag_len != s->t.tag_len) return XPAR_E_MALFORMED;
  st = tagset_range(s, s->seen_tag, t->first_slice, t->count);
  if (st != XPAR_OK) return st;
  for (i = 0; i < t->count; i++) {
    xpar_memcpy(s->t.slice_tag + (sz) ((t->first_slice + i) * t->tag_len),
                t->tag + (sz) (i * t->tag_len), t->tag_len);
    s->seen_tag[t->first_slice + i] = 1;
  }
  return XPAR_OK;
}

xpar_status xpar_tagset_slcl(xpar_tagset * s, const xpar_slcl * t) {
  u64 i, c;
  xpar_status st;
  if (t->cells_per_slice != s->t.cells_per_slice) return XPAR_E_MALFORMED;
  st = tagset_range(s, s->seen_cell, t->first_slice, t->count);
  if (st != XPAR_OK) return st;
  for (i = 0; i < t->count; i++) {
    for (c = 0; c < t->cells_per_slice; c++)
      s->t.cell_crc[(t->first_slice + i) * t->cells_per_slice + c] =
        t->crc[i * t->cells_per_slice + c];
    s->seen_cell[t->first_slice + i] = 1;
  }
  return XPAR_OK;
}

u32 xpar_tagset_complete(const xpar_tagset * s) {
  u32 r = XPAR_TAGS_CRC | XPAR_TAGS_TAG | XPAR_TAGS_CELL;  u64 i;
  if (!s->t.tag_len)         r &= ~XPAR_TAGS_TAG;
  if (!s->t.cells_per_slice) r &= ~XPAR_TAGS_CELL;
  for (i = 0; i < s->t.slice_count; i++) {
    if (!s->seen_crc || !s->seen_crc[i])                 r &= ~XPAR_TAGS_CRC;
    if ((r & XPAR_TAGS_TAG) && (!s->seen_tag || !s->seen_tag[i]))
      r &= ~XPAR_TAGS_TAG;
    if ((r & XPAR_TAGS_CELL) && (!s->seen_cell || !s->seen_cell[i]))
      r &= ~XPAR_TAGS_CELL;
  }
  return r;
}

xpar_status xpar_rcvs_read(const u8 * body, sz n, u64 slice_size,
                           xpar_rcvs * out) {
  u64 z;
  xpar_memset(out, 0, sizeof *out);
  if (n < 16) return XPAR_E_SHORT;
  z = slice_size ? slice_size : (u64) n - 16;
  if ((u64) n != 16 + z) return XPAR_E_MALFORMED;
  if (xpar_rd64(body + 8)) return XPAR_E_MALFORMED;
  out->exponent = xpar_rd64(body);
  out->data     = body + 16;
  out->length   = z;
  return XPAR_OK;
}

void xpar_rcvs_write(xpar_buf * out, u64 exponent, const void * data, sz len,
                     const u8 * set_id, const xpar_key * key) {
  u8 h[16];  xpar_part part[2];
  xpar_memset(h, 0, sizeof h);
  xpar_wr64(h, exponent);
  part[0].p = h;     part[0].n = sizeof h;
  part[1].p = data;  part[1].n = len;
  xpar_pkt_writev(out, XPAR_T_RCVS, pkt_flags(0, key), set_id, part, 2, key);
}

u32 xpar_rcvs_stream_header(u8 out[XPAR_PKT_HDR + 16], u64 exponent,
                            const void * data, sz len, const u8 * set_id,
                            const xpar_key * key) {
  static const u8 zero[XPAR_PKT_ALIGN] = { 0 };
  xpar_blake3_t h;
  u64 packet_len;
  u32 flags = pkt_flags(0, key);
  u32 pad;
#if UINTPTR_MAX > UINT32_MAX
  xpar_assert((u64) len <= XPAR_PKT_LEN_MAX - XPAR_PKT_HDR - 16);
#endif
  packet_len = xpar_align_up(XPAR_PKT_HDR + 16 + (u64) len,
                             XPAR_PKT_ALIGN);
  pad = (u32) (packet_len - XPAR_PKT_HDR - 16 - len);
  xpar_memset(out, 0, XPAR_PKT_HDR + 16);
  xpar_memcpy(out, XPAR_PKT_MAGIC, 8);
  xpar_wr64(out + 8, packet_len);
  xpar_memcpy(out + 16, set_id, XPAR_SET_ID_LEN);
  xpar_memcpy(out + 32, XPAR_T_RCVS, 4);
  xpar_wr32(out + 36, flags);
  xpar_wr64(out + XPAR_PKT_HDR, exponent);
  if (key) xpar_blake3_init_keyed(&h, key->k_pkt);
  else     xpar_blake3_init(&h);
  xpar_blake3_update(&h, out, 40);
  xpar_blake3_update(&h, out + XPAR_PKT_HDR, 16);
  if (len) xpar_blake3_update(&h, data, len);
  if (pad) xpar_blake3_update(&h, zero, pad);
  xpar_blake3_final(&h, out + 40, 8);
  return pad;
}

static bool layt_names_unique(const xpar_layt * l) {
  u64 want = 16;
  u32 capacity, i;
  u32 * slot;
  /*  Use u64 so an untrusted count cannot wrap the table size.  */
  while (want < (u64) l->count * 2) want *= 2;
  if (want > (u64) 1 << 31) return false;
  capacity = (u32) want;
  slot = (u32 *) xpar_calloc(capacity, sizeof *slot);
  for (i = 0; i < l->count; i++) {
    const char * name = l->vol[i].name;
    sz len = xpar_strlen(name);
    u8 digest[8];
    u32 at;
    xpar_blake3_hash(name, len, digest, sizeof digest);
    at = (u32) xpar_rd64(digest) & (capacity - 1);
    while (slot[at]) {
      const char * prior = l->vol[slot[at] - 1].name;
      if (xpar_strlen(prior) == len && !xpar_memcmp(prior, name, len)) {
        xpar_free(slot);
        return false;
      }
      at = (at + 1) & (capacity - 1);
    }
    slot[at] = i + 1;
  }
  xpar_free(slot);
  return true;
}

static xpar_status layt_body(const u8 * body, sz n, xpar_layt * out) {
  u32 v, i, indices = 0;
  sz p = 8;
  if (n < 8) return XPAR_E_SHORT;
  v = xpar_rd32(body);
  out->this_volume = xpar_rd32(body + 4);
  if (!v || out->this_volume != XPAR_VOL_STANDALONE)
    return XPAR_E_MALFORMED;
  /*  Every entry is at least 32 bytes, so a count that cannot fit at the
      minimum size is refused before the array exists.  */
  if ((u64) v * 32 > (u64) n - 8) return XPAR_E_MALFORMED;
  out->vol = (xpar_vol *) xpar_calloc(v, sizeof(xpar_vol));
  out->count = v;
  for (i = 0; i < v; i++) {
    const u8 * e = body + p;  u32 nl;  u64 used;
    if (n - p < 32) return XPAR_E_MALFORMED;
    out->vol[i].kind           = e[0];
    out->vol[i].vflags         = e[1];
    nl                         = xpar_rd16(e + 2);
    out->vol[i].recovery_first = xpar_rd32(e + 4);
    out->vol[i].stream_offset  = xpar_rd64(e + 8);
    out->vol[i].byte_length    = xpar_rd64(e + 16);
    out->vol[i].vol_tag        = xpar_rd64(e + 24);
    if (out->vol[i].kind > XPAR_VOL_RECOVERY ||
        (out->vol[i].vflags & ~1u)) return XPAR_E_MALFORMED;
    if (out->vol[i].kind == XPAR_VOL_INDEX) {
      indices++;
      if (out->vol[i].recovery_first || out->vol[i].stream_offset ||
          out->vol[i].byte_length || out->vol[i].vol_tag)
        return XPAR_E_MALFORMED;
    }
    if (out->vol[i].kind == XPAR_VOL_DATA && out->vol[i].recovery_first)
      return XPAR_E_MALFORMED;
    if (out->vol[i].kind == XPAR_VOL_RECOVERY &&
        (out->vol[i].stream_offset || out->vol[i].vol_tag))
      return XPAR_E_MALFORMED;
    if ((u64) nl > (u64) (n - p) - 32) return XPAR_E_MALFORMED;
    if (xpar_has_nul(e + 32, nl) ||
        xpar_path_check((const char *) e + 32, nl, XPAR_PATH_WIN) !=
          XPAR_PATH_OK) return XPAR_E_MALFORMED;
    /*  One path component, so a name cannot steer a volume open into a
        subdirectory. xpar_path_check alone accepts a relative path.  */
    For(u32, q, nl, if (e[32 + q] == '/' || e[32 + q] == '\\')
                      return XPAR_E_MALFORMED)
    out->vol[i].name = dup_str(e + 32, nl);
    used = xpar_align_up((u64) 32 + nl, XPAR_PKT_ALIGN);
    if (used > (u64) (n - p)) return XPAR_E_MALFORMED;
    for (u64 j = 32 + nl; j < used; j++)
      if (body[p + (sz) j]) return XPAR_E_MALFORMED;
    p += (sz) used;
  }
  if (indices != 1 || !layt_names_unique(out) ||
      (u64) n - p >= XPAR_PKT_ALIGN)
    return XPAR_E_MALFORMED;
  while (p < n) if (body[p++]) return XPAR_E_MALFORMED;
  return XPAR_OK;
}

xpar_status xpar_layt_read(const u8 * body, sz n, xpar_layt * out) {
  xpar_status st;
  xpar_memset(out, 0, sizeof *out);
  st = layt_body(body, n, out);
  if (st != XPAR_OK) xpar_layt_free(out);
  return st;
}

void xpar_layt_free(xpar_layt * l) {
  For(u32, i, l->count, xpar_free(l->vol[i].name))
  xpar_free(l->vol);
  xpar_memset(l, 0, sizeof *l);
}

void xpar_layt_write(xpar_buf * out, const xpar_layt * l, const u8 * set_id,
                     const xpar_key * key) {
  xpar_buf body;  u8 * h;  u32 i;
  xpar_buf_init(&body);
  h = xpar_buf_grow(&body, 8);
  xpar_wr32(h,     l->count);
  xpar_wr32(h + 4, XPAR_VOL_STANDALONE);
  for (i = 0; i < l->count; i++) {
    sz nl = l->vol[i].name ? xpar_strlen(l->vol[i].name) : 0;
    sz at = body.len;
    u8 * e;
    xpar_assert(nl <= 0xFFFF);
    e = xpar_buf_grow(&body, 32);
    e[0] = l->vol[i].kind;  e[1] = l->vol[i].vflags;
    xpar_wr16(e + 2,  (u16) nl);
    xpar_wr32(e + 4,  l->vol[i].recovery_first);
    xpar_wr64(e + 8,  l->vol[i].stream_offset);
    xpar_wr64(e + 16, l->vol[i].byte_length);
    xpar_wr64(e + 24, l->vol[i].vol_tag);
    xpar_buf_put(&body, l->vol[i].name, nl);
    xpar_buf_grow(&body, (sz) (xpar_align_up(body.len - at, XPAR_PKT_ALIGN) -
                               (body.len - at)));
  }
  xpar_pkt_write(out, XPAR_T_LAYT, pkt_flags(XPAR_PF_CRITICAL, key), set_id,
                 body.data, body.len, key);
  xpar_buf_free(&body);
}

typedef struct { u64 off, len; } layt_tile;

static bool tile_less(const layt_tile * a, const layt_tile * b) {
  return a->off < b->off || (a->off == b->off && a->len < b->len);
}

static void tile_down(layt_tile * a, u32 count, u32 root) {
  for (;;) {
    u32 child, pick;
    layt_tile t;
    if (root >= count / 2) return;
    child = root * 2 + 1;
    pick = child;
    if (child + 1 < count && tile_less(&a[child], &a[child + 1]))
      pick = child + 1;
    if (!tile_less(&a[root], &a[pick])) return;
    t = a[root];  a[root] = a[pick];  a[pick] = t;
    root = pick;
  }
}

static void tile_sort(layt_tile * a, u32 count) {
  u32 i;
  for (i = count / 2; i; i--) tile_down(a, count, i - 1);
  for (i = count; i > 1; i--) {
    layt_tile t = a[0];  a[0] = a[i - 1];  a[i - 1] = t;
    tile_down(a, i - 1, 0);
  }
}

xpar_status xpar_layt_tiles(const xpar_layt * l, u64 stream_length) {
  layt_tile * tile;
  u64 next = 0;
  u32 i, data = 0, at = 0;
  for (i = 0; i < l->count; i++)
    if (l->vol[i].kind == XPAR_VOL_DATA) data++;
  if (!data) return XPAR_OK;
  if (!stream_length) {
    for (i = 0; i < l->count; i++)
      if (l->vol[i].kind == XPAR_VOL_DATA &&
          (l->vol[i].stream_offset != 0 || l->vol[i].byte_length != 0))
        return XPAR_E_MALFORMED;
    return XPAR_OK;
  }
#if UINTPTR_MAX == UINT32_MAX
  if (data > (u32) ((sz) -1 / sizeof(*tile))) return XPAR_E_MALFORMED;
#endif
  tile = (layt_tile *) xpar_alloc_raw((sz) data * sizeof(*tile));
  for (i = 0; i < l->count; i++) if (l->vol[i].kind == XPAR_VOL_DATA) {
    tile[at].off = l->vol[i].stream_offset;
    tile[at].len = l->vol[i].byte_length;
    at++;
  }
  tile_sort(tile, data);
  for (i = 0; i < data; i++) {
    if (tile[i].off != next || !tile[i].len ||
        !add64(next, tile[i].len, &next)) {
      xpar_free(tile);
      return XPAR_E_MALFORMED;
    }
  }
  xpar_free(tile);
  return next == stream_length ? XPAR_OK : XPAR_E_MALFORMED;
}

void xpar_strm_write_header(xpar_buf * out, u64 stream_length,
                            const u8 * set_id, const xpar_key * key) {
  const sz fixed = XPAR_PKT_HDR + 16;
  xpar_blake3_t h;
  u8 * p;
  xpar_strm_write(out, 0, NULL, 0, set_id, key);
  p = out->data + out->len - fixed;
  xpar_wr64(p + 8, xpar_align_up(fixed + stream_length, XPAR_PKT_ALIGN));
  if (key) xpar_blake3_init_keyed(&h, key->k_pkt);
  else     xpar_blake3_init(&h);
  xpar_blake3_update(&h, p, 40);
  xpar_blake3_final(&h, p + 40, 8);
}

/*  Volume tags.  */

void xpar_vol_tag_begin(xpar_blake3_t * h) {
  xpar_blake3_init(h);
  xpar_blake3_update(h, "xpar2 volume tag v1", 19);
}

u64 xpar_vol_tag_final(xpar_blake3_t * h) {
  u8 out[8];
  xpar_blake3_final(h, out, sizeof out);
  return xpar_rd64(out);
}

bool xpar_vol_tag_match(const char * path, const xpar_vol * v) {
  xpar_file * f = xpar_open(path, XPAR_O_RDONLY);
  xpar_blake3_t h;
  u8 buf[65536];
  i64 size;
  u64 left = v->byte_length;
  if (!f) return false;
  size = xpar_size(f);
  if (size < 0 || (u64) size != v->byte_length) {
    xpar_close(f);  return false;
  }
  if (!v->vol_tag) { xpar_close(f);  return true; }
  /*  Sequentially, so a renamed multi-gigabyte volume needs no mapping.  */
  xpar_vol_tag_begin(&h);
  while (left) {
    sz take = (sz) MIN(left, (u64) sizeof buf);
    if (xpar_read(f, buf, take) != take) { xpar_close(f);  return false; }
    xpar_blake3_update(&h, buf, take);
    left -= take;
  }
  xpar_close(f);
  return xpar_vol_tag_final(&h) == v->vol_tag;
}

xpar_status xpar_auth_read(const u8 * body, sz n, xpar_auth * out) {
  xpar_memset(out, 0, sizeof *out);
  if (n != 32) return n < 32 ? XPAR_E_SHORT : XPAR_E_MALFORMED;
  out->kdf_id           = xpar_rd32(body);
  out->slice_tag_keyed  = body[4];
  out->packet_tag_keyed = body[5];
  out->unkeyed_retained = body[6];
  xpar_memcpy(out->key_check, body + 8, 16);
  if (out->slice_tag_keyed != 1 || out->packet_tag_keyed != 1 ||
      out->unkeyed_retained > 1)
    return XPAR_E_MALFORMED;
  if (body[7] || body[24] || body[25] || body[26] || body[27] ||
      body[28] || body[29] || body[30] || body[31])
    return XPAR_E_MALFORMED;
  if (out->kdf_id != 0) return XPAR_E_UNSUPPORTED;
  return XPAR_OK;
}

void xpar_auth_write(xpar_buf * out, const xpar_auth * a, const u8 * set_id,
                     const xpar_key * key) {
  u8 b[32];
  xpar_memset(b, 0, sizeof b);
  xpar_wr32(b, a->kdf_id);
  b[4] = a->slice_tag_keyed;  b[5] = a->packet_tag_keyed;
  b[6] = a->unkeyed_retained;
  xpar_memcpy(b + 8, a->key_check, 16);
  xpar_pkt_write(out, XPAR_T_AUTH, pkt_flags(XPAR_PF_CRITICAL, key), set_id,
                 b, sizeof b, key);
}

bool xpar_auth_key_ok(const xpar_auth * a, const u8 * master) {
  u8 want[16];
  xpar_key_check(want, master);
  return xpar_ct_equal(want, a->key_check, 16);
}

void xpar_text_write(xpar_buf * out, const char * type, const char * text,
                     const u8 * set_id, const xpar_key * key) {
  sz n = text ? xpar_strlen(text) : 0;
  xpar_pkt_write(out, type, pkt_flags(0, key), set_id, text, n, key);
}

void xpar_crtr_write(xpar_buf * out, const char * creator, const u8 * set_id,
                     const xpar_key * key, const xpar_wropt * o) {
  const char * s = (o && o->reproducible) ? "xpar" : creator;
  xpar_text_write(out, XPAR_T_CRTR, s, set_id, key);
}

u64 xpar_armg_length(u8 symbol_bits, u32 n, u32 k, u64 depth, u64 plain) {
  u64 w, frame, frames, out;
  if (symbol_bits != 8 && symbol_bits != 16) return 0;
  w = symbol_bits / 8;
  if (n < 16) return 0;
  if ((u64) n > (((u64) 1 << symbol_bits) - 1)) return 0;
  if (k == 0 || k >= n) return 0;
  if (((n - k) & 1) != 0) return 0;
  if (depth == 0 || depth > XPAR_ARMG_DEPTH_MAX) return 0;
  if (!mul64(depth, (u64) k * w, &frame)) return 0;
  frames = xpar_ceil_div(plain, frame);
  if (!mul64(frames, depth, &out)) return 0;
  if (!mul64(out, (u64) n * w, &out)) return 0;
  return out;
}

xpar_status xpar_armg_read(const u8 * body, sz n, xpar_armg * out) {
  u64 want, need;
  xpar_memset(out, 0, sizeof *out);
  if (n < 48) return XPAR_E_SHORT;
  /* Bytes 1-3 are reserved. */
  if (body[1] || xpar_rd16(body + 2)) return XPAR_E_MALFORMED;
  out->symbol_bits     = body[0];
  out->poly            = xpar_rd32(body + 4);
  out->n               = xpar_rd32(body + 8);
  out->k               = xpar_rd32(body + 12);
  out->fcr             = xpar_rd32(body + 16);
  out->prim            = xpar_rd32(body + 20);
  out->depth           = xpar_rd64(body + 24);
  out->plain_length    = xpar_rd64(body + 32);
  out->armoured_length = xpar_rd64(body + 40);

  want = xpar_armg_length(out->symbol_bits, out->n, out->k, out->depth,
                          out->plain_length);
  if (!want || want != out->armoured_length) return XPAR_E_MALFORMED;
  if (out->plain_length > out->armoured_length) return XPAR_E_MALFORMED;
  if (!add64(48, out->armoured_length, &need)) return XPAR_E_MALFORMED;
  if (need > (u64) n) return XPAR_E_MALFORMED;
  if (!pad_ok(body, need, n)) return XPAR_E_MALFORMED;
  out->data = body + 48;
  return XPAR_OK;
}

void xpar_armg_write(xpar_buf * out, const xpar_armg * a,
                     const void * armoured, const u8 * set_id,
                     const xpar_key * key) {
  u8 h[48];  xpar_part part[2];
  xpar_memset(h, 0, sizeof h);
  h[0] = a->symbol_bits;
  xpar_wr32(h + 4,  a->poly);
  xpar_wr32(h + 8,  a->n);
  xpar_wr32(h + 12, a->k);
  xpar_wr32(h + 16, a->fcr);
  xpar_wr32(h + 20, a->prim);
  xpar_wr64(h + 24, a->depth);
  xpar_wr64(h + 32, a->plain_length);
  xpar_wr64(h + 40, a->armoured_length);
  part[0].p = h;         part[0].n = sizeof h;
  part[1].p = armoured;  part[1].n = (sz) a->armoured_length;
  xpar_pkt_writev(out, XPAR_T_ARMG, pkt_flags(0, key), set_id, part, 2, key);
}

xpar_status xpar_strm_read(const u8 * body, sz n, xpar_strm * out) {
  xpar_memset(out, 0, sizeof *out);
  if (n < 16) return XPAR_E_SHORT;
  if (xpar_rd64(body + 8)) return XPAR_E_MALFORMED;
  out->stream_offset = xpar_rd64(body);
  out->data          = body + 16;
  out->length        = (u64) n - 16;
  return XPAR_OK;
}

void xpar_strm_write(xpar_buf * out, u64 stream_offset, const void * data,
                     sz len, const u8 * set_id, const xpar_key * key) {
  u8 h[16];  xpar_part part[2];
  xpar_memset(h, 0, sizeof h);
  xpar_wr64(h, stream_offset);
  part[0].p = h;     part[0].n = sizeof h;
  part[1].p = data;  part[1].n = len;
  xpar_pkt_writev(out, XPAR_T_STRM,
                  pkt_flags(XPAR_PF_BODY_UNCHECKED, key), set_id, part, 2,
                  key);
}

void xpar_crit_write(xpar_buf * out, const xpar_crit * c, const u8 * set_id,
                     const xpar_key * key, const xpar_wropt * o) {
  u32 i;
  bool posix = c->posix_count != 0;
  if (o && o->reproducible && !o->keep_posix) posix = false;

  xpar_setd_write(out, c->setd, set_id, key);
  for (i = 0; i < c->file_count; i++)
    xpar_entry_write(out, &c->file[i], set_id, key, o);
  if (posix)
    xpar_posx_write_all(out, c->posix, c->posix_count, set_id, key);
  if (c->slice_count && c->slice_crc)
    xpar_slcr_write_all(out, c->slice_crc, c->slice_count, set_id, key);
  if (c->auth) xpar_auth_write(out, c->auth, set_id, key);
  if (c->layt) xpar_layt_write(out, c->layt, set_id, key);
}

bool xpar_replicate_here(u64 crit, u64 payload, u32 i, u32 count) {
  u64 thr = payload / 20;
  if (thr < ((u64) 1 << 20)) thr = (u64) 1 << 20;
  if (crit <= thr) return true;
  if (i == 0) return true;
  if (count && i == count - 1) return true;
  return xpar_is_pow2(i);
}

/*  splitmix64's finaliser (Steele, Lea and Flood, 2014). Used only to
    spread the identity over the index table; nothing depends on its
    output being stable across builds.  */
static u64 mix64(u64 x) {
  x ^= x >> 30;  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;  x *= 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static u32 type_key(const char * t) {
  return (u32) ((u8) t[0]) | ((u32) (u8) t[1] << 8) |
         ((u32) (u8) t[2] << 16) | ((u32) (u8) t[3] << 24);
}

static u64 file_disc(const u8 * body) {
  return mix64(xpar_rd64(body) ^ mix64(xpar_rd64(body + 8)));
}

/*  What distinguishes two packets of one type within one generation. Read
    only after the body has been proved long enough to hold it, so a short
    body simply discriminates as zero rather than reading past the end.  */
static u64 crit_disc(const char * t, const u8 * body, u64 n) {
  if (!xpar_memcmp(t, XPAR_T_FILE, 4) && n >= 16) return file_disc(body);
  if (!xpar_memcmp(t, XPAR_T_SLCR, 4) && n >= 8) return xpar_rd64(body);
  if (!xpar_memcmp(t, XPAR_T_SLTG, 4) && n >= 8) return xpar_rd64(body);
  if (!xpar_memcmp(t, XPAR_T_SLCL, 4) && n >= 8) return xpar_rd64(body);
  if (!xpar_memcmp(t, XPAR_T_RCVS, 4) && n >= 8) return xpar_rd64(body);
  if (!xpar_memcmp(t, XPAR_T_POSX, 4) && n >= 4) return xpar_rd32(body);
  return 0;
}

static u64 crit_hash(const u8 * set_id, const char * type, u64 disc) {
  return mix64(mix64(xpar_rd64(set_id) ^ disc) ^
               ((u64) type_key(type) << 32) ^ xpar_rd64(set_id + 8));
}

void xpar_critset_init(xpar_critset * s) { xpar_memset(s, 0, sizeof *s); }

void xpar_critset_free(xpar_critset * s) {
  xpar_free(s->pkt);  xpar_free(s->idx);
  xpar_memset(s, 0, sizeof *s);
}

static bool crit_same(const xpar_critset * s, u32 slot, const u8 * set_id,
                      const char * type, u64 disc) {
  const xpar_crit_pkt * e = &s->pkt[slot];
  if (xpar_memcmp(e->hdr.set_id, set_id, XPAR_SET_ID_LEN) != 0 ||
      !xpar_pkt_is(&e->hdr, type) ||
      crit_disc(e->hdr.type, e->body, e->body_len) != disc) return false;
  return true;
}

static void crit_grow(xpar_critset * s) {
  u32 want = s->count + 1, cap, i;
  if (want <= s->cap && s->idx && (u64) s->count * 2 <= s->mask) return;
  cap = s->cap ? s->cap * 2 : 16;
  while (cap < want) cap *= 2;
  /*  The index holds four slots per packet, so the capacity has to stay
      below 2^30 for cap * 4 to be a u32. A volume large enough to reach
      that would need 2^28 packets and twelve gigabytes.  */
  xpar_assert(cap <= (u32) 1 << 28);
  s->pkt = (xpar_crit_pkt *) xpar_realloc(s->pkt,
                                          (sz) cap * sizeof(xpar_crit_pkt));
  s->cap = cap;
  /*  The index is rebuilt rather than rehashed incrementally: it holds
      slot numbers only, so a rebuild is a linear pass over packets that
      are already in memory.  */
  xpar_free(s->idx);
  s->mask = cap * 4 - 1;
  s->idx  = (u32 *) xpar_calloc((sz) s->mask + 1, sizeof(u32));
  for (i = 0; i < s->count; i++) {
    u64 h = crit_hash(s->pkt[i].hdr.set_id, s->pkt[i].hdr.type,
                      crit_disc(s->pkt[i].hdr.type, s->pkt[i].body,
                                s->pkt[i].body_len));
    u32 j = (u32) (h & s->mask);
    while (s->idx[j]) j = (j + 1) & s->mask;
    s->idx[j] = i + 1;
  }
}

bool xpar_critset_add(xpar_critset * s, const xpar_pkt * hdr,
                      const u8 * body) {
  u64 n = hdr->length - XPAR_PKT_HDR, h;
  u64 disc = crit_disc(hdr->type, body, n);
  u32 j;
  crit_grow(s);
  h = crit_hash(hdr->set_id, hdr->type, disc);
  j = (u32) (h & s->mask);
  while (s->idx[j]) {
    u32 slot = s->idx[j] - 1;
    if (crit_same(s, slot, hdr->set_id, hdr->type, disc)) {
      if (xpar_pkt_is(hdr, XPAR_T_FILE) && n >= XPAR_SET_ID_LEN &&
          s->pkt[slot].body_len >= XPAR_SET_ID_LEN &&
          xpar_memcmp(s->pkt[slot].body, body, XPAR_SET_ID_LEN) != 0) {
        j = (j + 1) & s->mask;
        continue;
      }
      s->pkt[slot].copies++;  s->copies++;
      /*  CRTR is per-volume provenance and may differ across copies.  */
      if (xpar_pkt_is(hdr, XPAR_T_CRTR)) return false;
      if (s->pkt[slot].body_len != n ||
          xpar_memcmp(s->pkt[slot].body, body, (sz) n) != 0) {
        s->pkt[slot].conflicts++;  s->conflicts++;
      }
      return false;
    }
    j = (j + 1) & s->mask;
  }
  s->idx[j] = s->count + 1;
  s->pkt[s->count].hdr       = *hdr;
  s->pkt[s->count].body      = body;
  s->pkt[s->count].body_len  = n;
  s->pkt[s->count].copies    = 1;
  s->pkt[s->count].conflicts = 0;
  s->count++;  s->copies++;
  return true;
}

const xpar_crit_pkt * xpar_critset_find(const xpar_critset * s,
                                        const u8 * set_id, const char * type,
                                        u64 disc) {
  u64 h;  u32 j;
  if (!s->idx) return NULL;
  h = crit_hash(set_id, type, disc);
  j = (u32) (h & s->mask);
  while (s->idx[j]) {
    u32 slot = s->idx[j] - 1;
    if (crit_same(s, slot, set_id, type, disc)) return &s->pkt[slot];
    j = (j + 1) & s->mask;
  }
  return NULL;
}

const xpar_crit_pkt * xpar_critset_find_file(const xpar_critset * s,
                                             const u8 * set_id,
                                             const u8 * file_id) {
  u64 disc = file_disc(file_id);
  u64 h;
  u32 j;
  if (!s->idx) return NULL;
  h = crit_hash(set_id, XPAR_T_FILE, disc);
  j = (u32) (h & s->mask);
  while (s->idx[j]) {
    u32 slot = s->idx[j] - 1;
    const xpar_crit_pkt * p = &s->pkt[slot];
    if (crit_same(s, slot, set_id, XPAR_T_FILE, disc) &&
        p->body_len >= XPAR_SET_ID_LEN &&
        !xpar_memcmp(p->body, file_id, XPAR_SET_ID_LEN)) return p;
    j = (j + 1) & s->mask;
  }
  return NULL;
}

void xpar_posix_records_free(xpar_posix_rec * rec, u32 count) {
  For(u32, i, count, xpar_posix_rec_free(&rec[i]))
  xpar_free(rec);
}

/*  Minimum encoded POSX record size.  */
#define POSX_REC_MIN 16

xpar_status xpar_posx_collect(const xpar_critset * c, const u8 * set_id,
                              u32 count, xpar_posix_rec ** out) {
  xpar_posix_rec * rec;
  u8 * seen;
  u32 i, covered = 0;
  u64 have = 0;
  *out = NULL;
  /*  Reject counts that cannot fit in the available POSX data.  */
  for (i = 0; i < c->count; i++) {
    const xpar_crit_pkt * p = &c->pkt[i];
    if (!xpar_pkt_is(&p->hdr, XPAR_T_POSX) ||
        xpar_memcmp(p->hdr.set_id, set_id, XPAR_SET_ID_LEN)) continue;
    have += p->body_len;
  }
  if ((u64) count * POSX_REC_MIN > have) return XPAR_E_MALFORMED;
  rec = (xpar_posix_rec *) xpar_calloc(count ? count : 1, sizeof *rec);
  seen = (u8 *) xpar_calloc(count ? count : 1, 1);
  for (i = 0; i < count; i++) rec[i].uid = rec[i].gid = UINT32_MAX;
  for (i = 0; i < c->count; i++) {
    const xpar_crit_pkt * p = &c->pkt[i];
    xpar_posx t;
    u32 j;
    if (!xpar_pkt_is(&p->hdr, XPAR_T_POSX) ||
        xpar_memcmp(p->hdr.set_id, set_id, XPAR_SET_ID_LEN)) continue;
    if (xpar_posx_read(p->body, (sz) p->body_len, &t) != XPAR_OK)
      goto malformed;
    if (t.first_record >= count || t.count > count - t.first_record) {
      xpar_posx_free(&t);
      goto malformed;
    }
    for (j = 0; j < t.count; j++) {
      u32 at = t.first_record + j;
      if (seen[at]) { xpar_posx_free(&t);  goto malformed; }
      seen[at] = 1;  covered++;
      rec[at] = t.rec[j];
      xpar_memset(&t.rec[j], 0, sizeof t.rec[j]);
    }
    xpar_posx_free(&t);
  }
  xpar_free(seen);
  if (covered != count) {
    xpar_posix_records_free(rec, count);
    return XPAR_E_MALFORMED;
  }
  if (!count) {
    xpar_free(rec);
    rec = NULL;
  }
  *out = rec;
  return XPAR_OK;

malformed:
  xpar_free(seen);
  xpar_posix_records_free(rec, count);
  return XPAR_E_MALFORMED;
}
