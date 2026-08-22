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

/*  Cauchy-matrix outer erasure codec and codec dispatcher.  */

#include "codec.h"
#include "codec-int.h"
#include "gf.h"

/*  One coefficient tile is reused across each byte block. The live set is
    (MAT_SRC + MAT_DST) * MAT_BLK bytes plus prepared coefficients.  */

#define MAT_SRC 16
#define MAT_DST 16
#define MAT_BLK 8192

typedef union {
  xpar_gf8_coef  c8 [MAT_SRC * MAT_DST];
  xpar_gf16_coef c16[MAT_SRC * MAT_DST];
} mat_tile;

typedef struct {
  u8  f16;         /*  0 for GF(2^8), 1 for GF(2^16).  */
  u32 s, r;
} mat_codec;

typedef struct {
  const mat_codec * cd;
  u32   e, nkeep;
  u32 * lost;      /*  The e erased data indices, ascending.  */
  u32 * use;       /*  The e recovery rows the inverse was built on.  */
  u32 * keep;      /*  The surviving data indices, ascending.  */
  void * inv;      /*  e by e, row major, u8 or u16.  */
} mat_plan;

struct xpar_codec      { u8 kind;  void * impl; };
struct xpar_codec_plan { u8 kind;  void * impl; };

static sz mat_stride(sz bytes) { return (sz) xpar_align_up(bytes, 64); }

/*  For A[k][l] = 1/(x_k + y_l), the inverse entry at [l][k] is

      B[l][k] = P(x_k) Q(y_l) / ((x_k + y_l) a_k b_l)

    with P(x_k) = prod_t (x_k + y_t),  a_k = prod_{t != k} (x_k + x_t),
         Q(y_l) = prod_t (x_t + y_l),  b_l = prod_{t != l} (y_l + y_t).

    In characteristic two subtraction is XOR and the sign is one. Node
    distinctness keeps every denominator nonzero. Exponents are reduced
    modulo 2^w - 1, so inversion is negation and one conditional subtraction
    reduces each accumulated exponent.  */

static u32 addm(u32 a, u32 b, u32 mod) {
  u32 s = a + b;
  return s >= mod ? s - mod : s;
}

static void cauchy_inv8(u32 n, const u8 * x, const u8 * y, u8 * b) {
  u32 * rk = (u32 *) xpar_alloc_raw((sz) 2 * n * sizeof(u32));
  u32 * cl = rk + n;
  for (u32 k = 0; k < n; k++) {
    u32 a = 0;
    for (u32 t = 0; t < n; t++) a = addm(a, xpar_gf8_log[x[k] ^ y[t]], 255);
    for (u32 t = 0; t < n; t++)
      if (t != k) a = addm(a, 255u - xpar_gf8_log[x[k] ^ x[t]], 255);
    rk[k] = a;
  }
  for (u32 l = 0; l < n; l++) {
    u32 a = 0;
    for (u32 t = 0; t < n; t++) a = addm(a, xpar_gf8_log[x[t] ^ y[l]], 255);
    for (u32 t = 0; t < n; t++)
      if (t != l) a = addm(a, 255u - xpar_gf8_log[y[l] ^ y[t]], 255);
    cl[l] = a;
  }
  for (u32 l = 0; l < n; l++)
    for (u32 k = 0; k < n; k++) {
      u32 t = addm(rk[k], cl[l], 255);
      t = addm(t, 255u - xpar_gf8_log[x[k] ^ y[l]], 255);
      b[(sz) l * n + k] = xpar_gf8_exp[t];
    }
  xpar_free(rk);
}

static void cauchy_inv16(u32 n, const u16 * x, const u16 * y, u16 * b) {
  u32 * rk = (u32 *) xpar_alloc_raw((sz) 2 * n * sizeof(u32));
  u32 * cl = rk + n;
  for (u32 k = 0; k < n; k++) {
    u32 a = 0;
    for (u32 t = 0; t < n; t++)
      a = addm(a, xpar_gf16_log[x[k] ^ y[t]], 65535);
    for (u32 t = 0; t < n; t++)
      if (t != k) a = addm(a, 65535u - xpar_gf16_log[x[k] ^ x[t]], 65535);
    rk[k] = a;
  }
  for (u32 l = 0; l < n; l++) {
    u32 a = 0;
    for (u32 t = 0; t < n; t++)
      a = addm(a, xpar_gf16_log[x[t] ^ y[l]], 65535);
    for (u32 t = 0; t < n; t++)
      if (t != l) a = addm(a, 65535u - xpar_gf16_log[y[l] ^ y[t]], 65535);
    cl[l] = a;
  }
  for (u32 l = 0; l < n; l++)
    for (u32 k = 0; k < n; k++) {
      u32 t = addm(rk[k], cl[l], 65535);
      t = addm(t, 65535u - xpar_gf16_log[x[k] ^ y[l]], 65535);
      b[(sz) l * n + k] = xpar_gf16_exp[t];
    }
  xpar_free(rk);
}

/*  One descriptor serves generated and stored coefficient matrices.  */

typedef struct {
  u8  f16;
  u32 base;             /*  Highest recovery node: x = base - row.  */
  u32 colbase;          /*  First contiguous source column.  */
  const u32 * rowmap;   /*  Destination slot to matrix row, or NULL.  */
  const u32 * colmap;   /*  Source slot to matrix column, or NULL.  */
  const void * mat;     /*  When set, entries are read from here.  */
  u32 stride;
} mat_coefs;

static u32 mat_entry(const mat_coefs * cf, u32 i, u32 j) {
  if (cf->mat)
    return cf->f16 ? (u32) ((const u16 *) cf->mat)[(sz) j * cf->stride + i]
                   : (u32) ((const u8  *) cf->mat)[(sz) j * cf->stride + i];
  {
    u32 row = cf->rowmap ? cf->rowmap[j] : j;
    u32 col = cf->colbase + (cf->colmap ? cf->colmap[i] : i);
    /*  Recovery nodes descend from the field top while data nodes ascend;
        rows therefore remain stable when EOF reveals S.  */
    u32 d = (cf->base - row) ^ col;
    return cf->f16 ? (u32) xpar_gf16_inv((u16) d)
                   : (u32) xpar_gf8_inv((u8) d);
  }
}

static void mat_run(const mat_coefs * cf, u8 * const * dst, u32 nd,
                    const u8 * const * src, u32 ns, sz bytes,
                    mat_tile * tl) {
  const xpar_gf_kernels * gk = xpar_gf_active();
  /*  Paired destinations help shuffle tiers but slow affine and VBMI.  */
  bool macx2 = xpar_strcmp(gk->name, "gfni256") &&
               xpar_strcmp(gk->name, "gfni512") &&
               xpar_strcmp(gk->name, "vbmi512") &&
               xpar_strcmp(gk->name, "scalar");
  bool mac16x2 = !xpar_strcmp(gk->name, "gfni256");
  for (u32 j0 = 0; j0 < nd; j0 += MAT_DST) {
    u32 jn = MIN((u32) MAT_DST, nd - j0);
    for (u32 i0 = 0; i0 < ns; i0 += MAT_SRC) {
      u32 in = MIN((u32) MAT_SRC, ns - i0);
      for (u32 i = 0; i < in; i++)
        for (u32 j = 0; j < jn; j++) {
          u32 c = mat_entry(cf, i0 + i, j0 + j);
          if (cf->f16) xpar_gf16_prepare(&tl->c16[i * MAT_DST + j], (u16) c);
          else         xpar_gf8_prepare (&tl->c8 [i * MAT_DST + j], (u8)  c);
        }
      for (sz off = 0; off < bytes; off += MAT_BLK) {
        sz len = MIN((sz) MAT_BLK, bytes - off);
        for (u32 i = 0; i < in; i++) {
          const u8 * sp = src[i0 + i] + off;
          u32 j = 0;
          if (cf->f16 && mac16x2)
            for (; j + 2 <= jn; j += 2) {
              u8 * d[2];
              for (u32 k = 0; k < 2; k++) d[k] = dst[j0 + j + k] + off;
              gk->mac16x2(d, sp, len, &tl->c16[i * MAT_DST + j]);
            }
          else if (!cf->f16 && macx2)
            for (; j + 2 <= jn; j += 2) {
              u8 * d[2];
              for (u32 k = 0; k < 2; k++) d[k] = dst[j0 + j + k] + off;
              gk->mac8x2(d, sp, len, &tl->c8[i * MAT_DST + j]);
            }
          for (; j < jn; j++)
            if (cf->f16)
              gk->mac16(dst[j0 + j] + off, sp, len, &tl->c16[i * MAT_DST + j]);
            else
              gk->mac8 (dst[j0 + j] + off, sp, len, &tl->c8 [i * MAT_DST + j]);
        }
      }
    }
  }
}

static bool mat_supports(u8 field_log2, u64 s, u64 r) {
  if (field_log2 != 8 && field_log2 != 16) return false;
  if (s < 1 || r < 1) return false;
  return s + r <= ((u64) 1 << field_log2);
}

static void * mat_new(u8 field_log2, u64 s, u64 r) {
  xpar_gf_init();
  mat_codec * cd = (mat_codec *) xpar_calloc(1, sizeof(mat_codec));
  cd->f16 = field_log2 == 16;
  cd->s = (u32) s;  cd->r = (u32) r;
  return cd;
}

static void mat_free(void * self) { xpar_free(self); }

static xpar_codec_status mat_encode(void * self, const u8 * const * data,
                                    u8 * const * rec, sz bytes) {
  mat_codec * cd = (mat_codec *) self;
  mat_coefs cf;  mat_tile * tl;
  xpar_assert(!cd->f16 || (bytes & 1) == 0);
  if (!bytes) return XPAR_CODEC_OK;
  tl = (mat_tile *) xpar_alloc_raw(sizeof(mat_tile));
  for (u32 j = 0; j < cd->r; j++) xpar_memset(rec[j], 0, bytes);
  xpar_memset(&cf, 0, sizeof cf);
  cf.f16 = cd->f16;  cf.base = cd->f16 ? 65535u : 255u;
  mat_run(&cf, rec, cd->r, data, cd->s, bytes, tl);
  xpar_free(tl);
  return XPAR_CODEC_OK;
}

/*  Choose the lowest E present recovery rows so plans depend only on the
    erasure pattern and remain cacheable.  */
static void * mat_plan_new(void * self, const u8 * dpres, const u8 * rpres,
                           xpar_codec_status * status) {
  mat_codec * cd = (mat_codec *) self;
  u32 s = cd->s, r = cd->r, e = 0, got = 0;
  for (u32 i = 0; i < s; i++) if (!dpres[i]) e++;
  for (u32 i = 0; i < r; i++) if (rpres[i]) got++;
  if (got < e) { *status = XPAR_CODEC_TOO_MANY_LOST;  return NULL; }
  *status = XPAR_CODEC_OK;
  mat_plan * pl = (mat_plan *) xpar_calloc(1, sizeof(mat_plan));
  pl->cd = cd;  pl->e = e;  pl->nkeep = s - e;
  if (!e) return pl;
  pl->lost = (u32 *) xpar_alloc_raw((sz) e * sizeof(u32));
  pl->use  = (u32 *) xpar_alloc_raw((sz) e * sizeof(u32));
  pl->keep = (u32 *) xpar_alloc_raw((sz) (pl->nkeep ? pl->nkeep : 1) *
                                    sizeof(u32));
  for (u32 i = 0, k = 0, t = 0; i < s; i++) {
    if (dpres[i]) pl->keep[t++] = i;  else pl->lost[k++] = i;
  }
  for (u32 i = 0, k = 0; i < r && k < e; i++) if (rpres[i]) pl->use[k++] = i;
  if (cd->f16) {
    u16 * x = (u16 *) xpar_alloc_raw((sz) 2 * e * sizeof(u16));
    u16 * y = x + e;
    for (u32 k = 0; k < e; k++) x[k] = (u16) (65535u - pl->use[k]);
    for (u32 l = 0; l < e; l++) y[l] = (u16) pl->lost[l];
    pl->inv = xpar_alloc_raw((sz) e * e * sizeof(u16));
    cauchy_inv16(e, x, y, (u16 *) pl->inv);
    xpar_free(x);
  } else {
    u8 * x = (u8 *) xpar_alloc_raw((sz) 2 * e);
    u8 * y = x + e;
    for (u32 k = 0; k < e; k++) x[k] = (u8) (255u - pl->use[k]);
    for (u32 l = 0; l < e; l++) y[l] = (u8) pl->lost[l];
    pl->inv = xpar_alloc_raw((sz) e * e);
    cauchy_inv8(e, x, y, (u8 *) pl->inv);
    xpar_free(x);
  }
  return pl;
}

static void mat_plan_free(void * self) {
  mat_plan * pl = (mat_plan *) self;
  if (!pl) return;
  xpar_free(pl->lost);  xpar_free(pl->use);
  xpar_free(pl->keep);  xpar_free(pl->inv);
  xpar_free(pl);
}

/*  Remove surviving data from each selected recovery row, then apply the
    inverse to the remaining erased-slice syndromes. Scratch is per call so
    one plan may be applied concurrently.  */
static xpar_codec_status mat_plan_apply(const void * self,
                                        u8 * const * data,
                                        u8 * const * rec, sz bytes) {
  const mat_plan * pl = (const mat_plan *) self;
  const mat_codec * cd = pl->cd;
  u32 e = pl->e;
  xpar_assert(!cd->f16 || (bytes & 1) == 0);
  if (!e || !bytes) return XPAR_CODEC_OK;
  sz stride = mat_stride(bytes);
  u64 need = (u64) e * (u64) stride;
  FATAL_UNLESS("Column chunk too wide for this host's address space.",
               need <= (u64) (sz) -1);
  u8 * pool = (u8 *) xpar_alloc_aligned((sz) need, 64);
  u8 ** syn = (u8 **) xpar_alloc_raw((sz) e * sizeof(u8 *));
  u8 ** out = (u8 **) xpar_alloc_raw((sz) e * sizeof(u8 *));
  const u8 ** in = (const u8 **) xpar_alloc_raw(
                     (sz) (pl->nkeep ? pl->nkeep : 1) * sizeof(u8 *));
  mat_tile * tl = (mat_tile *) xpar_alloc_raw(sizeof(mat_tile));
  mat_coefs cf;
  for (u32 k = 0; k < e; k++) {
    syn[k] = pool + (sz) k * stride;
    xpar_memcpy(syn[k], rec[pl->use[k]], bytes);
  }
  for (u32 i = 0; i < pl->nkeep; i++) in[i] = data[pl->keep[i]];
  xpar_memset(&cf, 0, sizeof cf);
  cf.f16 = cd->f16;  cf.base = cd->f16 ? 65535u : 255u;
  cf.rowmap = pl->use;
  cf.colmap = pl->keep;
  mat_run(&cf, syn, e, in, pl->nkeep, bytes, tl);
  for (u32 l = 0; l < e; l++) {
    out[l] = data[pl->lost[l]];
    xpar_memset(out[l], 0, bytes);
  }
  cf.rowmap = NULL;  cf.colmap = NULL;
  cf.mat = pl->inv;  cf.stride = e;
  mat_run(&cf, out, e, (const u8 * const *) syn, e, bytes, tl);
  xpar_free(tl);  xpar_free((void *) in);
  xpar_free(out); xpar_free(syn);  xpar_free_aligned(pool);
  return XPAR_CODEC_OK;
}

static bool codec_is_fft(u8 kind) {
  return kind == XPAR_CODEC_FFT || kind == XPAR_CODEC_FFT_LOW;
}

bool xpar_codec_supports(u8 codec, u8 field_log2, u64 s, u64 r) {
  u64 m = xpar_next_pow2(codec == XPAR_CODEC_FFT_LOW ? s : r);
  u8 axis = codec == XPAR_CODEC_MATRIX ? field_log2
                                       : (u8) xpar_log2_floor(m);
  return xpar_codec_supports_axis(codec, field_log2, s, r, axis);
}

bool xpar_codec_supports_axis(u8 codec, u8 field_log2, u64 s, u64 r,
                              u8 axis_log2) {
  if (codec_is_fft(codec))
    return xpar_fft_supports_axis(codec, field_log2, s, r, axis_log2);
  if (codec == XPAR_CODEC_MATRIX)
    return axis_log2 == field_log2 && mat_supports(field_log2, s, r);
  return false;
}

xpar_codec * xpar_codec_new(u8 codec, u8 field_log2, u64 s, u64 r) {
  u64 m = xpar_next_pow2(codec == XPAR_CODEC_FFT_LOW ? s : r);
  u8 axis = codec == XPAR_CODEC_MATRIX ? field_log2
                                       : (u8) xpar_log2_floor(m);
  return xpar_codec_new_axis(codec, field_log2, s, r, axis);
}

xpar_codec * xpar_codec_new_axis(u8 codec, u8 field_log2, u64 s, u64 r,
                                 u8 axis_log2) {
  FATAL_UNLESS("internal: unsupported codec geometry.",
               xpar_codec_supports_axis(codec, field_log2, s, r, axis_log2));
  xpar_codec * c = (xpar_codec *) xpar_calloc(1, sizeof(xpar_codec));
  c->kind = codec;
  c->impl = codec_is_fft(codec)
              ? xpar_fft_new_axis(codec, field_log2, s, r, axis_log2)
              : mat_new(field_log2, s, r);
  return c;
}

void xpar_codec_free(xpar_codec * c) {
  if (!c) return;
  if (codec_is_fft(c->kind)) xpar_fft_free(c->impl);
  else                           mat_free(c->impl);
  xpar_free(c);
}

xpar_codec_status xpar_codec_encode(xpar_codec * c, const u8 * const * data,
                                    u8 * const * recovery, sz bytes) {
  return codec_is_fft(c->kind)
           ? xpar_fft_encode(c->impl, data, recovery, bytes)
           : mat_encode(c->impl, data, recovery, bytes);
}

xpar_codec_status xpar_codec_matrix_accumulate_many(
  xpar_codec * c, u64 data_first, const u8 * const * data, u64 data_count,
  u64 recovery_first, u8 * const * recovery, u64 recovery_count,
  sz bytes, bool clear) {
  mat_codec * cd;
  mat_coefs cf;
  mat_tile * tl;
  u64 j;
  if (c->kind != XPAR_CODEC_MATRIX) return XPAR_CODEC_UNSUPPORTED;
  cd = (mat_codec *) c->impl;
  if (data_first > cd->s || data_count > cd->s - data_first ||
      recovery_first > cd->r ||
      recovery_count > cd->r - recovery_first)
    return XPAR_CODEC_UNSUPPORTED;
  if (!bytes || !data_count || !recovery_count) return XPAR_CODEC_OK;
  if (clear)
    for (j = 0; j < recovery_count; j++) xpar_memset(recovery[j], 0, bytes);
  xpar_memset(&cf, 0, sizeof cf);
  cf.f16 = cd->f16;
  cf.base = (cd->f16 ? 65535u : 255u) - (u32) recovery_first;
  cf.colbase = (u32) data_first;
  tl = (mat_tile *) xpar_alloc_raw(sizeof(mat_tile));
  mat_run(&cf, recovery, (u32) recovery_count, data, (u32) data_count,
          bytes, tl);
  xpar_free(tl);
  return XPAR_CODEC_OK;
}

xpar_codec_status xpar_codec_matrix_accumulate(xpar_codec * c, u64 data_index,
                                               const u8 * data,
                                               u64 recovery_first,
                                               u8 * const * recovery,
                                               u64 recovery_count, sz bytes,
                                               bool clear) {
  const u8 * source[1];
  source[0] = data;
  return xpar_codec_matrix_accumulate_many(
           c, data_index, source, 1, recovery_first, recovery,
           recovery_count, bytes, clear);
}

xpar_codec_plan * xpar_codec_plan_new(xpar_codec * c,
                                      const u8 * data_present,
                                      const u8 * recovery_present,
                                      xpar_codec_status * status) {
  void * impl = codec_is_fft(c->kind)
                  ? xpar_fft_plan_new(c->impl, data_present,
                                      recovery_present, status)
                  : mat_plan_new(c->impl, data_present, recovery_present,
                                 status);
  if (!impl) return NULL;
  xpar_codec_plan * p = (xpar_codec_plan *)
                          xpar_calloc(1, sizeof(xpar_codec_plan));
  p->kind = c->kind;  p->impl = impl;
  return p;
}

void xpar_codec_plan_free(xpar_codec_plan * p) {
  if (!p) return;
  if (codec_is_fft(p->kind)) xpar_fft_plan_free(p->impl);
  else                           mat_plan_free(p->impl);
  xpar_free(p);
}

xpar_codec_status xpar_codec_plan_apply(const xpar_codec_plan * p,
                                        u8 * const * data,
                                        u8 * const * recovery, sz bytes) {
  return codec_is_fft(p->kind)
           ? xpar_fft_plan_apply(p->impl, data, recovery, bytes)
           : mat_plan_apply(p->impl, data, recovery, bytes);
}

/*  Footprints include caller-resident buffers: matrix encode holds R
    accumulators and one input column.  */

static u64 mat_elem(u8 field_log2) { return field_log2 == 8 ? 1 : 2; }

u64 xpar_codec_encode_footprint(u8 codec, u8 field_log2, u64 s, u64 r,
                                sz bytes) {
  u64 m = xpar_next_pow2(codec == XPAR_CODEC_FFT_LOW ? s : r);
  u8 axis = codec == XPAR_CODEC_MATRIX ? field_log2
                                       : (u8) xpar_log2_floor(m);
  return xpar_codec_encode_footprint_axis(codec, field_log2, s, r, axis,
                                           bytes);
}

u64 xpar_codec_encode_footprint_axis(u8 codec, u8 field_log2, u64 s, u64 r,
                                     u8 axis_log2, sz bytes) {
  if (codec_is_fft(codec))
    return xpar_fft_encode_footprint_axis(codec, field_log2, s, r,
                                          axis_log2, bytes);
  return (r + 1) * (u64) mat_stride(bytes) + sizeof(mat_tile) + 256;
}

u64 xpar_codec_decode_footprint(u8 codec, u8 field_log2, u64 s, u64 r,
                                sz bytes) {
  u64 m = xpar_next_pow2(codec == XPAR_CODEC_FFT_LOW ? s : r);
  u8 axis = codec == XPAR_CODEC_MATRIX ? field_log2
                                       : (u8) xpar_log2_floor(m);
  return xpar_codec_decode_footprint_axis(codec, field_log2, s, r, axis,
                                           bytes);
}

u64 xpar_codec_decode_footprint_axis(u8 codec, u8 field_log2, u64 s, u64 r,
                                     u8 axis_log2, sz bytes) {
  if (codec_is_fft(codec))
    return xpar_fft_decode_footprint_axis(codec, field_log2, s, r,
                                          axis_log2, bytes);
  /*  Worst case E = R, including caller and plan-application scratch.  */
  return (2 * r + 1) * (u64) mat_stride(bytes) + r * r * mat_elem(field_log2)
         + (2 * r + s) * 12 + sizeof(mat_tile) + 256;
}

u64 xpar_codec_encode_work(u8 codec, u64 s, u64 r, sz bytes) {
  if (codec_is_fft(codec)) return xpar_fft_encode_work(codec, s, r, bytes);
  return s * r * (u64) bytes;
}
