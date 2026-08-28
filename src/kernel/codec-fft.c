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

/*  Additive-FFT erasure coding over GF(2^8) and GF(2^16).

    Encoding uses m = NextPow2(R); decoding uses
    n = NextPow2(m + S) and has cost independent of the erasure count.
    FFT sets require R <= S and S + m <= 2^w.  They are prefix-stable
    exactly while m remains unchanged.

    Values retain gf.h's polynomial representation.  GF(2^8)
    coefficients are prepared eagerly; GF(2^16) coefficients are
    prepared at each butterfly group to avoid a full-field table.  */

#include "codec.h"
#include "codec-int.h"
#include "gf.h"

/*  Plan construction lazily builds the decoder's Walsh table and is not
    safe to call concurrently on one codec.  Apply operations have local
    scratch and are reentrant.  */

typedef struct {
  u8   f16, low;           /*  Field width and transform orientation.  */
  u32  bits, order, mod;   /*  w, 2^w, 2^w - 1.  */
  u32  s, r, m, n;         /*  m rounds the smaller transform axis.  */
  void * skew;             /*  `mod` field values, indexed from zero.  */
  xpar_gf8_coef * prep8;   /*  256 entries in GF(2^8), NULL otherwise.  */
  u16 * walsh;             /*  FWHT of the log table; NULL until decode.  */
} fft_codec;

typedef struct {
  const fft_codec * cd;
  u32   e;                 /*  Erased data slices; zero is a valid plan.  */
  u32 * lost;              /*  Their indices, ascending.  */
  u8 *  dpres, * rpres;    /*  Copies of the caller's two flag arrays.  */
  u16 * elog;              /*  Exponents over the active transform span.  */
} fft_plan;

static u32 skew_at(const fft_codec * cd, u32 i) {
  return cd->f16 ? (u32) ((const u16 *) cd->skew)[i]
                 : (u32) ((const u8  *) cd->skew)[i];
}

static u32 fld_exp(const fft_codec * cd, u32 e) {
  return cd->f16 ? (u32) xpar_gf16_exp[e] : (u32) xpar_gf8_exp[e];
}

static void fld_mul(u8 f16, u8 * d, const u8 * s, sz n, u32 c) {
  if (f16) xpar_gf16_mul_region(d, s, n, (u16) c);
  else     xpar_gf8_mul_region (d, s, n, (u8)  c);
}

/*  Butterflies are independent along the byte axis.  Strips bound the
    live working set and scratch allocation while retaining useful kernel
    widths.  */

#define FFT_LIVE (8u << 20)

static sz fft_strip(u32 count, sz bytes) {
  u64 t = (u64) FFT_LIVE / (count ? count : 1u);
  t &= ~(u64) 63;
  if (t < 64) t = 64;
  return (sz) ((u64) bytes < t ? (u64) bytes : t);
}

/*  Prepare one coefficient per butterfly group and strip.  */

/*  Buffers are strided to 64 bytes and come from one allocation. A
    vector kernel that straddles a cache line pays for it on every
    buffer, and n separate blocks at n = 8192 is both a fragmentation
    source and a page-fault storm.  */

static sz pool_stride(sz bytes) { return (sz) xpar_align_up(bytes, 64); }

static u8 * pool_new(u32 count, sz stride) {
  u64 need = (u64) count * (u64) stride;
  FATAL_UNLESS("Column chunk too wide for this host's address space.",
               need <= (u64) (sz) -1);
  return (u8 *) xpar_alloc_aligned((sz) need, 64);
}

/*  Walsh-Hadamard transform, modulo 2^w - 1.
    The error locator is a product over the erased positions, so it is a
    sum in the exponent, and a sum over `i XOR e` is an XOR convolution:
    transform, multiply pointwise, transform back. No scaling is needed
    on the way back because the transform length 2^w is congruent to 1
    modulo 2^w - 1, which is the reason the exponent group and the
    transform length are the pair they are.

    The two halves of a butterfly fold the carry out of bit w back into
    bit 0 and admit 2^w - 1 as a second spelling of zero, so a value
    stays in [0, 2^w - 1] and never needs a division. The subtraction
    relies on the borrow wrapping in 32-bit arithmetic and on 2^(32-w)
    being a multiple of 2^w, which holds for every w <= 16.  */

static u32 fwht_add(u32 a, u32 b, u32 bits, u32 mask) {
  u32 s = a + b;
  return (s + (s >> bits)) & mask;
}

static u32 fwht_sub(u32 a, u32 b, u32 bits, u32 mask) {
  u32 d = a - b;
  return (d + (d >> bits)) & mask;
}

/*  a * b modulo 2^w - 1, by folding the high half of the product onto
    the low one twice rather than dividing. The product fits 32 bits
    because (2^16 - 1)^2 is 2^32 - 2^17 + 1, and two folds are exact for
    a modulus of this shape. It matters: the pointwise step is 2^w of
    these, and a 32-bit division apiece would cost more than both Walsh
    passes around it.  */
static u32 fwht_mulmod(u32 a, u32 b, u32 bits, u32 mask) {
  u32 p = a * b;
  p = (p & mask) + (p >> bits);
  return (p & mask) + (p >> bits);
}

/*  Process two layers per load. Blocks starting at `trunc` contain only
    zero subblocks and can be skipped.  */
static void fwht(u16 * d, u32 len, u32 trunc, u32 bits) {
  u32 mask = (1u << bits) - 1, dist = 1, dist4 = 4;
  for (; dist4 <= len; dist = dist4, dist4 <<= 2)
    for (u32 r = 0; r < trunc; r += dist4)
      for (u32 i = r; i < r + dist; i++) {
        u32 t0 = d[i],            t1 = d[i + dist];
        u32 t2 = d[i + dist * 2], t3 = d[i + dist * 3], a;
        a = fwht_add(t0, t1, bits, mask);
        t1 = fwht_sub(t0, t1, bits, mask);  t0 = a;
        a = fwht_add(t2, t3, bits, mask);
        t3 = fwht_sub(t2, t3, bits, mask);  t2 = a;
        a = fwht_add(t0, t2, bits, mask);
        t2 = fwht_sub(t0, t2, bits, mask);  t0 = a;
        a = fwht_add(t1, t3, bits, mask);
        t3 = fwht_sub(t1, t3, bits, mask);  t1 = a;
        d[i]            = (u16) t0;  d[i + dist]     = (u16) t1;
        d[i + dist * 2] = (u16) t2;  d[i + dist * 3] = (u16) t3;
      }
  xpar_assert(dist == len);
}

/*  Skew values.
    The twisted factors of the transform, derived from the Cantor basis
    of the field. The recurrence is Leopard's FFTInitialize read into the
    standard representation: where upstream starts from the indices
    2, 4, ... 2^(w-1) of its relabelled field, the same elements here are
    the Cantor basis entries 1 through w-1, and where it XORs an index
    with 1 it XORs a value with the basis element of index 1, which is
    the constant 1 in both fields.

    The largest index written is 2*2^(w-1) - 2^k - 1 <= 2^w - 2, so the
    table is exactly 2^w - 1 entries: the same bound the encoder needs,
    because ceil(S/m)*m <= 2^w - m whenever S + m <= 2^w and m divides
    2^w.  */

static void skew_build8(u8 * sk) {
  u8 t[7];
  for (u32 i = 1; i < 8; i++) t[i - 1] = xpar_gf8_cantor[i];
  for (u32 v = 0; v < 7; v++) {
    u32 step = 1u << (v + 1);
    sk[(1u << v) - 1] = 0;
    for (u32 i = v; i < 7; i++) {
      u32 s = 1u << (i + 1);
      for (u32 j = (1u << v) - 1; j < s; j += step)
        sk[j + s] = (u8) (sk[j] ^ t[i]);
    }
    u8 ib = xpar_gf8_inv(xpar_gf8_mul(t[v],
                                      (u8) (t[v] ^ xpar_gf8_cantor[0])));
    for (u32 i = v + 1; i < 7; i++)
      t[i] = xpar_gf8_mul(t[i],
                          xpar_gf8_mul((u8) (t[i] ^ xpar_gf8_cantor[0]), ib));
  }
}

static void skew_build16(u16 * sk) {
  u16 t[15];
  for (u32 i = 1; i < 16; i++) t[i - 1] = xpar_gf16_cantor[i];
  for (u32 v = 0; v < 15; v++) {
    u32 step = 1u << (v + 1);
    sk[(1u << v) - 1] = 0;
    for (u32 i = v; i < 15; i++) {
      u32 s = 1u << (i + 1);
      for (u32 j = (1u << v) - 1; j < s; j += step)
        sk[j + s] = (u16) (sk[j] ^ t[i]);
    }
    u16 ib = xpar_gf16_inv(xpar_gf16_mul(t[v],
                                (u16) (t[v] ^ xpar_gf16_cantor[0])));
    for (u32 i = v + 1; i < 15; i++)
      t[i] = xpar_gf16_mul(t[i],
                 xpar_gf16_mul((u16) (t[i] ^ xpar_gf16_cantor[0]), ib));
  }
}

/*  Slot i of the transform evaluates at the field element whose Cantor
    coordinates are the bits of i, so the table below is the log of that
    element, and slot 0 is given the exponent 0 rather than the log of
    zero. That convention is what drops the (x + x) factor out of the
    error locator at an erased position, which is exactly the term the
    formal derivative removes later.  */
static void walsh_build(fft_codec * cd) {
  u32 order = cd->order;
  u16 * lw = (u16 *) xpar_alloc_raw((sz) order * sizeof(u16));
  lw[0] = 0;
  for (u32 b = 0; b < cd->bits; b++) {
    u32 wd = 1u << b;
    u16 bas = cd->f16 ? xpar_gf16_cantor[b] : (u16) xpar_gf8_cantor[b];
    for (u32 j = 0; j < wd; j++) lw[j + wd] = (u16) (lw[j] ^ bas);
  }
  for (u32 i = 1; i < order; i++)
    lw[i] = cd->f16 ? xpar_gf16_log[lw[i]] : (u16) xpar_gf8_log[lw[i]];
  lw[0] = 0;
  fwht(lw, order, order, cd->bits);
  cd->walsh = lw;
}

/*  The transforms.
    Upstream unrolls two layers at a time so that four buffers stay in
    registers across a fused butterfly. Here the butterfly is already one
    dispatched region kernel over the whole column, so the layers are
    written plainly; the two forms visit the same butterflies with the
    same skew values, and the plain one additionally skips a block that
    the unrolled one enters and finds empty.

    A zero skew value degenerates the butterfly to y ^= x in both
    directions, which is a plain XOR and not a multiply by zero.  */

static void fft_fwd(const fft_codec * cd, u8 ** w, u32 trunc, u32 len,
                    u32 base, sz bytes) {
  const xpar_gf_kernels * gk = xpar_gf_active();
  xpar_gf16_coef m16;
  for (u32 dist = len >> 1; dist; dist >>= 1)
    for (u32 r = 0; r < trunc; r += dist << 1) {
      u32 c = skew_at(cd, base + r + dist - 1);
      const xpar_gf8_coef * m8 = NULL;
      if (c) {
        if (cd->f16) xpar_gf16_prepare(&m16, (u16) c);
        else         m8 = &cd->prep8[c];
      }
      for (u32 i = r; i < r + dist; i++) {
        u8 * x = w[i], * y = w[i + dist];
        if (!c)           gk->xor2 (y, x, bytes);
        else if (cd->f16) gk->fft16(x, y, bytes, &m16);
        else              gk->fft8 (x, y, bytes, m8);
      }
    }
}

static void fft_inv(const fft_codec * cd, u8 ** w, u32 trunc, u32 len,
                    u32 base, sz bytes) {
  const xpar_gf_kernels * gk = xpar_gf_active();
  xpar_gf16_coef m16;
  for (u32 dist = 1; dist < len; dist <<= 1)
    for (u32 r = 0; r < trunc; r += dist << 1) {
      u32 c = skew_at(cd, base + r + dist - 1);
      const xpar_gf8_coef * m8 = NULL;
      if (c) {
        if (cd->f16) xpar_gf16_prepare(&m16, (u16) c);
        else         m8 = &cd->prep8[c];
      }
      for (u32 i = r; i < r + dist; i++) {
        u8 * x = w[i], * y = w[i + dist];
        if (!c)           gk->xor2  (y, x, bytes);
        else if (cd->f16) gk->ifft16(x, y, bytes, &m16);
        else              gk->ifft8 (x, y, bytes, m8);
      }
    }
}

static u32 fft_m(bool low, u64 s, u64 r) {
  return (u32) xpar_next_pow2(low ? s : r);
}

static u32 fft_n(u32 m, bool low, u64 s, u64 r) {
  return (u32) xpar_next_pow2(m + (low ? r : s));
}

bool xpar_fft_supports_axis(u8 kind, u8 field_log2, u64 s, u64 r,
                            u8 axis_log2) {
  bool low = kind == XPAR_CODEC_FFT_LOW;
  u64 m;
  if (kind != XPAR_CODEC_FFT && !low) return false;
  if (field_log2 != 8 && field_log2 != 16) return false;
  if (s < 1 || r < 1) return false;
  if (low ? r <= s : r > s) return false;
  if (axis_log2 > field_log2) return false;
  m = (u64) 1 << axis_log2;
  if (low && m != xpar_next_pow2(s)) return false;
  if (!low && r > m) return false;
  return (low ? r : s) + m <= ((u64) 1 << field_log2);
}

void * xpar_fft_new_axis(u8 kind, u8 field_log2, u64 s, u64 r,
                         u8 axis_log2) {
  xpar_gf_init();
  FATAL_UNLESS("internal: unsupported FFT codec geometry.",
               xpar_fft_supports_axis(kind, field_log2, s, r, axis_log2));
  fft_codec * cd = (fft_codec *) xpar_calloc(1, sizeof(fft_codec));
  cd->f16   = field_log2 == 16;
  cd->low   = kind == XPAR_CODEC_FFT_LOW;
  cd->bits  = field_log2;
  cd->order = 1u << field_log2;
  cd->mod   = cd->order - 1;
  cd->s = (u32) s;  cd->r = (u32) r;
  cd->m = 1u << axis_log2;
  cd->n = fft_n(cd->m, cd->low, s, r);
  if (cd->f16) {
    u16 * sk = (u16 *) xpar_alloc_raw((sz) cd->mod * sizeof(u16));
    skew_build16(sk);  cd->skew = sk;
  } else {
    u8 * sk = (u8 *) xpar_alloc_raw((sz) cd->mod);
    skew_build8(sk);  cd->skew = sk;
    cd->prep8 = (xpar_gf8_coef *) xpar_alloc_raw(256 *
                                                 sizeof(xpar_gf8_coef));
    Fi(256, xpar_gf8_prepare(&cd->prep8[i], (u8) i));
  }
  return cd;
}

void xpar_fft_free(void * self) {
  fft_codec * cd = (fft_codec *) self;
  if (!cd) return;
  xpar_free(cd->skew);  xpar_free(cd->prep8);  xpar_free(cd->walsh);
  xpar_free(cd);
}

/*  Encode.  */

xpar_codec_status xpar_fft_encode(void * self, const u8 * const * data,
                                  u8 * const * rec, sz bytes) {
  fft_codec * cd = (fft_codec *) self;
  const xpar_gf_kernels * gk = xpar_gf_active();
  u32 m = cd->m, s = cd->s, r = cd->r;
  xpar_assert(!cd->f16 || (bytes & 1) == 0);
  if (!bytes) return XPAR_CODEC_OK;
  if (cd->low) {
    sz strip = fft_strip(2 * m, bytes), stride = pool_stride(strip);
    u8 * pool = pool_new(2 * m, stride);
    u8 ** w = (u8 **) xpar_alloc_raw((sz) 2 * m * sizeof(u8 *));
    for (u32 j = 0; j < 2 * m; j++) w[j] = pool + (sz) j * stride;
    for (sz off = 0; off < bytes; off += strip) {
      sz len = MIN(strip, bytes - off);
      for (u32 j = 0; j < s; j++) xpar_memcpy(w[j], data[j] + off, len);
      for (u32 j = s; j < m; j++) xpar_memset(w[j], 0, len);
      fft_inv(cd, w, s, m, 0, len);
      for (u32 b = 0; b < r; b += m) {
        u32 cnt = MIN(m, r - b);
        u8 ** t = w + m;
        for (u32 j = 0; j < cnt; j++) t[j] = rec[b + j] + off;
        for (u32 j = 0; j < m; j++) xpar_memcpy(t[j], w[j], len);
        fft_fwd(cd, t, cnt, m, m + b, len);
        for (u32 j = 0; j < cnt; j++)
          t[j] = pool + (sz) (m + j) * stride;
      }
    }
    xpar_free(w);
    xpar_free_aligned(pool);
    return XPAR_CODEC_OK;
  }
  u32 first = MIN(s, m);
  sz strip = fft_strip(2 * m, bytes), stride = pool_stride(strip);
  u8 * pool = pool_new(2 * m - r, stride);
  /*  The first R work buffers are the caller's recovery slices: the
      forward transform's outputs land there, so aliasing them saves both
      R buffers and a copy of the answer.  */
  u8 ** w = (u8 **) xpar_alloc_raw((sz) 2 * m * sizeof(u8 *));
  for (u32 j = r; j < 2 * m; j++) w[j] = pool + (sz) (j - r) * stride;
  for (sz off = 0; off < bytes; off += strip) {
    sz len = MIN(strip, bytes - off);
    for (u32 j = 0; j < r; j++)     w[j] = rec[j] + off;
    for (u32 j = 0; j < first; j++) xpar_memcpy(w[j], data[j] + off, len);
    for (u32 j = first; j < m; j++) xpar_memset(w[j], 0, len);
    fft_inv(cd, w, first, m, m, len);
    for (u32 b = m; b < s; b += m) {
      u32 cnt = MIN(m, s - b);
      u8 ** t = w + m;
      for (u32 j = 0; j < cnt; j++) xpar_memcpy(t[j], data[b + j] + off, len);
      for (u32 j = cnt; j < m; j++) xpar_memset(t[j], 0, len);
      fft_inv(cd, t, cnt, m, m + b, len);
      for (u32 j = 0; j < m; j++) gk->xor2(w[j], t[j], len);
    }
    fft_fwd(cd, w, r, m, 0, len);
  }
  xpar_free(w);  xpar_free_aligned(pool);
  return XPAR_CODEC_OK;
}

/*  Plan construction pays the full-field Walsh passes once per erasure
    pattern.  Full-field scratch is temporary so encoders do not retain it.  */

void * xpar_fft_plan_new(void * self, const u8 * dpres, const u8 * rpres,
                         xpar_codec_status * status) {
  fft_codec * cd = (fft_codec *) self;
  u32 m = cd->m, s = cd->s, r = cd->r, e = 0, got = 0;
  for (u32 i = 0; i < s; i++) if (!dpres[i]) e++;
  for (u32 i = 0; i < r; i++) if (rpres[i]) got++;
  if (got < e) { *status = XPAR_CODEC_TOO_MANY_LOST;  return NULL; }
  *status = XPAR_CODEC_OK;
  fft_plan * pl = (fft_plan *) xpar_calloc(1, sizeof(fft_plan));
  pl->cd = cd;  pl->e = e;
  pl->dpres = (u8 *) xpar_alloc_raw(s);
  pl->rpres = (u8 *) xpar_alloc_raw(r ? r : 1);
  xpar_memcpy(pl->dpres, dpres, s);
  xpar_memcpy(pl->rpres, rpres, r);
  if (!e) return pl;
  pl->lost = (u32 *) xpar_alloc_raw((sz) e * sizeof(u32));
  for (u32 i = 0, k = 0; i < s; i++) if (!dpres[i]) pl->lost[k++] = i;
  if (!cd->walsh) walsh_build(cd);
  u16 * loc = (u16 *) xpar_calloc(cd->order, sizeof(u16));
  u32 active;
  if (cd->low) {
    for (u32 i = 0; i < s; i++) if (!dpres[i]) loc[i] = 1;
    for (u32 i = 0; i < r; i++) if (!rpres[i]) loc[m + i] = 1;
    active = m + r;
    for (u32 i = active; i < cd->order; i++) loc[i] = 1;
  } else {
    for (u32 i = 0; i < r; i++) if (!rpres[i]) loc[i] = 1;
    for (u32 i = r; i < m; i++) loc[i] = 1;
    for (u32 i = 0; i < s; i++) if (!dpres[i]) loc[m + i] = 1;
    active = m + s;
  }
  fwht(loc, cd->order, cd->low ? cd->order : active, cd->bits);
  for (u32 i = 0; i < cd->order; i++)
    loc[i] = (u16) fwht_mulmod(loc[i], cd->walsh[i], cd->bits, cd->mod);
  fwht(loc, cd->order, cd->order, cd->bits);
  pl->elog = (u16 *) xpar_alloc_raw((sz) active * sizeof(u16));
  for (u32 i = 0; i < active; i++) {
    u32 v = loc[i];
    pl->elog[i] = (u16) (v >= cd->mod ? v - cd->mod : v);
  }
  xpar_free(loc);
  return pl;
}

void xpar_fft_plan_free(void * self) {
  fft_plan * pl = (fft_plan *) self;
  if (!pl) return;
  xpar_free(pl->lost);   xpar_free(pl->dpres);
  xpar_free(pl->rpres);  xpar_free(pl->elog);
  xpar_free(pl);
}

xpar_codec_status xpar_fft_plan_apply(const void * self, u8 * const * data,
                                      const u8 * const * rec, sz bytes) {
  const fft_plan * pl = (const fft_plan *) self;
  const fft_codec * cd = pl->cd;
  const xpar_gf_kernels * gk = xpar_gf_active();
  u32 m = cd->m, s = cd->s, r = cd->r, n = cd->n;
  xpar_assert(!cd->f16 || (bytes & 1) == 0);
  if (!pl->e || !bytes) return XPAR_CODEC_OK;
  sz strip = fft_strip(n, bytes), stride = pool_stride(strip);
  u8 * pool = pool_new(n, stride);
  u8 ** w = (u8 **) xpar_alloc_raw((sz) n * sizeof(u8 *));
  for (u32 i = 0; i < n; i++) w[i] = pool + (sz) i * stride;
  for (sz off = 0; off < bytes; off += strip) {
    sz len = MIN(strip, bytes - off);
    u32 active;
    if (cd->low) {
      for (u32 i = 0; i < s; i++) {
        if (pl->dpres[i])
          fld_mul(cd->f16, w[i], data[i] + off, len,
                  fld_exp(cd, pl->elog[i]));
        else xpar_memset(w[i], 0, len);
      }
      for (u32 i = s; i < m; i++) xpar_memset(w[i], 0, len);
      for (u32 i = 0; i < r; i++) {
        if (pl->rpres[i])
          fld_mul(cd->f16, w[m + i], rec[i] + off, len,
                  fld_exp(cd, pl->elog[m + i]));
        else xpar_memset(w[m + i], 0, len);
      }
      active = m + r;
    } else {
      for (u32 i = 0; i < r; i++) {
        if (pl->rpres[i])
          fld_mul(cd->f16, w[i], rec[i] + off, len,
                  fld_exp(cd, pl->elog[i]));
        else xpar_memset(w[i], 0, len);
      }
      for (u32 i = r; i < m; i++) xpar_memset(w[i], 0, len);
      for (u32 i = 0; i < s; i++) {
        if (pl->dpres[i])
          fld_mul(cd->f16, w[m + i], data[i] + off, len,
                  fld_exp(cd, pl->elog[m + i]));
        else xpar_memset(w[m + i], 0, len);
      }
      active = m + s;
    }
    for (u32 i = active; i < n; i++) xpar_memset(w[i], 0, len);
    fft_inv(cd, w, active, n, 0, len);
    /*  Formal derivative. The width is the lowest set bit of i, so i is
        a multiple of it and i + width never leaves the array.  */
    for (u32 i = 1; i < n; i++) {
      u32 width = ((i ^ (i - 1)) + 1) >> 1;
      for (u32 k = 0; k < width; k++)
        gk->xor2(w[i - width + k], w[i + k], len);
    }
    fft_fwd(cd, w, active, n, 0, len);
    for (u32 l = 0; l < pl->e; l++) {
      u32 i = pl->lost[l], at = cd->low ? i : m + i;
      u32 v = pl->elog[at];
      fld_mul(cd->f16, data[i] + off, w[at], len,
              fld_exp(cd, v ? cd->mod - v : 0));
    }
  }
  xpar_free(w);  xpar_free_aligned(pool);
  return XPAR_CODEC_OK;
}

/*  Footprint and work.  */

/*  The trailing 256 bytes cover the codec and plan structs themselves,
    which are fixed and small but not zero.  */
static u64 fft_tables(u8 field_log2, bool decode) {
  u64 order = (u64) 1 << field_log2;
  u64 t = (order - 1) * (field_log2 == 8 ? 1 : 2);
  if (field_log2 == 8) t += 256 * (u64) sizeof(xpar_gf8_coef);
  if (decode) t += order * 2 * 2;  /*  Walsh table plus its scratch.  */
  return t;
}

u64 xpar_fft_encode_footprint_axis(u8 kind, u8 field_log2, u64 s, u64 r,
                                   u8 axis_log2, sz bytes) {
  bool low = kind == XPAR_CODEC_FFT_LOW;
  u64 m = (u64) 1 << axis_log2;
  u64 stride = pool_stride(fft_strip((u32) (2 * m), bytes));
  u64 scratch = low ? 2 * m : 2 * m - r;
  return (s + r) * (u64) pool_stride(bytes) + scratch * stride
         + 2 * m * (u64) sizeof(u8 *)
         + fft_tables(field_log2, false) + 256;
}

u64 xpar_fft_decode_footprint_axis(u8 kind, u8 field_log2, u64 s, u64 r,
                                   u8 axis_log2, sz bytes) {
  bool low = kind == XPAR_CODEC_FFT_LOW;
  u64 m = (u64) 1 << axis_log2, n = fft_n((u32) m, low, s, r);
  u64 active = m + (low ? r : s);
  u64 stride = pool_stride(fft_strip((u32) n, bytes));
  return (s + r) * (u64) pool_stride(bytes) + n * stride
         + n * (u64) sizeof(u8 *)
         + active * 2 + s + r + r * 4
         + fft_tables(field_log2, true) + 256;
}

u64 xpar_fft_encode_work(u8 kind, u64 s, u64 r, sz bytes) {
  bool low = kind == XPAR_CODEC_FFT_LOW;
  u64 m = fft_m(low, s, r), butterflies;
  if (low) butterflies = (m + xpar_ceil_div(r, m) * m) / 2;
  else     butterflies = (s + m) / 2;
  return butterflies * (u64) xpar_log2_floor(m) * (u64) bytes;
}
