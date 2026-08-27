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

/*  GF(2^8) and GF(2^16) tables, coefficient preparation, scalar kernels,
    and dispatch.  alpha = 2 must have full multiplicative order.  Scalar
    kernels also provide vector tails and the conformance reference.  */

#include "gf.h"
#include "port-cpu.h"

u8 xpar_gf8_exp[512];
u8 xpar_gf8_log[256];
u8 xpar_gf8_inv_tab[256];

static u16 * gf16_exp_store;
static u16 * gf16_log_store;

const u16 * xpar_gf16_exp;
const u16 * xpar_gf16_log;

/*  8x8 GF(2) matrices of `x -> x * c` for every c, 2 KiB, so that
    preparing a GF(2^8) coefficient for the affine tiers is one load
    rather than eight multiplies and sixty-four bit tests.

    Sixty-five thousand of those would be 2 MiB, so GF(2^16) keeps only
    the sixteen matrices of the basis elements x^j. `x -> x * c` is
    linear in c as well as in x, so the matrix of an arbitrary c is the
    XOR of the matrices of its set bits: at most sixty-four XORs instead
    of sixteen multiplies and two hundred and fifty-six bit tests.  */
static u64 gf8_aff_tab[256];
static u64 gf16_aff_basis[16][4];

/*  Cantor bases.  */
const u8 xpar_gf8_cantor[8] = { 1, 214, 152, 146, 86, 200, 88, 230 };
const u16 xpar_gf16_cantor[16] = {
  0x0001, 0xACCA, 0x3C0E, 0x163E, 0xC582, 0xED2E, 0x914C, 0x4012,
  0x6C98, 0x10D8, 0x6A72, 0xB900, 0xFDB8, 0xFB34, 0xFF38, 0x991E
};

static void gf8_tables(void) {
  u32 x = 1, i;
  for (i = 0; i < 255; i++) {
    /*  Order exactly 255, not merely dividing it. Every nonzero element
        satisfies a^255 = 1 under any irreducible modulus, so only an
        early return to 1 distinguishes a non-primitive one.  */
    xpar_assert(i == 0 || x != 1);
    xpar_gf8_exp[i] = (u8) x;  xpar_gf8_log[x] = (u8) i;
    x <<= 1;  if (x & 0x100u) x ^= XPAR_GF8_POLY;
  }
  xpar_assert(x == 1);
  /*  Doubling the exp table lets a product index it with the plain sum
      of two logs, which is at most 508, and removes the reduction from
      the inner loop of every scalar kernel.  */
  for (i = 255; i < 512; i++) xpar_gf8_exp[i] = xpar_gf8_exp[i - 255];
  xpar_gf8_log[0] = 0;
  xpar_gf8_inv_tab[0] = 0;
  for (i = 1; i < 256; i++)
    xpar_gf8_inv_tab[i] = xpar_gf8_exp[255u - xpar_gf8_log[i]];
}

static void gf16_tables(void) {
  u32 x = 1, i;
  gf16_exp_store = (u16 *) xpar_alloc_raw(131070u * sizeof(u16));
  gf16_log_store = (u16 *) xpar_alloc_raw(65536u * sizeof(u16));
  for (i = 0; i < 65535u; i++) {
    xpar_assert(i == 0 || x != 1);
    gf16_exp_store[i] = (u16) x;  gf16_log_store[x] = (u16) i;
    x <<= 1;  if (x & 0x10000u) x ^= XPAR_GF16_POLY;
  }
  xpar_assert(x == 1);
  for (i = 65535u; i < 131070u; i++)
    gf16_exp_store[i] = gf16_exp_store[i - 65535u];
  gf16_log_store[0] = 0;
  xpar_gf16_exp = gf16_exp_store;
  xpar_gf16_log = gf16_log_store;
}

/*  Affine matrices.
    GF2P8AFFINEQB computes, for each input byte b and the qword M of its
    own 64-bit lane, `out.bit[i] = parity(M.byte[7-i] & b)`. So byte 7-i
    of M is the row that produces output bit i, and bit k of that row is
    the coefficient of input bit k. Multiplication by a constant c is
    GF(2)-linear, so its matrix has column k equal to `(1 << k) * c`, and
    bit i of that column belongs at qword bit `8*(7-i) + k`.  */

static u64 gf8_affine(u8 c) {
  u64 mx = 0;  int k, i;
  for (k = 0; k < 8; k++) {
    u8 col = xpar_gf8_mul((u8) (1u << k), c);
    for (i = 0; i < 8; i++)
      if ((col >> i) & 1) mx |= (u64) 1 << (8 * (7 - i) + k);
  }
  return mx;
}

/*  The same derivation over the 16x16 matrix of a GF(2^16) constant, cut
    into the four 8x8 blocks a byte-wise affine instruction can apply:
    [0] low byte of the product from the low byte of the input, [1] low
    from high, [2] high from low, [3] high from high.  */
static void gf16_affine(u16 c, u64 blk[4]) {
  int k, i;
  blk[0] = blk[1] = blk[2] = blk[3] = 0;
  for (k = 0; k < 8; k++) {
    u16 cl = xpar_gf16_mul((u16) (1u << k), c);
    u16 ch = xpar_gf16_mul((u16) (0x100u << k), c);
    for (i = 0; i < 8; i++) {
      u64 bit = (u64) 1 << (8 * (7 - i) + k);
      if ((cl >> i) & 1)       blk[0] |= bit;
      if ((ch >> i) & 1)       blk[1] |= bit;
      if ((cl >> (i + 8)) & 1) blk[2] |= bit;
      if ((ch >> (i + 8)) & 1) blk[3] |= bit;
    }
  }
}

/*  Coefficient preparation.
    Both forms are always built, so a prepared coefficient outlives a
    tier change and `benchmark --tiers` can switch tiers under one.

    The split tables come out of the same linearity that gives the affine
    matrices. A nibble table entry is `(i << 4k) * c`, and `i << 4k` is
    the sum of the basis elements x^(4k+b) over the set bits b of i, so
    the sixteen entries are the sixteen XOR-combinations of four
    products. Walking the basis with a doubling loop builds each table in
    fifteen XORs instead of sixteen log-table multiplies, and it needs no
    table at all: the basis is x^j * c, reached by repeated doubling from
    c itself.  */

static u8 gf8_xtime(u8 v) {
  return (u8) ((v << 1) ^ ((v & 0x80u) ? (XPAR_GF8_POLY & 0xFFu) : 0));
}
static u16 gf16_xtime(u16 v) {
  return (u16) ((v << 1) ^ ((v & 0x8000u) ? (XPAR_GF16_POLY & 0xFFFFu) : 0));
}

void xpar_gf8_prepare(xpar_gf8_coef * m, u8 c) {
  u8 col[8];  int i, j;
  m->c = c;  m->affine = gf8_aff_tab[c];
  col[0] = c;
  for (j = 1; j < 8; j++) col[j] = gf8_xtime(col[j - 1]);
  m->tab[0] = 0;  m->tab[16] = 0;
  for (j = 0; j < 4; j++)
    for (i = 0; i < (1 << j); i++) {
      m->tab[(1 << j) + i]      = (u8) (m->tab[i] ^ col[j]);
      m->tab[16 + (1 << j) + i] = (u8) (m->tab[16 + i] ^ col[4 + j]);
    }
}

static bool gf_want_tab6 = false;

void xpar_gf16_prepare(xpar_gf16_coef * m, u16 c) {
  u16 col[16], t[16];  int i, j, k;
  m->c = c;
  m->affine[0] = m->affine[1] = m->affine[2] = m->affine[3] = 0;
  for (j = 0; j < 16; j++)
    if ((c >> j) & 1) {
      m->affine[0] ^= gf16_aff_basis[j][0];
      m->affine[1] ^= gf16_aff_basis[j][1];
      m->affine[2] ^= gf16_aff_basis[j][2];
      m->affine[3] ^= gf16_aff_basis[j][3];
    }
  col[0] = c;
  for (j = 1; j < 16; j++) col[j] = gf16_xtime(col[j - 1]);
  for (k = 0; k < 4; k++) {
    t[0] = 0;
    for (j = 0; j < 4; j++)
      for (i = 0; i < (1 << j); i++)
        t[(1 << j) + i] = (u16) (t[i] ^ col[4 * k + j]);
    for (i = 0; i < 16; i++) {
      m->tab[2 * k    ][i] = (u8) t[i];
      m->tab[2 * k + 1][i] = (u8) (t[i] >> 8);
    }
  }
  /*  Only VBMI uses tab6; tier selection precedes coefficient setup.  */
  if (!gf_want_tab6) return;
  for (k = 0; k < 3; k++) {
    int bits = k == 2 ? 4 : 6;
    u16 w[64];
    w[0] = 0;
    /*  Extend each block with one basis column.  */
    for (j = 0; j < bits; j++)
      for (i = 0; i < (1 << j); i++)
        w[(1 << j) + i] = (u16) (w[i] ^ col[6 * k + j]);
    /*  Repeat the final four-bit table to fill 64 entries.  */
    for (i = 1 << bits; i < 64; i++) w[i] = w[i & ((1 << bits) - 1)];
    for (i = 0; i < 64; i++) {
      m->tab6[2 * k][i] = (u8) w[i];
      m->tab6[2 * k + 1][i] = (u8) (w[i] >> 8);
    }
  }
}

/*  Reference kernels.
    log[0] is 0 rather than a sentinel (gf.h says why), so a zero operand
    would otherwise index exp at log[c] and yield c. The mask is the
    correction, and it is branchless because the operand is data and the
    branch would be unpredictable.  */

static u8 gf8_mulc(u8 a, u32 lc) {
  u32 z = a ? 0xFFFFFFFFu : 0u;
  return (u8) (xpar_gf8_exp[lc + xpar_gf8_log[a]] & z);
}

static u16 gf16_mulc(u16 a, u32 lc) {
  u32 z = a ? 0xFFFFFFFFu : 0u;
  return (u16) (xpar_gf16_exp[lc + xpar_gf16_log[a]] & z);
}

void xpar_gf8_mac_ref(u8 * d, const u8 * s, sz n, u8 c) {
  if (!c) return;
  u32 lc = xpar_gf8_log[c];
  for (sz i = 0; i < n; i++) d[i] ^= gf8_mulc(s[i], lc);
}

void xpar_gf8_mul_ref(u8 * d, const u8 * s, sz n, u8 c) {
  if (!c) { xpar_memset(d, 0, n);  return; }
  u32 lc = xpar_gf8_log[c];
  for (sz i = 0; i < n; i++) d[i] = gf8_mulc(s[i], lc);
}

/*  A GF(2^16) region is little-endian u16 by definition, not by host
    property: the shuffle kernels necessarily read the even byte of a
    symbol as its low half, so the scalar walk goes through rd16/wr16 and
    the two agree on a big-endian host. A trailing odd byte is not part
    of any symbol and is left alone.  */

void xpar_gf16_mac_ref(u8 * d, const u8 * s, sz n, u16 c) {
  if (!c) return;
  u32 lc = xpar_gf16_log[c];
  for (sz i = 0; i + 2 <= n; i += 2)
    xpar_wr16(d + i, (u16) (xpar_rd16(d + i) ^
                            gf16_mulc(xpar_rd16(s + i), lc)));
}

void xpar_gf16_mul_ref(u8 * d, const u8 * s, sz n, u16 c) {
  if (!c) { xpar_memset(d, 0, n & ~(sz) 1);  return; }
  u32 lc = xpar_gf16_log[c];
  for (sz i = 0; i + 2 <= n; i += 2)
    xpar_wr16(d + i, gf16_mulc(xpar_rd16(s + i), lc));
}

void xpar_xor2_ref(u8 * d, const u8 * s, sz n) {
  for (sz i = 0; i < n; i++) d[i] ^= s[i];
}

void xpar_xor3_ref(u8 * d, const u8 * a, const u8 * b, sz n) {
  for (sz i = 0; i < n; i++) d[i] = (u8) (a[i] ^ b[i]);
}

/*  The additive-FFT butterflies, fused so that x and y are read once and
    written once.  */

void xpar_gf8_fft2_ref(u8 * x, u8 * y, sz n, u8 c) {
  if (!c) { xpar_xor2_ref(y, x, n);  return; }
  u32 lc = xpar_gf8_log[c];
  for (sz i = 0; i < n; i++) {
    u8 vx = (u8) (x[i] ^ gf8_mulc(y[i], lc));
    x[i] = vx;  y[i] = (u8) (y[i] ^ vx);
  }
}

void xpar_gf8_ifft2_ref(u8 * x, u8 * y, sz n, u8 c) {
  if (!c) { xpar_xor2_ref(y, x, n);  return; }
  u32 lc = xpar_gf8_log[c];
  for (sz i = 0; i < n; i++) {
    u8 vy = (u8) (y[i] ^ x[i]);
    y[i] = vy;  x[i] = (u8) (x[i] ^ gf8_mulc(vy, lc));
  }
}

void xpar_gf16_fft2_ref(u8 * x, u8 * y, sz n, u16 c) {
  if (!c) { xpar_xor2_ref(y, x, n & ~(sz) 1);  return; }
  u32 lc = xpar_gf16_log[c];
  for (sz i = 0; i + 2 <= n; i += 2) {
    u16 vy = xpar_rd16(y + i);
    u16 vx = (u16) (xpar_rd16(x + i) ^ gf16_mulc(vy, lc));
    xpar_wr16(x + i, vx);  xpar_wr16(y + i, (u16) (vy ^ vx));
  }
}

void xpar_gf16_ifft2_ref(u8 * x, u8 * y, sz n, u16 c) {
  if (!c) { xpar_xor2_ref(y, x, n & ~(sz) 1);  return; }
  u32 lc = xpar_gf16_log[c];
  for (sz i = 0; i + 2 <= n; i += 2) {
    u16 vx = xpar_rd16(x + i);
    u16 vy = (u16) (xpar_rd16(y + i) ^ vx);
    xpar_wr16(y + i, vy);
    xpar_wr16(x + i, (u16) (vx ^ gf16_mulc(vy, lc)));
  }
}

/*  Static tier preference avoids unstable startup calibration.  Prefer
    256-bit GFNI over 512-bit GFNI to avoid wide-vector frequency costs.  */

typedef struct {
  const xpar_gf_kernels * k;
  u32 need;    /*  CPU feature bits, all of which must be present.  */
} gf_tier;

static const gf_tier gf_tiers[] = {
#ifdef HAVE_GFNI
  { &xpar_gf_kernels_gfni256, XPAR_CPU_GFNI | XPAR_CPU_AVX2   },
#endif
#ifdef HAVE_GFNI512
  /*  All three AVX-512 subsets, not just F: the kernel is compiled with
      -mavx512bw and -mavx512vl and uses instructions from both, so gating
      on F alone would dispatch to it on a Knights Landing part and fault.  */
  { &xpar_gf_kernels_gfni512, XPAR_CPU_GFNI | XPAR_CPU_AVX512F |
                              XPAR_CPU_AVX512BW | XPAR_CPU_AVX512VL },
#endif
#ifdef HAVE_VBMI
  { &xpar_gf_kernels_vbmi512, XPAR_CPU_AVX512F | XPAR_CPU_AVX512BW |
                               XPAR_CPU_VBMI                         },
#endif
#ifdef HAVE_AVX2
  { &xpar_gf_kernels_avx2,    XPAR_CPU_AVX2                   },
#endif
#ifdef HAVE_PMULL
  { &xpar_gf_kernels_neon_clmul, XPAR_CPU_NEON | XPAR_CPU_PMULL },
#endif
#ifdef HAVE_SVE
  { &xpar_gf_kernels_sve, XPAR_CPU_SVE },
#endif
#ifdef HAVE_NEON
  { &xpar_gf_kernels_neon,    XPAR_CPU_NEON                   },
#endif
#ifdef HAVE_SSSE3
  { &xpar_gf_kernels_ssse3,   XPAR_CPU_SSSE3                  },
#endif
#ifdef HAVE_VSX
  { &xpar_gf_kernels_vsx,     XPAR_CPU_VSX                    },
#endif
#ifdef HAVE_RVV_CLMUL
  { &xpar_gf_kernels_rvv_clmul, XPAR_CPU_RVV | XPAR_CPU_RVVCLMUL },
#endif
#ifdef HAVE_RVV
  { &xpar_gf_kernels_rvv_shuffle, XPAR_CPU_RVV                },
#endif
  { &xpar_gf_kernels_scalar,  0                               }
};

#define GF_NTIERS ((int) ARRAY_LEN(gf_tiers))

static const xpar_gf_kernels * gf_k = &xpar_gf_kernels_scalar;
static int gf_cur = GF_NTIERS - 1;

int xpar_gf_tier_count(void) { return GF_NTIERS; }

const char * xpar_gf_tier_name(int tier) {
  return (tier < 0 || tier >= GF_NTIERS) ? NULL : gf_tiers[tier].k->name;
}

bool xpar_gf_tier_usable(int tier) {
  if (tier < 0 || tier >= GF_NTIERS) return false;
  return (xpar_cpu_features() & gf_tiers[tier].need) == gf_tiers[tier].need;
}

int xpar_gf_tier(void) { return gf_cur; }

const xpar_gf_kernels * xpar_gf_active(void) { return gf_k; }

bool xpar_gf_use_tier(int tier) {
  if (!xpar_gf_tier_usable(tier)) return false;
  gf_cur = tier;  gf_k = gf_tiers[tier].k;
  gf_want_tab6 = false;
#ifdef HAVE_VBMI
  if (gf_k == &xpar_gf_kernels_vbmi512) gf_want_tab6 = true;
#endif
  return true;
}

bool xpar_gf_use_tier_name(const char * name) {
  for (int i = 0; i < GF_NTIERS; i++)
    if (!xpar_strcmp(name, gf_tiers[i].k->name)) return xpar_gf_use_tier(i);
  return false;
}

void xpar_gf_use_default_tier(void) {
  for (int i = 0; i < GF_NTIERS; i++)
    if (xpar_gf_use_tier(i)) return;
}

void xpar_gf_init(void) {
  static bool done = false;
  if (done) return;
  gf8_tables();
  for (int c = 0; c < 256; c++) gf8_aff_tab[c] = gf8_affine((u8) c);
  gf16_tables();
  for (int j = 0; j < 16; j++)
    gf16_affine((u16) (1u << j), gf16_aff_basis[j]);
  xpar_gf_use_default_tier();
  done = true;
}

/*  Convenience wrappers.  */

static void gf_check(const void * d, const void * s, sz n) {
  const u8 * a = (const u8 *) d, * b = (const u8 *) s;
  xpar_assert(a == b || a + n <= b || b + n <= a);
}

void xpar_gf8_mul_region(u8 * dst, const u8 * src, sz n, u8 c) {
  if (!n) return;
  gf_check(dst, src, n);
  if (!c)     { xpar_memset(dst, 0, n);  return; }
  if (c == 1) { if (dst != src) xpar_memcpy(dst, src, n);  return; }
  xpar_gf8_coef m;  xpar_gf8_prepare(&m, c);
  gf_k->mul8(dst, src, n, &m);
}

void xpar_gf16_mul_region(u8 * dst, const u8 * src, sz n, u16 c) {
  xpar_assert((n & 1) == 0);
  if (!n) return;
  gf_check(dst, src, n);
  if (!c)     { xpar_memset(dst, 0, n);  return; }
  if (c == 1) { if (dst != src) xpar_memcpy(dst, src, n);  return; }
  xpar_gf16_coef m;  xpar_gf16_prepare(&m, c);
  gf_k->mul16(dst, src, n, &m);
}
