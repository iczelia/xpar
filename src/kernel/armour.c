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

/*  Reed-Solomon inner coding, framing, and correction.  */

#include "armour.h"
#include "port-cpu.h"

/*  Both field widths share the scalar decoder.  */

static u32 f_mul(const xpar_armour * a, u32 x, u32 y);
static u32 f_inv(const xpar_armour * a, u32 x);

struct xpar_armour {
  xpar_armour_params p;
  const xpar_armour_kernels * kern;
  u32 wb;        /*  W: symbol size in bytes, 1 or 2.  */
  u32 t2;        /*  n - k = 2t parity symbols.  */
  u32 t;         /*  Correction capacity, symbols per codeword.  */
  u32 order;     /*  2^w - 1.  */
  sz  lane;      /*  D*W: one symbol position of a frame, in bytes.  */
  u32 vb;        /*  Frames batched into one virtual lane; see below.  */
  sz  vlane;     /*  vb*lane: the widest lane the kernels are given.  */
  sz  vstep;     /*  Whole-vector step of the active tier; 0 when scalar.  */
  xpar_gf8_coef  * gen8;    /*  Taps, reversed: gen[u] = g[2t-1-u].  */
  xpar_gf16_coef * gen16;
  xpar_gf8_coef  * rt8;     /*  rt[j] = alpha^(fcr + j*prim).  */
  xpar_gf16_coef * rt16;
  u32 * gpoly;   /*  t2+1: g[i] is the coefficient of x^i, g[t2] = 1.  */
  u8  * par;     /*  t2*vlane: the encoder's rotating parity register.  */
  u8  * fb;      /*  vlane: one symbol position of feedback.  */
  u8  * syn;     /*  t2*vlane: the batch's syndromes, symbol-major.  */
  u8  * gsym;    /*  vlane: one symbol position gathered across frames.  */
  u32 * s;       /*  t2: one codeword's syndromes, unpacked.  */
  u32 * lam;     /*  t2+2: the error locator.  */
  u32 * bp;      /*  t2+2: Berlekamp-Massey's saved connection poly.  */
  u32 * tmp;     /*  t2+2.  */
  u32 * om;      /*  t2+2: Omega, the Forney numerator.  */
  u32 * ev;      /*  2t+2: Chien or erasure syndrome running terms.  */
  u32 * step;    /*  2t+2: fixed powers for those running terms.  */
  u32 * pos;     /*  2t+2: located error or known-erasure degrees.  */
  u32 * val;     /*  2t+2: located error or erasure magnitudes.  */
};

static u32 f_mul(const xpar_armour * a, u32 x, u32 y) {
  return a->wb == 1 ? xpar_gf8_mul((u8) x, (u8) y)
                    : xpar_gf16_mul((u16) x, (u16) y);
}
static u32 f_inv(const xpar_armour * a, u32 x) {
  return a->wb == 1 ? xpar_gf8_inv((u8) x) : xpar_gf16_inv((u16) x);
}
static u32 f_div(const xpar_armour * a, u32 x, u32 y) {
  return f_mul(a, x, f_inv(a, y));
}
static u32 f_alpha(const xpar_armour * a, u32 e) {
  return a->wb == 1 ? xpar_gf8_alpha_pow(e) : xpar_gf16_alpha_pow(e);
}
/*  alpha^(x*y) with the product reduced modulo the group order. The
    multiplication is widened defensively; at GF(2^16) the largest
    product is 65534^2, which does still fit in u32.  */
static u32 f_alpha_mul(const xpar_armour * a, u32 x, u32 y) {
  return f_alpha(a, (u32) (((u64) x * (u64) y) % a->order));
}
static u32 sym_rd(const xpar_armour * a, const u8 * p) {
  return a->wb == 1 ? (u32) *p : (u32) xpar_rd16(p);
}
static void sym_wr(const xpar_armour * a, u8 * p, u32 v) {
  if (a->wb == 1) *p = (u8) v;  else xpar_wr16(p, (u16) v);
}

/*  Parameters.  */

/*  GF(2^8) takes prim = 11 after the CCSDS convention, but neither the
    field nor fcr = 212 is the CCSDS RS(255,223) choice; the format pins
    all three. GF(2^16) takes the plain fcr = prim = 1.

    The polynomial is the one gf.h builds its tables and its region
    kernels around, x^8+x^4+x^3+x^2+1 and x^16+x^5+x^3+x^2+1.  */
void xpar_armour_defaults(xpar_armour_params * p, u32 symbol_bits) {
  p->symbol_bits = symbol_bits;
  p->depth       = 1;
  if (symbol_bits == 8) {
    p->poly = XPAR_GF8_POLY & 0xFFu;
    p->n    = 255;    p->k    = 223;
    p->fcr  = 212;    p->prim = 11;
  } else {
    p->poly = XPAR_GF16_POLY & 0xFFFFu;
    p->n    = 65535;  p->k    = 65407;
    p->fcr  = 1;      p->prim = 1;
  }
}

static u32 gcd_u32(u32 x, u32 y) {
  while (y) { u32 r = x % y;  x = y;  y = r; }
  return x;
}

const char * xpar_armour_check(const xpar_armour_params * p) {
  u32 order, t2, pr;  u64 frame;
  if (p->symbol_bits != 8 && p->symbol_bits != 16)
    return "Armour symbol width must be 8 or 16.";
  order = p->symbol_bits == 8 ? 255u : 65535u;
  if (p->poly != (p->symbol_bits == 8 ? (XPAR_GF8_POLY  & 0xFFu)
                                      : (XPAR_GF16_POLY & 0xFFFFu)))
    return "Armour field polynomial is not the one this build implements.";
  /*  Format minimum.  */
  if (p->n < 16 || p->n > order)     return "Armour n is out of range.";
  if (p->k < 1 || p->k >= p->n)      return "Armour k is out of range.";
  t2 = p->n - p->k;
  if (t2 & 1)                        return "Armour n - k must be even.";
  /*  The 2t roots must be distinct or the code does not have distance
      2t+1, and the Chien search must not map two degrees to one locator.
      Both follow from prim being a unit modulo 2^w - 1.  */
  pr = p->prim % order;
  if (pr == 0 || gcd_u32(pr, order) != 1)
    return "Armour prim must be invertible modulo 2^w - 1.";
  /* Version 2.0 fixes the generator for each field. */
  if (p->symbol_bits == 8) {
    if (p->fcr != 212 || p->prim != 11)
      return "Invalid GF(2^8) armour fcr/prim.";
  } else if (p->fcr != 1 || p->prim != 1) {
    return "Invalid GF(2^16) armour fcr/prim.";
  }
  if (p->depth < 1 || p->depth > (1u << 24))
    return "Armour depth must be in [1, 2^24].";
  frame = p->depth * (u64) p->n * (u64) (p->symbol_bits / 8);
  if (frame > (u64) (sz) -1)
    return "Armour frame does not fit in this host's address space.";
  return NULL;
}

/*  Tier dispatch.
    Preference order is the same ISA ranking gf.c resolves and for the
    same reason: the affine tier leads because one GF2P8AFFINEQB replaces
    two shuffles and an XOR, and everything below it is a width order.  */

typedef struct { const xpar_armour_kernels * k;  u32 need; } arm_tier;

static const arm_tier arm_tiers[] = {
#ifdef HAVE_GFNI
  { &xpar_armour_kernels_gfni256, XPAR_CPU_GFNI | XPAR_CPU_AVX2 },
#endif
#ifdef HAVE_AVX2
  { &xpar_armour_kernels_avx2,    XPAR_CPU_AVX2                 },
#endif
#ifdef HAVE_NEON
  { &xpar_armour_kernels_neon,    XPAR_CPU_NEON                 },
#endif
  { &xpar_armour_kernels_scalar,  0                             }
};

#define ARM_NTIERS ((int) ARRAY_LEN(arm_tiers))

static const xpar_armour_kernels * arm_k = &xpar_armour_kernels_scalar;
static int arm_cur = -1;

int xpar_armour_tier_count(void) { return ARM_NTIERS; }

const char * xpar_armour_tier_name(int tier) {
  return (tier < 0 || tier >= ARM_NTIERS) ? NULL : arm_tiers[tier].k->name;
}

bool xpar_armour_tier_usable(int tier) {
  if (tier < 0 || tier >= ARM_NTIERS) return false;
  return (xpar_cpu_features() & arm_tiers[tier].need) == arm_tiers[tier].need;
}

int xpar_armour_tier(void) {
  if (arm_cur < 0) xpar_armour_use_default_tier();
  return arm_cur;
}

bool xpar_armour_use_tier(int tier) {
  if (!xpar_armour_tier_usable(tier)) return false;
  arm_cur = tier;  arm_k = arm_tiers[tier].k;
  return true;
}

void xpar_armour_use_default_tier(void) {
  for (int i = 0; i < ARM_NTIERS; i++)
    if (xpar_armour_use_tier(i)) return;
}

#define ARM_REF_TILE  4096

static void run8(u8 * restrict p, sz stride, const xpar_gf8_coef * g, u32 c,
                 u32 lo, u32 hi) {
  while (c--) { *p ^= (u8) (g->tab[lo] ^ g->tab[hi]);  p += stride;  g++; }
}

static void sweep8(u8 * restrict p, sz stride, const xpar_gf8_coef * g, u32 c,
                   const u8 * restrict fb, sz n) {
  while (c--) {
    const u8 * tb = g->tab;  sz q;
    for (q = 0; q < n; q++)
      p[q] ^= (u8) (tb[fb[q] & 15u] ^ tb[16 + (fb[q] >> 4)]);
    p += stride;  g++;
  }
}

void xpar_armour_taps8_ref(u8 * restrict par, sz stride, u32 t2, u32 head,
                           const xpar_gf8_coef * gen,
                           const u8 * restrict fb, sz n) {
  u32 first = t2 - head;
  if ((u64) t2 * (u64) n > ARM_REF_TILE) {
    sweep8(par + (sz) head * stride, stride, gen, first, fb, n);
    sweep8(par, stride, gen + first, head, fb, n);
    return;
  }
  for (sz i = 0; i < n; i++) {
    u32 lo = fb[i] & 15u, hi = 16u + (fb[i] >> 4);
    run8(par + (sz) head * stride + i, stride, gen, first, lo, hi);
    run8(par + i, stride, gen + first, head, lo, hi);
  }
}

/*  Four nibble lookups per output byte, the same decomposition one field
    wider: tab[2q][i] and tab[2q+1][i] are the low and the high byte of
    (i << 4q) * c. The nibbles are split by the caller because all 2t
    taps share them.  */
static void nib16(u32 * q, u16 v) {
  q[0] = v & 15u;         q[1] = (v >> 4) & 15u;
  q[2] = (v >> 8) & 15u;  q[3] = v >> 12;
}
static u16 mul16_nib(const xpar_gf16_coef * m, const u32 * q) {
  return (u16) ((u32) (u8) (m->tab[0][q[0]] ^ m->tab[2][q[1]] ^
                            m->tab[4][q[2]] ^ m->tab[6][q[3]]) |
                ((u32) (u8) (m->tab[1][q[0]] ^ m->tab[3][q[1]] ^
                             m->tab[5][q[2]] ^ m->tab[7][q[3]]) << 8));
}

static void run16(u8 * restrict p, sz stride, const xpar_gf16_coef * g, u32 c,
                  const u32 * q) {
  while (c--) {
    xpar_wr16(p, (u16) (xpar_rd16(p) ^ mul16_nib(g, q)));
    p += stride;  g++;
  }
}

static void sweep16(u8 * restrict p, sz stride, const xpar_gf16_coef * g,
                    u32 c, const u8 * restrict fb, sz n) {
  while (c--) {
    sz q;
    for (q = 0; q + 2 <= n; q += 2) {
      u32 nb[4];
      nib16(nb, xpar_rd16(fb + q));
      xpar_wr16(p + q, (u16) (xpar_rd16(p + q) ^ mul16_nib(g, nb)));
    }
    p += stride;  g++;
  }
}

void xpar_armour_taps16_ref(u8 * restrict par, sz stride, u32 t2, u32 head,
                            const xpar_gf16_coef * gen,
                            const u8 * restrict fb, sz n) {
  u32 first = t2 - head;
  if ((u64) t2 * (u64) n > ARM_REF_TILE) {
    sweep16(par + (sz) head * stride, stride, gen, first, fb, n);
    sweep16(par, stride, gen + first, head, fb, n);
    return;
  }
  for (sz i = 0; i + 2 <= n; i += 2) {
    u32 q[4];
    nib16(q, xpar_rd16(fb + i));
    run16(par + (sz) head * stride + i, stride, gen, first, q);
    run16(par + i, stride, gen + first, head, q);
  }
}

void xpar_armour_horner8_ref(u8 * restrict syn, sz stride, u32 t2,
                             const xpar_gf8_coef * rt,
                             const u8 * restrict sym, sz n) {
  u32 j;  sz i;
  if ((u64) t2 * (u64) n > ARM_REF_TILE) {
    for (j = 0; j < t2; j++) {
      const u8 * tb = rt[j].tab;
      u8 * p = syn + (sz) j * stride;
      for (i = 0; i < n; i++)
        p[i] = (u8) (tb[p[i] & 15u] ^ tb[16 + (p[i] >> 4)] ^ sym[i]);
    }
    return;
  }
  for (i = 0; i < n; i++) {
    u8 * p = syn + i;  u8 s = sym[i];
    for (j = 0; j < t2; j++) {
      u32 v = *p;
      *p = (u8) (rt[j].tab[v & 15u] ^ rt[j].tab[16 + (v >> 4)] ^ s);
      p += stride;
    }
  }
}

void xpar_armour_horner16_ref(u8 * restrict syn, sz stride, u32 t2,
                              const xpar_gf16_coef * rt,
                              const u8 * restrict sym, sz n) {
  u32 q[4], j;  sz i;
  if ((u64) t2 * (u64) n > ARM_REF_TILE) {
    for (j = 0; j < t2; j++) {
      u8 * p = syn + (sz) j * stride;
      for (i = 0; i + 2 <= n; i += 2) {
        nib16(q, xpar_rd16(p + i));
        xpar_wr16(p + i, (u16) (mul16_nib(&rt[j], q) ^ xpar_rd16(sym + i)));
      }
    }
    return;
  }
  for (i = 0; i + 2 <= n; i += 2) {
    u8 * p = syn + i;  u16 s = xpar_rd16(sym + i);
    for (j = 0; j < t2; j++) {
      nib16(q, xpar_rd16(p));
      xpar_wr16(p, (u16) (mul16_nib(&rt[j], q) ^ s));
      p += stride;
    }
  }
}

/*  Construction.  */

/*  g(x) = prod (x - alpha^(fcr + j*prim)), built root by root so the
    O(t^2) cost is paid once per file rather than once per frame. The
    coefficient of x^(2t) is 1 and is not stored as a tap.  */
static void build_generator(xpar_armour * a) {
  u32 t2 = a->t2, i, j;
  a->gpoly = (u32 *) xpar_calloc(t2 + 1, sizeof(u32));
  a->gpoly[0] = 1;
  for (j = 0; j < t2; j++) {
    u32 r = f_alpha(a, (u32) (((u64) a->p.fcr +
                               (u64) j * a->p.prim) % a->order));
    a->gpoly[j + 1] = a->gpoly[j];
    for (i = j; i > 0; i--)
      a->gpoly[i] = a->gpoly[i - 1] ^ f_mul(a, a->gpoly[i], r);
    a->gpoly[0] = f_mul(a, a->gpoly[0], r);
  }
  xpar_assert(a->gpoly[t2] == 1);
}

/*  Preparing a coefficient costs a few hundred operations, so the 2t
    taps and the 2t syndrome roots are prepared here and reused for every
    frame of the file (gf.h says why the region kernels want it that
    way).  */
static void prepare_coefs(xpar_armour * a) {
  u32 t2 = a->t2, u;
  if (a->wb == 1) {
    a->gen8 = (xpar_gf8_coef *) xpar_calloc(t2, sizeof(xpar_gf8_coef));
    a->rt8  = (xpar_gf8_coef *) xpar_calloc(t2, sizeof(xpar_gf8_coef));
    for (u = 0; u < t2; u++) {
      xpar_gf8_prepare(&a->gen8[u], (u8) a->gpoly[t2 - 1 - u]);
      xpar_gf8_prepare(&a->rt8[u],
                       (u8) f_alpha(a, (u32) (((u64) a->p.fcr +
                                    (u64) u * a->p.prim) % a->order)));
    }
  } else {
    a->gen16 = (xpar_gf16_coef *) xpar_calloc(t2, sizeof(xpar_gf16_coef));
    a->rt16  = (xpar_gf16_coef *) xpar_calloc(t2, sizeof(xpar_gf16_coef));
    for (u = 0; u < t2; u++) {
      xpar_gf16_prepare(&a->gen16[u], (u16) a->gpoly[t2 - 1 - u]);
      xpar_gf16_prepare(&a->rt16[u],
                        (u16) f_alpha(a, (u32) (((u64) a->p.fcr +
                                     (u64) u * a->p.prim) % a->order)));
    }
  }
}

/*  Batch narrow frame lanes to SIMD width and zero-pad to a vector.  */

#define ARM_VLANE  2048   /*  Target virtual lane, bytes.  */
#define ARM_VMAX   4096   /*  Frame ceiling, so a tiny lane stays bounded.  */
#define ARM_PAD    128    /*  Guard lanes the batch buffers always carry.  */

static u32 batch_frames(const xpar_armour * a) {
  u64 v;
  /*  Scalar code does not benefit from batching.  */
  if (!xpar_strcmp(a->kern->name, "scalar")) return 1;
  if (a->lane >= ARM_VLANE) return 1;
  v = xpar_ceil_div(ARM_VLANE, (u64) a->lane);
  return (u32) (v > ARM_VMAX ? ARM_VMAX : v);
}

/*  Pad the active batch to a whole vector.  */
static sz virt_lane(const xpar_armour * a, u32 cnt) {
  u64 used = (u64) cnt * (u64) a->lane;
  if (!a->vstep) return (sz) used;
  return (sz) xpar_align_up(used, (u64) a->vstep);
}

xpar_armour * xpar_armour_new(const xpar_armour_params * p) {
  const char * why = xpar_armour_check(p);
  xpar_armour * a;
  if (why) FATAL("%s", why);
  if (arm_cur < 0) xpar_armour_use_default_tier();
  a = (xpar_armour *) xpar_calloc(1, sizeof(xpar_armour));
  a->p     = *p;
  a->kern  = arm_k;
  a->wb    = p->symbol_bits / 8;
  a->t2    = p->n - p->k;
  a->t     = a->t2 / 2;
  a->order = p->symbol_bits == 8 ? 255u : 65535u;
  a->lane  = (sz) p->depth * a->wb;
  build_generator(a);
  prepare_coefs(a);
  /*  GF(2^16) kernels step two vectors at a time; see ARM_BODY16.  */
  a->vstep = (sz) a->kern->vbytes * (a->wb == 1 ? 1 : 2);
  a->vb    = batch_frames(a);
  /*  Include the buffer guard.  */
  a->vlane = (sz) xpar_align_up(virt_lane(a, a->vb), ARM_PAD);
  a->par  = (u8 *) xpar_alloc_aligned((sz) a->t2 * a->vlane, 64);
  a->fb   = (u8 *) xpar_alloc_aligned(a->vlane, 64);
  a->syn  = (u8 *) xpar_alloc_aligned((sz) a->t2 * a->vlane, 64);
  a->gsym = (u8 *) xpar_alloc_aligned(a->vlane, 64);
  a->s    = (u32 *) xpar_calloc(a->t2, sizeof(u32));
  a->lam  = (u32 *) xpar_calloc(a->t2 + 2, sizeof(u32));
  a->bp   = (u32 *) xpar_calloc(a->t2 + 2, sizeof(u32));
  a->tmp  = (u32 *) xpar_calloc(a->t2 + 2, sizeof(u32));
  a->om   = (u32 *) xpar_calloc(a->t2 + 2, sizeof(u32));
  a->ev   = (u32 *) xpar_calloc(a->t2 + 2, sizeof(u32));
  a->step = (u32 *) xpar_calloc(a->t2 + 2, sizeof(u32));
  a->pos  = (u32 *) xpar_calloc(a->t2 + 2, sizeof(u32));
  a->val  = (u32 *) xpar_calloc(a->t2 + 2, sizeof(u32));
  return a;
}

void xpar_armour_free(xpar_armour * a) {
  if (!a) return;
  xpar_free(a->gpoly);  xpar_free(a->gen8);  xpar_free(a->gen16);
  xpar_free(a->rt8);    xpar_free(a->rt16);
  xpar_free_aligned(a->par);  xpar_free_aligned(a->fb);
  xpar_free_aligned(a->syn);  xpar_free_aligned(a->gsym);
  xpar_free(a->s);    xpar_free(a->lam);   xpar_free(a->bp);
  xpar_free(a->tmp);  xpar_free(a->om);    xpar_free(a->ev);
  xpar_free(a->step); xpar_free(a->pos);   xpar_free(a->val);
  xpar_free(a);
}

const xpar_armour_params * xpar_armour_params_of(const xpar_armour * a) {
  return &a->p;
}

u64 xpar_armour_frame_plain(const xpar_armour * a) {
  return a->p.depth * (u64) a->p.k * a->wb;
}
u64 xpar_armour_frame_disk(const xpar_armour * a) {
  return a->p.depth * (u64) a->p.n * a->wb;
}
u64 xpar_armour_size(const xpar_armour * a, u64 plain_length) {
  return xpar_ceil_div(plain_length, xpar_armour_frame_plain(a)) *
         xpar_armour_frame_disk(a);
}

u64 xpar_armour_burst(const xpar_armour * a) {
  return ((u64) a->t * a->p.depth - 1) * a->wb;
}

u64 xpar_armour_batch(const xpar_armour * a) { return a->vb; }

void xpar_armour_generator(const xpar_armour * a, u32 * g) {
  for (u32 i = 0; i <= a->t2; i++) g[i] = a->gpoly[i];
}

/*  Encoding.
    The parity register is t2 slots of `lane` bytes with logical slot u
    at physical slot (head + u) mod t2. A step of the recurrence shifts
    the register down by one and adds gen[u]*fb to every tap; the shift
    is the head advancing and the vacated slot being zeroed, so the whole
    step is 2t region MACs and no data movement.

    Message symbol i is the coefficient of x^(n-1-i).  */

/*  fb = data ^ top, and the vacated slot goes to zero. Below a vector's
    worth of bytes the two calls cost more than the work, and at D = 1
    they are two calls per input symbol; the open form is one pass and
    no call at all.  */
static void feedback(const xpar_gf_kernels * gk, u8 * fb, const u8 * dat,
                     u8 * top, sz lane) {
  if (lane > 16) { gk->xor3(fb, dat, top, lane);  xpar_memset(top, 0, lane); }
  else for (sz q = 0; q < lane; q++) {
    fb[q] = (u8) (dat[q] ^ top[q]);  top[q] = 0;
  }
}

/*  Gather or scatter one symbol position across a frame batch.  */

static void gather(sz lane, u8 * dst, const u8 * src, u64 fx, u32 cnt) {
  u32 f;
  if (lane == 1) for (f = 0; f < cnt; f++) dst[f] = src[(sz) (f * fx)];
  else if (lane == 2)
    for (f = 0; f < cnt; f++)
      xpar_wr16(dst + 2 * f, xpar_rd16(src + (sz) (f * fx)));
  else
    for (f = 0; f < cnt; f++)
      xpar_memcpy(dst + (sz) f * lane, src + (sz) (f * fx), lane);
}

static void scatter(sz lane, u8 * dst, u64 fx, u32 cnt, const u8 * src) {
  u32 f;
  if (lane == 1) for (f = 0; f < cnt; f++) dst[(sz) (f * fx)] = src[f];
  else if (lane == 2)
    for (f = 0; f < cnt; f++)
      xpar_wr16(dst + (sz) (f * fx), xpar_rd16(src + 2 * f));
  else
    for (f = 0; f < cnt; f++)
      xpar_memcpy(dst + (sz) (f * fx), src + (sz) f * lane, lane);
}

/*  Encode a frame batch with the single-frame recurrence.  */
static void encode_batch(const xpar_armour * a, u8 * base, u64 fx, u32 cnt) {
  const xpar_gf_kernels * gk = xpar_gf_active();
  sz lane = a->lane, used = (sz) cnt * lane, vl = virt_lane(a, cnt);
  u32 t2 = a->t2, head = 0, i, u;
  xpar_memset(a->par, 0, (sz) t2 * vl);
  xpar_memset(a->gsym + used, 0, vl - used);
  for (i = 0; i < a->p.k; i++) {
    gather(lane, a->gsym, base + (sz) i * lane, fx, cnt);
    feedback(gk, a->fb, a->gsym, a->par + (sz) head * vl, vl);
    if (++head == t2) head = 0;
    if (a->wb == 1)
      a->kern->taps8(a->par, vl, t2, head, a->gen8, a->fb, vl);
    else
      a->kern->taps16(a->par, vl, t2, head, a->gen16, a->fb, vl);
  }
  for (u = 0; u < t2; u++)
    scatter(lane, base + (sz) (a->p.k + u) * lane, fx, cnt,
            a->par + (sz) ((head + u) % t2) * vl);
}

static void encode_run(const xpar_armour * a, u8 * frames, u64 count) {
  u64 fx = xpar_armour_frame_disk(a), done = 0;
  while (done < count) {
    u64 left = count - done;
    u32 cnt = (u32) (left < (u64) a->vb ? left : (u64) a->vb);
    encode_batch(a, frames + done * fx, fx, cnt);
    done += cnt;
  }
}

void xpar_armour_encode_frame(const xpar_armour * a, u8 * frame) {
  encode_batch(a, frame, xpar_armour_frame_disk(a), 1);
}

void xpar_armour_encode(const xpar_armour * a, u8 * out,
                        const u8 * plain, u64 plain_length) {
  u64 fd = xpar_armour_frame_plain(a), fx = xpar_armour_frame_disk(a);
  u64 frames = xpar_ceil_div(plain_length, fd), f, off = 0;
  for (f = 0; f < frames; f++) {
    u8 * fr = out + f * fx;
    u64 take = plain_length - off < fd ? plain_length - off : fd;
    xpar_memcpy(fr, plain + off, (sz) take);
    if (take < fd) xpar_memset(fr + take, 0, (sz) (fd - take));
    off += take;
  }
  encode_run(a, out, frames);
}

void xpar_armour_extract(const xpar_armour * a, u8 * plain, u64 plain_length,
                         const u8 * region) {
  u64 fd = xpar_armour_frame_plain(a), fx = xpar_armour_frame_disk(a);
  u64 frames = xpar_ceil_div(plain_length, fd), f, off = 0;
  for (f = 0; f < frames; f++) {
    u64 take = plain_length - off < fd ? plain_length - off : fd;
    xpar_memcpy(plain + off, region + f * fx, (sz) take);
    off += take;
  }
}

/*  Syndromes.
    S_j = c(alpha^(fcr + j*prim)), by Horner over the codeword. The 2t
    recurrences are independent of each other and, across the D codewords
    of a frame, independent again, so one symbol position is 2t region
    operations and the whole pass is 2t byte-operations per input byte.  */
static void batch_syndromes(const xpar_armour * a, const u8 * base, u64 fx,
                            u32 cnt) {
  sz lane = a->lane, used = (sz) cnt * lane, vl = virt_lane(a, cnt);
  u32 i;
  xpar_memset(a->syn, 0, (sz) a->t2 * vl);
  xpar_memset(a->gsym + used, 0, vl - used);
  for (i = 0; i < a->p.n; i++) {
    gather(lane, a->gsym, base + (sz) i * lane, fx, cnt);
    if (a->wb == 1)
      a->kern->horner8 (a->syn, vl, a->t2, a->rt8,  a->gsym, vl);
    else
      a->kern->horner16(a->syn, vl, a->t2, a->rt16, a->gsym, vl);
  }
}

static bool region_zero(const u8 * p, sz n) {
  sz i;  u8 acc = 0;
  for (i = 0; i < n; i++) acc |= p[i];
  return acc == 0;
}

/*  Berlekamp-Massey.
    Returns the degree L of the connection polynomial in a->lam. A
    codeword within capacity gives L <= t and L distinct locator roots;
    anything else is past capacity and the caller rejects it.  */
static u32 berlekamp(const xpar_armour * a) {
  u32 t2 = a->t2, L = 0, m = 1, bb = 1, r, i;
  sz words = (sz) (t2 + 2) * sizeof(u32);
  xpar_memset(a->lam, 0, words);  xpar_memset(a->bp, 0, words);
  a->lam[0] = 1;  a->bp[0] = 1;
  for (r = 0; r < t2; r++) {
    u32 d = a->s[r], c;
    for (i = 1; i <= L; i++) d ^= f_mul(a, a->lam[i], a->s[r - i]);
    if (!d) { m++;  continue; }
    c = f_div(a, d, bb);
    xpar_memcpy(a->tmp, a->lam, words);
    for (i = 0; i + m <= t2 + 1; i++)
      if (a->bp[i]) a->lam[i + m] ^= f_mul(a, c, a->bp[i]);
    if (2 * L <= r) {
      L = r + 1 - L;
      xpar_memcpy(a->bp, a->tmp, words);
      bb = d;  m = 1;
    } else m++;
  }
  return L;
}

/*  Chien.
    Sweeps the degrees a *shortened* codeword actually occupies, r in
    [0, n), which is what makes shortening free: a locator root outside
    that window is simply never found, the root count comes up short, and
    the codeword is rejected instead of being corrected at a position
    that does not exist.

    Evaluated incrementally. ev[i] holds lam[i] * y^i for the current
    y = alpha^(-prim*r), so a step is one multiply per term by the fixed
    alpha^(-prim*i) rather than an exponentiation per position.  */
static u32 chien(const xpar_armour * a, u32 L) {
  u32 cnt = 0, r, i, pinv = a->order - (a->p.prim % a->order);
  for (i = 0; i <= L; i++) {
    a->ev[i]   = a->lam[i];
    a->step[i] = f_alpha_mul(a, pinv, i);
  }
  for (r = 0; r < a->p.n; r++) {
    u32 sum = 0;
    for (i = 0; i <= L; i++) sum ^= a->ev[i];
    if (!sum) {
      if (cnt == L) return L + 1;   /*  More roots than degree: reject.  */
      a->pos[cnt++] = r;
    }
    for (i = 1; i <= L; i++) a->ev[i] = f_mul(a, a->ev[i], a->step[i]);
  }
  return cnt;
}

/*  Forney.
    With S_j = sum Y_l X_l^j, X_l = alpha^(prim*r_l) and
    Y_l = v_l * alpha^(fcr*r_l), the standard identities give
    Omega = S*Lambda mod x^(2t) and Y_l = X_l * Omega(1/X_l) /
    Lambda'(1/X_l). The stored magnitude is then v_l, with the fcr twist
    divided back out, which is where fcr stops mattering to anything
    downstream.

    False, so the caller rejects the codeword: a zero denominator, or a
    zero magnitude, which is what a spurious locator root produces.  */
static bool forney(const xpar_armour * a, u32 L) {
  u32 i, j, l;
  for (j = 0; j < L; j++) {
    u32 acc = 0;
    for (i = 0; i <= j && i <= L; i++) acc ^= f_mul(a, a->lam[i], a->s[j - i]);
    a->om[j] = acc;
  }
  for (l = 0; l < L; l++) {
    u32 r = a->pos[l];
    u32 x = f_alpha_mul(a, a->p.prim % a->order, r);
    u32 y = f_inv(a, x), yp = 1, num = 0, den = 0;
    for (j = 0; j < L; j++) {
      num ^= f_mul(a, a->om[j], yp);
      yp   = f_mul(a, yp, y);
    }
    yp = 1;
    for (i = 1; i <= L; i += 2) {
      den ^= f_mul(a, a->lam[i], yp);
      yp   = f_mul(a, yp, f_mul(a, y, y));
    }
    if (!den) return false;
    a->val[l] = f_div(a, f_mul(a, x, f_div(a, num, den)),
                      f_alpha_mul(a, a->p.fcr % a->order, r));
    if (!a->val[l]) return false;
  }
  return true;
}

/*  Re-derive all 2t syndromes from the errors the decoder claims and
    require every one of them to agree. Past capacity the key equation
    still has a solution and Chien still finds roots, so without this the
    decoder would happily write t plausible corrections into a codeword
    carrying t+1 errors. It costs 2t multiplies per located error against
    the 2t*n of the syndrome pass that preceded it.

    It cannot catch a codeword that has been carried onto a genuinely
    different valid codeword, because that one's syndromes are correct.
    That residue is what the caller's tag is for.  */
static bool syndromes_agree(const xpar_armour * a, u32 L) {
  u32 j, l;
  for (l = 0; l < L; l++) {
    u32 r = a->pos[l];
    a->ev[l]   = f_mul(a, a->val[l], f_alpha_mul(a, a->p.fcr % a->order, r));
    a->step[l] = f_alpha_mul(a, a->p.prim % a->order, r);
  }
  for (j = 0; j < a->t2; j++) {
    u32 acc = 0;
    for (l = 0; l < L; l++) {
      acc ^= a->ev[l];
      a->ev[l] = f_mul(a, a->ev[l], a->step[l]);
    }
    if (acc != a->s[j]) return false;
  }
  return true;
}

/*  One codeword of a frame, its syndromes already in a->syn. Returns the
    number of symbols corrected, or -1 when the codeword is past
    capacity; nothing is written to the frame unless every stage
    agreed.  */
static int decode_one(const xpar_armour * a, u8 * frame, u32 d,
                      const u8 * syn, sz stride) {
  sz lane = a->lane;  u32 j, L, cnt;
  bool zero = true;
  for (j = 0; j < a->t2; j++) {
    a->s[j] = sym_rd(a, syn + (sz) j * stride + (sz) d * a->wb);
    if (a->s[j]) zero = false;
  }
  if (zero) return 0;
  L = berlekamp(a);
  if (L == 0 || L > a->t) return -1;
  cnt = chien(a, L);
  if (cnt != L) return -1;
  if (!forney(a, L)) return -1;
  if (!syndromes_agree(a, L)) return -1;
  for (j = 0; j < L; j++) {
    u8 * p = frame + (sz) (a->p.n - 1 - a->pos[j]) * lane + (sz) d * a->wb;
    sym_wr(a, p, sym_rd(a, p) ^ a->val[j]);
  }
  return (int) L;
}

static void stat_add(xpar_armour_stat * st, int e) {
  if (!st) return;
  st->codewords++;
  if (e < 0) { st->failed++;  return; }
  if (e == 0) st->clean++;  else { st->corrected++;  st->symbols += (u32) e; }
  if ((u32) e > st->worst) st->worst = (u32) e;
  if (st->hist && (u32) e < st->hist_len) st->hist[(u32) e]++;
}

/*  One frame's t2 syndrome lanes inside a batch, which are strided.  */
static bool frame_clean(const xpar_armour * a, const u8 * syn, sz stride) {
  u32 j;  sz q;  u8 acc = 0;
  for (j = 0; j < a->t2; j++, syn += stride)
    for (q = 0; q < a->lane; q++) acc |= syn[q];
  return acc == 0;
}

static void stat_clean(xpar_armour_stat * st, u64 codewords) {
  if (!st) return;
  st->codewords += codewords;  st->clean += codewords;
  if (st->hist && st->hist_len) st->hist[0] += codewords;
}

/*  Count a frame as corrected only if all its codewords succeeded.  */
static void decode_run(const xpar_armour * a, u8 * frames, u64 count,
                       xpar_armour_stat * st, bool * any_bad,
                       bool * any_fixed) {
  u64 fx = xpar_armour_frame_disk(a), done = 0;
  *any_bad = false;  *any_fixed = false;
  while (done < count) {
    u64 left = count - done;
    u32 cnt = (u32) (left < (u64) a->vb ? left : (u64) a->vb);
    u8 * base = frames + done * fx;
    sz vl = virt_lane(a, cnt);
    u32 f, d;
    batch_syndromes(a, base, fx, cnt);
    if (st) st->frames += cnt;
    if (region_zero(a->syn, (sz) a->t2 * vl)) {
      stat_clean(st, (u64) cnt * a->p.depth);
      done += cnt;  continue;
    }
    for (f = 0; f < cnt; f++) {
      const u8 * fs = a->syn + (sz) f * a->lane;
      bool bad = false, fixed = false;
      if (frame_clean(a, fs, vl)) { stat_clean(st, a->p.depth);  continue; }
      for (d = 0; d < a->p.depth; d++) {
        int e = decode_one(a, base + (u64) f * fx, d, fs, vl);
        stat_add(st, e);
        if (e < 0) bad = true;  else if (e > 0) fixed = true;
      }
      if (bad) *any_bad = true;  else if (fixed) *any_fixed = true;
    }
    done += cnt;
  }
}

xpar_armour_status xpar_armour_decode_frames(const xpar_armour * a,
                                             u8 * frames, u64 count,
                                             xpar_armour_stat * st) {
  bool bad, fixed;
  decode_run(a, frames, count, st, &bad, &fixed);
  if (bad)   return XPAR_ARMOUR_FAILED;
  if (fixed) return XPAR_ARMOUR_CORRECTED;
  return XPAR_ARMOUR_CLEAN;
}

xpar_armour_status xpar_armour_decode_frame(const xpar_armour * a, u8 * frame,
                                            xpar_armour_stat * st) {
  return xpar_armour_decode_frames(a, frame, 1, st);
}

xpar_armour_status xpar_armour_decode(const xpar_armour * a,
                                      u8 * region, u64 region_length,
                                      u8 * plain, u64 plain_length,
                                      xpar_armour_check_fn check,
                                      const void * ctx,
                                      xpar_armour_stat * st) {
  u64 fx = xpar_armour_frame_disk(a);
  u64 frames = xpar_ceil_div(plain_length, xpar_armour_frame_plain(a));
  bool fixed = false;
  xpar_assert(check != NULL);
  xpar_assert(region_length == frames * fx);
  xpar_armour_extract(a, plain, plain_length, region);
  if (check(ctx, plain, plain_length)) return XPAR_ARMOUR_CLEAN;
  /*  A frame that reports FAILED is not decisive either way. Its damage
      may have been confined to parity, or to a codeword whose plaintext
      another frame's success makes whole; only the tag knows.  */
  { bool bad;  decode_run(a, region, frames, st, &bad, &fixed); }
  xpar_armour_extract(a, plain, plain_length, region);
  if (!check(ctx, plain, plain_length)) return XPAR_ARMOUR_FAILED;
  return fixed ? XPAR_ARMOUR_CORRECTED : XPAR_ARMOUR_CLEAN;
}
