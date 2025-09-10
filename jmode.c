/*
   Copyright (C) 2022-2024 Kamila Szewczyk

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "jmode.h"
#include "crc32c.h"
#include "platform.h"

#if defined(XPAR_OPENMP)
  #include <omp.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

// ============================================================================
//  Reed-Solomon code parameters (223 bytes of input, 32 bytes of parity).
// ============================================================================
#define K 223
#define N 255
#define T 32

// ============================================================================
//  Implementation of Reed-Solomon codes. Follows the BCH view. Original code
//  was written by Phil Karn, KA9Q, in 1999. This is a modified version due to
//  Kamila Szewczyk which exhibits significantly better performance.
// ============================================================================
static u8 LOG[256], EXP[256], PROD[256][256], DP[256][256];
u8 PROD_GEN[256][32];
void jmode_gf256_gentab(u8 poly) {
  for (int l = 0, b = 1; l < 255; l++) {
    LOG[b] = l;  EXP[l] = b;
    if ((b <<= 1) >= 256)
      b = (b - 256) ^ poly;
  }
  LOG[0] = 255; EXP[255] = 0;
  Fi(256, Fj(256,
    PROD[i][j] = (i && j) ? EXP[(LOG[i] + LOG[j]) % 255] : 0,
    DP[i][j] = (i != 255 && j) ? EXP[(i + LOG[j]) % 255] : 0))
  static const u8 gen[T] = {
    1, 91, 127, 86, 16, 30, 13, 235, 97, 165, 8, 42, 54, 86, 171, 32,
    113, 32, 171, 86, 54, 42, 8, 165, 97, 235, 13, 30, 16, 86, 127, 91
  };
  Fi(256, Fj(T, PROD_GEN[i][j] = PROD[i][gen[j]]))
}
static u8 gf256_div(u8 a, u8 b) {
  if (!a || !b) return 0;
  int d = LOG[a] - LOG[b];
  return EXP[d < 0 ? d + 255 : d];
}
#if defined(XPAR_X86_64)
#ifdef HAVE_FUNC_ATTRIBUTE_SYSV_ABI
  #define EXTERNAL_ABI __attribute__((sysv_abi))
#else
  #define EXTERNAL_ABI
#endif

extern EXTERNAL_ABI int xpar_x86_64_cpuflags(void);
extern EXTERNAL_ABI void rse32_x86_64_avx512(u8 data[K], u8 out[N]);
extern EXTERNAL_ABI void rse32_x86_64_generic(u8 data[K], u8 out[N]);
void rse32(u8 data[K], u8 out[N]) {
  static int cpuflags = -1;
  if (cpuflags == -1) cpuflags = xpar_x86_64_cpuflags();
  if (cpuflags & 0xC) rse32_x86_64_avx512(data, out);
  else rse32_x86_64_generic(data, out);
}
#else
void rse32(u8 data[K], u8 out[N]) {
  memset(out + K, 0, N - K);
  for (int i = K - 1; i >= 0; i--) {
    u8 x = data[i] ^ out[K + T - 1];
    for (int j = T - 1; j > 0; j--)
      out[K + j] = out[K + j - 1] ^ PROD_GEN[x][j];
    out[K] = PROD_GEN[x][0];
  }
  memcpy(out, data, K);
}
#endif
int rsd32(u8 data[N]) {
  int deg_lambda, el, deg_omega = 0;
  int i, j, r, k, syn_error, count;
  u8 q, tmp, num1, den, discr_r;
  u8 lambda[T + 1] = { 0 }, omega[T + 1] = { 0 }, eras_pos[T] = { 0 };
  u8 t[T + 1], s[T], root[T], reg[T + 1] = { 0 };
  u8 b_backing[3 * T + 1] = { 0 }, * b = b_backing + 2 * T;
  memset(s, data[0], T);
  // Fast syndrome computation: idea discovered by Marshall Lochbaum.
  for (int jb = 0; jb < 51; jb++) {
    u8 a5 = 255, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
    for (j = jb; j < 255; j += 51) {
      if (j == 0 || !data[j]) continue;
      tmp = data[j];
      tmp = PROD[EXP[(212 * j) % 255]][tmp];
      u8 a1 = EXP[(11 * j) % 255]; // a_j1
      u8 a2 = PROD[a1][a1];        // a_j2
      if (a5 == 255) a5 = PROD[a2][PROD[a1][a2]];
      t1 ^=                tmp;  // t1 = sum of t_j1, j = jb (mod 51)
      t2 ^=       PROD[a1][tmp]; // etc.
      t3 ^= tmp = PROD[a2][tmp];
      t4 ^=       PROD[a1][tmp];
      t5 ^=       PROD[a2][tmp];
    }
    if (a5 == 255) continue; // No j values do anything (unlikely)
    for (i = 0; ; i += 5) {
      s[i ] ^= t1; t1 = PROD[a5][t1];
      s[i+1] ^= t2; t2 = PROD[a5][t2];
      if (i+2 == 32) break;
      s[i+2] ^= t3; t3 = PROD[a5][t3];
      s[i+3] ^= t4; t4 = PROD[a5][t4];
      s[i+4] ^= t5; t5 = PROD[a5][t5];
    }
  }
  syn_error = 0; Fi(T, syn_error |= s[i])
  if (!syn_error) return 0;
  lambda[0] = 1;  r = el = 0;  memcpy(b, lambda, T + 1);
  while (++r <= T) {
    for (discr_r = 0, i = 0; i < r; i++)
      discr_r ^= PROD[lambda[i]][s[r - i - 1]];
    if (!discr_r) --b; else {
      t[0] = lambda[0];
      Fi(T, t[i + 1] = lambda[i + 1] ^ PROD[discr_r][b[i]])
      if (2 * el <= r - 1) {
        el = r - el;
        Fi(T + 1, b[i] = gf256_div(lambda[i], discr_r))
      } else --b;
      memcpy(lambda, t, T + 1);
    }
  }
  deg_lambda = 0;
  Fi(T + 1, if (lambda[i]) deg_lambda = i, reg[i] = lambda[i])
  for (count = 0, i = 1, k = 139; i <= 255; i++, k = (k + 139) % 255) {
    for (q = 1, j = deg_lambda; j > 0; j--)
      q ^= reg[j] = DP[j][reg[j]];
    if (q) continue;
    root[count] = i, eras_pos[count] = k;
    if (++count == deg_lambda) break; // Early exit.
  }
  if (deg_lambda != count) return -1;
  Fi(T,
    for (tmp = 0, j = MIN(deg_lambda, i); j >= 0; j--)
      tmp ^= PROD[s[i - j]][lambda[j]];
    if (tmp) deg_omega = i, omega[i] = tmp)
  for (j = count - 1; j >= 0; j--) {
    for (num1 = 0, i = deg_omega; i >= 0; i--)
      num1 ^= DP[(i * root[j]) % 255][omega[i]];
    for (den = 0, i = MIN(deg_lambda, T - 1) & ~1; i >= 0; i -= 2)
      den ^= DP[(i * root[j]) % 255][lambda[i + 1]];
    if (den == 0) return -1;
    data[eras_pos[j]] ^= gf256_div(DP[(root[j] * 111) % 255][num1], den);
  }
  return count;
}

// ============================================================================
//  Processing. We apply a few strategies that depend on some specifics of the
//  process at hand:
//  - If we can map the file to memory but can not perform parallel encoding,
//    we encode the data in a serial fashion (3).
//  - If the file can not be mapped, we assume that the output also can not be
//    mapped (4).
// ============================================================================
static int compute_interlacing_bs(int ifactor) {
  switch (ifactor) {
    default: case 1: return 1; break;
    case 2: return N; break;
    case 3: return N * N; break;
  }
}
static void trans2D(u8 * restrict in, u8 * restrict out) {
  Fi(N, Fj(N, out[j * N + i] = in[i * N + j]))
}
static void trans3D(u8 * restrict in, u8 * restrict out) {
#if defined(XPAR_OPENMP)
  #pragma omp parallel for
#endif
  Fi(N, Fj(N, Fk(N, out[k * N * N + j * N + i] = in[i * N * N + j * N + k])))
}
static void do_interlacing(u8 * restrict in, u8 * restrict out, int ifactor) {
  switch (ifactor) {
    case 1: memcpy(out, in, N); break;
    case 2: trans2D(in, out); break;
    case 3: trans3D(in, out); break;
  }
}
static void write_header(FILE * des, int ifactor) {
  u8 h[K] = { 0 }, out[N];
  h[0] = 'X'; h[1] = 'P'; h[2] = XPAR_MAJOR; h[3] = XPAR_MINOR;
  h[4] = ifactor + '0';
  rse32(h, out); xfwrite(h, 5, des); xfwrite(out + K, N - K, des);
}
static int parse_header(u8 out[N], int force, int ifactor_override) {
  if (out[0] != 'X' || out[1] != 'P')
    FATAL_UNLESS("Invalid header.", !force);
  out[0] = 'X'; out[1] = 'P'; memset(out + 5, 0, K - 5);
  if(rsd32(out) < 0)
    FATAL_UNLESS("Invalid header.", !force);
  int ifactor = out[4] - '0';
  if (ifactor < 1 || ifactor > 3) {
    FATAL_UNLESS("Invalid header.", !force);
    if (force) return ifactor_override;
  }
  return ifactor;
}
static int read_header(FILE * des, int force, int ifactor_override) {
  u8 out[N]; xfread(out, 5, des); xfread(out + K, N - K, des);
  return parse_header(out, force, ifactor_override);
}
#ifdef XPAR_ALLOW_MAPPING
static int read_header_from_map(mmap_t map, int force, int ifactor_override) {
  if (map.size < 5 + N - K)
    FATAL("Truncated file.");
  u8 out[N]; memcpy(out, map.map, 5); memcpy(out + K, map.map + 5, N - K);
  return parse_header(out, force, ifactor_override);
}
#endif
typedef struct { u32 bytes, crc; } block_hdr;
static void write_block_header(FILE * des, block_hdr h) {
  u8 b[8]; b[0] = 'X';
  if (h.bytes > 0xFFFFFF)
    FATAL("Could not write the header: block too big.");
  b[1] = h.bytes >> 16; b[2] = h.bytes >> 8; b[3] = h.bytes;
  b[4] = h.crc >> 24; b[5] = h.crc >> 16; b[6] = h.crc >> 8; b[7] = h.crc;
  xfwrite(b, 8, des);
}
static block_hdr parse_block_header(u8 b[8], bool force) {
  block_hdr h;  bool valid = b[0] == 'X';
  if (!valid) {
    FATAL_UNLESS("Invalid block header.", !force);
    h.bytes = 0xFFFFFF; h.crc = 0; return h;
  } else {
    h.bytes = (b[1] << 16) | (b[2] << 8) | b[3];
    h.crc = (b[4] << 24) | (b[5] << 16) | (b[6] << 8) | b[7];
    return h;
  }
}
static void encode4(FILE * in, FILE * out, int ifactor) {
  notty(out);
  u8 * in_buffer, * o1, * o2;
  int ibs = compute_interlacing_bs(ifactor);
  in_buffer = xmalloc(ibs * K), o1 = xmalloc(ibs * N), o2 = xmalloc(ibs * N);
  block_hdr bhdr;  write_header(out, ifactor);
  for (size_t n; n = xfread(in_buffer, ibs * K, in); ) {
    if(n < ibs * K) memset(in_buffer + n, 0, ibs * K - n);
  #if defined(XPAR_OPENMP)
    #pragma omp parallel for if(ifactor == 3)
  #endif
    Fi(ibs, rse32(in_buffer + i * K, o1 + i * N));
    do_interlacing(o1, o2, ifactor);
    xfwrite(o2, ibs * N, out);
    bhdr.bytes = n; bhdr.crc = crc32c(in_buffer, n);
    write_block_header(out, bhdr);
  }
  free(in_buffer), free(o1), free(o2); xfclose(out);
}
#ifdef XPAR_ALLOW_MAPPING
static void encode3(mmap_t in, FILE * out, int ifactor) {
  notty(out);
  u8 * in_buffer, * o1, * o2;
  int ibs = compute_interlacing_bs(ifactor);
  in_buffer = xmalloc(ibs * K), o1 = xmalloc(ibs * N), o2 = xmalloc(ibs * N);
  block_hdr bhdr;  write_header(out, ifactor);
  for (sz n;
       n = MIN(in.size, ibs * K), memcpy(in_buffer, in.map, n), n;
       in.size -= n, in.map += n) {
    if(n < ibs * K) memset(in_buffer + n, 0, ibs * K - n);
  #if defined(XPAR_OPENMP)
    #pragma omp parallel for if(ifactor == 3)
  #endif
    Fi(ibs, rse32(in_buffer + i * K, o1 + i * N));
    do_interlacing(o1, o2, ifactor);
    xfwrite(o2, ibs * N, out);
    bhdr.bytes = n; bhdr.crc = crc32c(in_buffer, n);
    write_block_header(out, bhdr);
  }
  free(in_buffer), free(o1), free(o2); xfclose(out);
}
#endif
static void decode4(FILE * in, FILE * out, int force, int ifactor_override,
             bool quiet, bool verbose) {
  notty(in);
  u8 * in1, * in2, * out_buffer;  int laces = 0, ecc = 0;
  block_hdr bhdr; u8 tmp[8];
  int ifactor = read_header(in, force, ifactor_override);
  sz ibs = compute_interlacing_bs(ifactor);
  in1 = xmalloc(ibs * N), in2 = xmalloc(ibs * N), out_buffer = xmalloc(ibs * K);
  for (sz n; n = xfread(in1, ibs * N, in); laces++) {
    if(n < ibs * N) {
      if (!quiet)
        fprintf(stderr, "Short read, lace %u (bytes %zu-%zu).\n",
          laces, laces * ibs * N, laces * ibs * N + n - 1);
      if (!force) exit(1);
      memset(in1 + n, 0, ibs * N - n);
    }
    if(xfread(tmp, 8, in) != 8) {
      if (!quiet)
        fprintf(stderr,
          "Short read (block header), lace %u (bytes %zu-%zu).\n",
          laces, laces * ibs * N, laces * ibs * N + n - 1);
      if (!force) exit(1);
    }
    bhdr = parse_block_header(tmp, force);
    do_interlacing(in1, in2, ifactor);
  #if defined(XPAR_OPENMP)
    #pragma omp parallel for if(ifactor == 3)
  #endif
    Fi(ibs,
      int n = rsd32(in2 + i * N);
      if (n < 0) {
        // POSIX requires single I/O function calls to be thread-safe.
        {
        const unsigned lace_ibs = laces * ibs + i;
        if (!quiet)
          fprintf(stderr,
            "Block %u (lace %u, bytes %u-%u) irrecoverable.\n",
            lace_ibs, laces, lace_ibs * N, lace_ibs * N + N - 1);
        if (!force) exit(1);
        }
      } else ecc += n;
      memcpy(out_buffer + i * K, in2 + i * N, K);
    )
    sz size = MIN(ibs * K, bhdr.bytes);
    u32 crc = crc32c(out_buffer, size);
    if (crc != bhdr.crc) {
      if (!quiet)
        fprintf(stderr, "CRC mismatch, block %zu (lace %u, bytes %zu-%zu).\n",
          laces * ibs, laces, laces * ibs * N, laces * ibs * N + size - 1);
      if (!force) exit(1);
    }
    xfwrite(out_buffer, size, out);
  }
  free(in1), free(in2), free(out_buffer); xfclose(out);
  if (!quiet && verbose)
    fprintf(stderr, "Decoded %u laces, %u errors corrected.\n", laces, ecc);
}
#ifdef XPAR_ALLOW_MAPPING
static void decode3(mmap_t in, FILE * out, int force, int ifactor_override,
             bool quiet, bool verbose) {
  u8 * in1, * in2, * out_buffer;  int laces = 0, ecc = 0;
  block_hdr bhdr; u8 tmp[8];
  int ifactor = read_header_from_map(in, force, ifactor_override);
  in.size -= 5 + N - K; in.map += 5 + N - K; // Skip the header.
  sz ibs = compute_interlacing_bs(ifactor);
  in1 = xmalloc(ibs * N), in2 = xmalloc(ibs * N), out_buffer = xmalloc(ibs * K);
  for (sz n;
      n = MIN(in.size, ibs * N), memcpy(in1, in.map, n),
          in.size -= n, in.map += n, n
      ; laces++) {
    if(n < ibs * N) {
      if (!quiet)
        fprintf(stderr, "Short read, lace %u (bytes %zu-%zu).\n",
          laces, laces * ibs * N, laces * ibs * N + n - 1);
      if (!force) exit(1);
      memset(in1 + n, 0, ibs * N - n);
    }
    if (in.size < 8) {
      if (!quiet)
        fprintf(stderr,
          "Short read (block header), lace %u (bytes %zu-%zu).\n",
          laces, laces * ibs * N, laces * ibs * N + n - 1);
      if (!force) exit(1);
      memcpy(tmp, in.map, in.size); in.map += in.size; in.size = 0;
    } else {
      memcpy(tmp, in.map, 8); in.size -= 8; in.map += 8;
    }
    bhdr = parse_block_header(tmp, force);
    do_interlacing(in1, in2, ifactor);
  #if defined(XPAR_OPENMP)
    #pragma omp parallel for if(ifactor == 3)
  #endif
    Fi(ibs,
      int n = rsd32(in2 + i * N);
      if (n < 0) {
        {
        const unsigned lace_ibs = laces * ibs + i;
        if (!quiet)
          fprintf(stderr,
            "Block %u (lace %u, bytes %u-%u) irrecoverable.\n",
            lace_ibs, laces, lace_ibs * N, lace_ibs * N + N - 1);
        if (!force) exit(1);
        }
      } else ecc += n;
      memcpy(out_buffer + i * K, in2 + i * N, K);
    )
    sz size = MIN(ibs * K, bhdr.bytes);
    u32 crc = crc32c(out_buffer, size);
    if (crc != bhdr.crc) {
      if (!quiet)
        fprintf(stderr, "CRC mismatch, block %zu (lace %u, bytes %zu-%zu).\n",
          laces * ibs, laces, laces * ibs * N, laces * ibs * N + size - 1);
      if (!force) exit(1);
    }
    xfwrite(out_buffer, size, out);
  }
  free(in1), free(in2), free(out_buffer); xfclose(out);
  if (!quiet && verbose)
    fprintf(stderr, "Decoded %u laces, %u errors corrected.\n", laces, ecc);
}
#endif

static struct stat validate_file(const char * filename) {
  struct stat st;
  if (stat(filename, &st) == -1) FATAL_PERROR("stat");
  if (S_ISDIR(st.st_mode)) FATAL("Input is a directory.");
  return st;
}
static FILE * open_output(joint_options_t o) {
  FILE * out = stdout;
  if (o.output_name) {
    struct stat st;
    memset(&st, 0, sizeof(struct stat));
    int exists = stat(o.output_name, &st);
    if ((st.st_size || S_ISDIR(st.st_mode)) && exists != -1 && !o.force)
      FATAL("Output file `%s' exists and is not empty.", o.output_name);
    if (!(out = fopen(o.output_name, "wb"))) FATAL_PERROR("fopen");
  }
  return out;
}
void do_joint_encode(joint_options_t o) {
  FILE * out = open_output(o), * in = stdin;
  if (o.input_name) {
    struct stat st = validate_file(o.input_name);
    if(!o.no_map) {
      #if defined(XPAR_ALLOW_MAPPING)
      mmap_t map = xpar_map(o.input_name);
      if (map.map) {
        encode3(map, out, o.interlacing);
        xpar_unmap(&map);
        return;
      }
      #endif
    }
    if (!(in = fopen(o.input_name, "rb"))) FATAL_PERROR("fopen");
  }
  encode4(in, out, o.interlacing);
}
void do_joint_decode(joint_options_t o) {
  FILE * out = open_output(o), * in = stdin;
  if (o.input_name) {
    struct stat st = validate_file(o.input_name);
    if(!o.no_map) {
      #if defined(XPAR_ALLOW_MAPPING)
      mmap_t map = xpar_map(o.input_name);
      if (map.map) {
        decode3(map, out, o.force, o.interlacing, o.quiet, o.verbose);
        xpar_unmap(&map);
        return;
      }
      #endif
    }
    if (!(in = fopen(o.input_name, "rb"))) FATAL_PERROR("fopen");
  }
  decode4(in, out, o.force, o.interlacing, o.quiet, o.verbose);
}