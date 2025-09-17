/*
   Copyright (C) 2022-2025 Kamila Szewczyk

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

#include "vmode.h"

#include <sys/stat.h>

#if defined(XPAR_OPENMP)
  #include <omp.h>
#endif

static u8 LOG[256], EXP[256], PROD[256][256];
void smode_gf256_gentab(u8 poly) {
  for (int l = 0, b = 1; l < 255; l++) {
    LOG[b] = l;  EXP[l] = b;
    if ((b <<= 1) >= 256)
      b = (b - 256) ^ poly;
  }
  Fi0(256, 1, Fj0(256, 1,
      PROD[i][j] = EXP[(LOG[i] + LOG[j]) % 255]))
}
static u8 gf256_div(u8 a, u8 b) {
  if (!a || !b) return 0;
  int d = LOG[a] - LOG[b];
  return EXP[d < 0 ? d + 255 : d];
}
static u8 gf256_exp(u8 a, int n) {
  if (n == 0) return 1;
  if (a == 0) return 0;
  int r = LOG[a] * n;
  while(255 <= r) r -= 255;
  return EXP[r];
}

// ============================================================================
//  Implementation of RS sharding erasure codes via the Vandermonde matrix
//  This implementation is O(n^3), using the Berlekamp-Welch algorithm.
// ============================================================================
typedef struct {
  int n, m; // n x m (rows x cols) matrix; v[row][col]
  u8 ** v, * bp;
} gf256mat;
static gf256mat * gf256mat_init(int n, int m) {
  gf256mat * mat = xmalloc(sizeof(gf256mat));
  mat->n = n; mat->m = m;
  mat->bp = xmalloc(n * m);
  mat->v = xmalloc(n * sizeof(u8 *));
  Fi(n, mat->v[i] = mat->bp + i * m)
  return mat;
}
static void gf256mat_free(gf256mat * mat) {
  free(mat->bp); free(mat->v); free(mat);
}
static gf256mat * gf256mat_eye(int n) {
  gf256mat * mat = gf256mat_init(n, n);
  Fi(n, mat->v[i][i] = 1)
  return mat;
}
static gf256mat * gf256mat_prod(gf256mat * a, gf256mat * b) {
  gf256mat * c = gf256mat_init(a->n, b->m);
  Fi(a->n, Fk(a->m, Fj(b->m,
      c->v[i][j] ^= PROD[a->v[i][k]][b->v[k][j]])))
  return c;
}
static gf256mat * gf256mat_cat(gf256mat * a, gf256mat * b) {
  gf256mat * c = gf256mat_init(a->n, a->m + b->m);
  Fi(a->n,
    memcpy(c->v[i], a->v[i], a->m);
    memcpy(c->v[i] + a->m, b->v[i], b->m))
  return c;
}
static gf256mat * gf256mat_submat(gf256mat * a, int r, int c, int n, int m) {
  gf256mat * b = gf256mat_init(n, m);
  Fi(n, memcpy(b->v[i], a->v[r + i] + c, m))
  return b;
}
static void gf256mat_swaprows(gf256mat * a, int r1, int r2) {
  u8 * tmp = a->v[r1];
  a->v[r1] = a->v[r2];
  a->v[r2] = tmp;
}
static void gf256mat_trans(gf256mat * a) {
  Fi(a->n, Fj0(a->m, i + 1,
    u8 tmp = a->v[i][j];
    a->v[i][j] = a->v[j][i];
    a->v[j][i] = tmp))
}
static gf256mat * gf256mat_inv(gf256mat * a) {
  gf256mat * b = gf256mat_eye(a->n);
  gf256mat * c = gf256mat_cat(a, b);
  Fi(c->n,
    int r = i;
    while (r < c->n && c->v[r][i] == 0)
      r++;
    if (r == c->n) { gf256mat_free(c); return NULL; }
    gf256mat_swaprows(c, i, r);
    u8 inv = gf256_div(1, c->v[i][i]);
    Fj(c->m,c->v[i][j] = PROD[inv][c->v[i][j]])
    Fj(c->n, if (j != i) {
      u8 f = c->v[j][i];
      Fk(c->m, c->v[j][k] ^= PROD[f][c->v[i][k]]);
    }))
  gf256mat * d = gf256mat_submat(c, 0, a->n, a->n, a->n);
  gf256mat_free(c);
  return d;
}
static gf256mat * vandermonde(int row, int col) {
  gf256mat * mat = gf256mat_init(row, col);
  Fi(row, Fj(col, mat->v[i][j] = gf256_exp(i, j)))
  return mat;
}
typedef struct {
  int data, parity, total;
  gf256mat * matrix;
  uint8_t ** rows;
} rs;
static rs * rs_init(int data_shards, int parity_shards) {
  rs * r = xmalloc(sizeof(rs));
  r->data = data_shards; r->parity = parity_shards;
  r->total = data_shards + parity_shards;
  gf256mat * v, * vsq, * vi;
  v = vandermonde(r->total, data_shards);
  vsq = gf256mat_submat(v, 0, 0, data_shards, data_shards);
  vi = gf256mat_inv(vsq);
  r->matrix = gf256mat_prod(v, vi);
  gf256mat_free(v);  gf256mat_free(vsq);  gf256mat_free(vi);
  r->rows = calloc(parity_shards, sizeof(uint8_t *));
  Fi(parity_shards, r->rows[i] = r->matrix->v[data_shards + i])
  return r;
}
static void gf256_prod(uint8_t * restrict dst, uint8_t a,
                       uint8_t * restrict b, size_t len) {
  Fi(len, dst[i] ^= PROD[a][b[i]]);
}
static void rs_encode(rs * r, uint8_t ** in, size_t len) {
  Fj(r->parity, memset(in[r->data + j], 0, len));
#if defined(XPAR_OPENMP)
  #pragma omp parallel for if((r->data + r->parity) > 8 && len > 100 * 1024 * 1024)
#endif
  Fj(r->parity, Fk(r->data, gf256_prod(in[r->data + j], r->rows[j][k], in[k], len)))
}
static bool rs_correct(rs * r, uint8_t ** in, uint8_t * shards_present, size_t len) {
  int present = 0;
  Fi(r->total, present += !!shards_present[i])
  if (present < r->data) return false;
  if (present == r->total) return true;
  gf256mat * mat = gf256mat_init(r->data, r->data);
  uint8_t ** shards = calloc(r->data, sizeof(uint8_t *));
  for (int i = 0, k = 0; i < r->total && k < r->data; i++)
    if (shards_present[i]) {
      Fj(r->data, mat->v[k][j] = r->matrix->v[i][j]);
      shards[k++] = in[i];
    }
  gf256mat * inv = gf256mat_inv(mat);
  gf256mat_free(mat);
  if (!inv) return false;
  gf256mat_trans(inv);

#if defined(XPAR_OPENMP)
  #pragma omp parallel for if((r->data + r->parity) > 8 && len > 100 * 1024 * 1024)
#endif
  Fi(r->data, Fj(r->data,
    if(!shards_present[i]) gf256_prod(in[i], inv->v[j][i], shards[j], len)
  ))
  gf256mat_free(inv);  free(shards);
  return true;
}
static void rs_destroy(rs * r) {
  gf256mat_free(r->matrix);
  free(r->rows); free(r);
}

// ============================================================================
//  Sharded mode encoders/decoders.
// ============================================================================
static void do_sharded_encode(sharded_encoding_options_t o, u8 * buf, sz size) {
  FILE * out[MAX_TOTAL_SHARDS];
  if (o.pshards >= o.dshards) FATAL("Too many parity shards.");
  Fi(o.dshards + o.pshards,
    char * name; asprintf(&name, "%s.xpa.%03d", o.output_prefix, i);
    struct stat st;  memset(&st, 0, sizeof(struct stat));
    int exists = stat(name, &st);
    if ((st.st_size || S_ISDIR(st.st_mode)) && exists != -1 && !o.force)
      FATAL("Output file `%s' exists and is not empty.", name);
    if (!(out[i] = fopen(name, "wb"))) FATAL_PERROR("fopen");
    free(name);
  )
  u8 * shards[MAX_TOTAL_SHARDS];
  sz shard_size = (size + o.dshards - 1) / o.dshards;
  if (shard_size <= 128)
    FATAL("Input file too small to be sharded with the given parameters.");
  Fi(o.dshards - 1, shards[i] = buf + i * shard_size);
  // last shard: use a temporary buffer to avoid overflowing
  shards[o.dshards - 1] = xmalloc(shard_size);
  if (size > (o.dshards - 1) * shard_size)
    memcpy(shards[o.dshards - 1], buf + (o.dshards - 1) * shard_size,
      size - (o.dshards - 1) * shard_size);
  Fi0(o.dshards + o.pshards, o.dshards, shards[i] = xmalloc(shard_size));
  rs * r = rs_init(o.dshards, o.pshards);
  rs_encode(r, shards, shard_size);
  rs_destroy(r);
  u8 size_bytes[8] = { 0 };
  Fj(8, size_bytes[j] = size >> (56 - 8 * j));
  Fi(o.dshards + o.pshards,
    u32 checksum = crc32c(shards[i], shard_size);
    xfwrite("XPAS", 4, out[i]);
    u8 checksum_bytes[4];  Fj(4, checksum_bytes[j] = checksum >> (24 - 8 * j));
    xfwrite(checksum_bytes, 4, out[i]);
    xfwrite(&o.dshards, 1, out[i]);
    xfwrite(&o.pshards, 1, out[i]);
    xfwrite(&i, 1, out[i]);
    xfwrite(size_bytes, 8, out[i]);
    xfwrite(shards[i], shard_size, out[i]);
  )
  Fi(o.dshards + o.pshards, xfclose(out[i]));
  Fi0(o.dshards + o.pshards, o.dshards, free(shards[i]));
  free(shards[o.dshards - 1]);
}

void sharded_encode(sharded_encoding_options_t o) {
  if(!o.no_map) {
    #if defined(XPAR_ALLOW_MAPPING)
    mmap_t map = xpar_map(o.input_name);
    if (map.map) {
      do_sharded_encode(o, map.map, map.size);
      xpar_unmap(&map);
      return;
    }
    #endif
  }
  FILE * in = fopen(o.input_name, "rb");
  if (!in) FATAL_PERROR("fopen");
  if (!is_seekable(in)) FATAL("Input not seekable.");
  fseek(in, 0, SEEK_END);
  sz size = ftell(in);
  fseek(in, 0, SEEK_SET);
  // Allocate a page more to avoid a problem with
  // the total # of shards not being a multiple of the size,
  // which will imply that the last shard will be smaller.
  // If we overallocate and wipe out the last page, we can
  // (controllably) overflow this buffer.
  u8 * buffer = xmalloc(size);
  if (xfread(buffer, size, in) != size) FATAL("Short read.");
  fclose(in);
  do_sharded_encode(o, buffer, size);
  free(buffer);
}
void sharded_decode(sharded_decoding_options_t opt) {
  sharded_hv_result_t res[MAX_TOTAL_SHARDS];
  if (opt.n_input_shards > MAX_TOTAL_SHARDS)
    FATAL(
      "Too many input shards. While many of them may be wrong and\n"
      "subsequently discarded, this functionality is not implemented\n"
      "yet. Please throw away some of the input shards and try again.\n"
    );
  Fi(opt.n_input_shards,
    res[i] = validate_shard_header(!opt.no_map, opt.input_files[i], opt, "XPAS");
    if (!res[i].valid) {
      if (!opt.quiet)
        fprintf(stderr,
          "Invalid shard header in `%s', skipping.\n", opt.input_files[i]);
      if (!opt.force) exit(1);
    }
  )
  // Consensus voting.
  u8 consensus_dshards, consensus_pshards;
  sz consensus_size, consensus_shard_size;
  {
    u8 b[MAX_TOTAL_SHARDS];
    Fi(opt.n_input_shards, b[i] = res[i].dshards);
    consensus_dshards = *(u8 *) most_frequent((u8 *) b, opt.n_input_shards, 1);
    Fi(opt.n_input_shards, b[i] = res[i].pshards);
    consensus_pshards = *(u8 *) most_frequent((u8 *) b, opt.n_input_shards, 1);
  }
  {
    sz b[MAX_TOTAL_SHARDS];
    Fi(opt.n_input_shards, b[i] = res[i].total_size);
    consensus_size =
      *(sz *) most_frequent((u8 *) b, opt.n_input_shards, sizeof(sz));
    Fi(opt.n_input_shards, b[i] = res[i].shard_size);
    consensus_shard_size =
      *(sz *) most_frequent((u8 *) b, opt.n_input_shards, sizeof(sz));
  }
  // Kick out shards that don't match the consensus.
  Fi(opt.n_input_shards,
    if (res[i].dshards != consensus_dshards
     || res[i].pshards != consensus_pshards
     || res[i].total_size != consensus_size
     || res[i].shard_size != consensus_shard_size) {
      res[i].valid = false;
      if (!opt.quiet)
        fprintf(stderr,
          "Shard `%s' does not match the consensus, skipping.\n",
          opt.input_files[i]);
      if (!opt.force) exit(1);
    }
  )
  // Check if we have a duplicate of any shard.
  Fi(opt.n_input_shards, Fj0(opt.n_input_shards, i + 1, 
    if (res[i].shard_number == res[j].shard_number
      && res[i].valid && res[j].valid) {
      FATAL(
        "Duplicate shard number %u. A future version of this tool\n"
        "will have the capability to try combinations of shards\n"
        "to pick the valid shard %u out of your input. For now,\n"
        "this is a fatal error. Please remove one of the offending\n"
        "shards:\n%s\n%s\n",
        res[i].shard_number, res[i].shard_number,
        opt.input_files[i], opt.input_files[j]
      );
    }
  ))
  // Free the invalid buffers, compact the valid ones.
  Fi(opt.n_input_shards,
    if (!res[i].valid) unmap_shard(&res[i]), res[i].shard_number = 0xFF);
  int n_valid_shards = 0;
  Fi(opt.n_input_shards, if (res[i].valid) {
    n_valid_shards++;
    for (int j = i; j && res[j].shard_number < res[j - 1].shard_number; j--) {
      sharded_hv_result_t tmp = res[j];
      res[j] = res[j - 1];
      res[j - 1] = tmp;
    }
  })
  // Log some information.
  if (opt.verbose) {
    fprintf(stderr,
      "Sharding consensus: %u data shards, %u parity shards\n",
      consensus_dshards, consensus_pshards);
    fprintf(stderr,
      "Sharding consensus: %zu bytes total, %zu bytes per shard\n",
      consensus_size, consensus_shard_size);
    fprintf(stderr, "Valid shards:\n");
    Fi(n_valid_shards, 
      fprintf(stderr,
        "  %s (#%d)\n",
        opt.input_files[i], res[i].shard_number);
    )
  }
  if (n_valid_shards < consensus_dshards) {
    FATAL(
      "Not enough valid shards to recover the data. The consensus\n"
      "requires %u shards, but only %d are available.\n",
      consensus_dshards, n_valid_shards);
  }
  FILE * out = fopen(opt.output_file, "wb");
  if (!out) FATAL_PERROR("fopen");
  if (n_valid_shards == consensus_dshards + consensus_pshards) {
    Fi(consensus_dshards,
      sz w = MIN(consensus_size, consensus_shard_size);
      xfwrite(res[i].buf + SHARD_HEADER_SIZE, w, out);
      consensus_size -= w)
    Fi(n_valid_shards, unmap_shard(&res[i]));
    return;
  }
  rs * r = rs_init(consensus_dshards, consensus_pshards);
  u8 * buffers[MAX_TOTAL_SHARDS] = { NULL }, pres[MAX_TOTAL_SHARDS] = { 0 };
  Fi(n_valid_shards,
    buffers[res[i].shard_number] = res[i].buf + SHARD_HEADER_SIZE,
    pres[res[i].shard_number] = 1)
  Fi(consensus_dshards + consensus_pshards, if (!buffers[i]) {
    buffers[i] = xmalloc(consensus_shard_size);
    memset(buffers[i], 0, consensus_shard_size);
  })
  if(!rs_correct(r, buffers, pres, consensus_shard_size))
    FATAL("Failed to correct the data.");
  Fi(consensus_dshards,
    sz w = MIN(consensus_size, consensus_shard_size);
    xfwrite(buffers[i], w, out);
    consensus_size -= w)
  xfclose(out);
  rs_destroy(r);
  Fi(n_valid_shards, unmap_shard(&res[i]));
  Fi(consensus_dshards + consensus_pshards, if (!pres[i]) free(buffers[i]));
}