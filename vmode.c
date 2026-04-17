/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.  */

#include "vmode.h"

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
  while (255 <= r) r -= 255;
  return EXP[r];
}

/*  Vandermonde-matrix RS sharding (O(n^3), Berlekamp-Welch).  */
typedef struct {
  int n, m; /*  n x m (rows x cols) matrix; v[row][col]  */
  u8 ** v, * bp;
} gf256mat;
static gf256mat * gf256mat_init(int n, int m) {
  gf256mat * mat = xpar_malloc(sizeof(gf256mat));
  mat->n = n; mat->m = m;
  mat->bp = xpar_malloc(n * m);
  mat->v = xpar_malloc(n * sizeof(u8 *));
  Fi(n, mat->v[i] = mat->bp + i * m)
  return mat;
}
static void gf256mat_free(gf256mat * mat) {
  xpar_free(mat->bp); xpar_free(mat->v); xpar_free(mat);
}
static gf256mat * gf256mat_eye(int n) {
  gf256mat * mat = gf256mat_init(n, n);
  Fi(n, mat->v[i][i] = 1)
  return mat;
}
static gf256mat * gf256mat_prod(gf256mat * restrict a, gf256mat * restrict b) {
  gf256mat * c = gf256mat_init(a->n, b->m);
  Fi(a->n, Fk(a->m, Fj(b->m,
      c->v[i][j] ^= PROD[a->v[i][k]][b->v[k][j]])))
  return c;
}
static gf256mat * gf256mat_cat(gf256mat * restrict a, gf256mat * restrict b) {
  gf256mat * c = gf256mat_init(a->n, a->m + b->m);
  Fi(a->n,
    xpar_memcpy(c->v[i], a->v[i], a->m);
    xpar_memcpy(c->v[i] + a->m, b->v[i], b->m))
  return c;
}
static gf256mat * gf256mat_submat(gf256mat * a, int r, int c, int n, int m) {
  gf256mat * b = gf256mat_init(n, m);
  Fi(n, xpar_memcpy(b->v[i], a->v[r + i] + c, m))
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
  gf256mat_free(b);
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
  rs * r = xpar_malloc(sizeof(rs));
  r->data = data_shards; r->parity = parity_shards;
  r->total = data_shards + parity_shards;
  gf256mat * v, * vsq, * vi;
  v = vandermonde(r->total, data_shards);
  vsq = gf256mat_submat(v, 0, 0, data_shards, data_shards);
  vi = gf256mat_inv(vsq);
  r->matrix = gf256mat_prod(v, vi);
  gf256mat_free(v);  gf256mat_free(vsq);  gf256mat_free(vi);
  r->rows = xpar_malloc(parity_shards * sizeof(uint8_t *));
  Fi(parity_shards, r->rows[i] = r->matrix->v[data_shards + i])
  return r;
}
static void gf256_prod(uint8_t * restrict dst, uint8_t a,
                       uint8_t * restrict b, size_t len) {
  Fi(len, dst[i] ^= PROD[a][b[i]]);
}
struct _pf_ctx_rs_enc {
  rs * r; uint8_t ** in; size_t len;
};
static void _pf_fn_rs_enc(sz j, void * p) {
  struct _pf_ctx_rs_enc * c = p;
  Fk(c->r->data, gf256_prod(c->in[c->r->data + j], c->r->rows[j][k],
                            c->in[k], c->len))
}
static void rs_encode(rs * r, uint8_t ** in, size_t len) {
  Fj(r->parity, xpar_memset(in[r->data + j], 0, len));
  struct _pf_ctx_rs_enc ctx = { r, in, len };
  if (r->data + r->parity > 8 && len > MiB(100))
    xpar_parallel_for((sz) r->parity, _pf_fn_rs_enc, &ctx);
  else
    Fj(r->parity, _pf_fn_rs_enc((sz) j, &ctx));
}

struct _pf_ctx_rs_cor {
  rs * r; uint8_t ** in; uint8_t * presence;
  gf256mat * inv; uint8_t ** shards; size_t len;
};
static void _pf_fn_rs_cor(sz i, void * p) {
  struct _pf_ctx_rs_cor * c = p;
  if (c->presence[i]) return;
  Fj(c->r->data, gf256_prod(c->in[i], c->inv->v[j][i], c->shards[j], c->len))
}
static bool rs_correct(rs * r, uint8_t ** in,
                       uint8_t * presence, size_t len) {
  int present = 0;
  Fi(r->total, present += !!presence[i])
  if (present < r->data) return false;
  if (present == r->total) return true;
  gf256mat * mat = gf256mat_init(r->data, r->data);
  uint8_t ** shards = xpar_malloc(r->data * sizeof(uint8_t *));
  for (int i = 0, k = 0; i < r->total && k < r->data; i++)
    if (presence[i]) {
      Fj(r->data, mat->v[k][j] = r->matrix->v[i][j]);
      shards[k++] = in[i];
    }
  gf256mat * inv = gf256mat_inv(mat);
  gf256mat_free(mat);
  if (!inv) return false;
  gf256mat_trans(inv);

  struct _pf_ctx_rs_cor ctx = { r, in, presence, inv, shards, len };
  if ((r->data + r->parity) > 8 && len > MiB(100))
    xpar_parallel_for((sz) r->data, _pf_fn_rs_cor, &ctx);
  else
    Fi(r->data, _pf_fn_rs_cor((sz) i, &ctx));
  gf256mat_free(inv);  xpar_free(shards);
  return true;
}
static void rs_destroy(rs * r) {
  gf256mat_free(r->matrix);
  xpar_free(r->rows); xpar_free(r);
}

/*  -----------------------------------------------------------------------
  Sharded mode encoders/decoders.  */
static void do_sharded_encode(sharded_encoding_options_t o,
                              u8 * buf, sz size) {
  xpar_file ** out = xpar_malloc(MAX_TOTAL_SHARDS * sizeof(xpar_file *));
  if (o.pshards >= o.dshards) FATAL("Too many parity shards.");
  Fi(o.dshards + o.pshards,
    char * name;
#if defined(XPAR_DOS)
    /*  8.3 DOS: shards get a numeric extension, no .xpa marker.  */
    char suf[8]; xpar_snprintf(suf, sizeof(suf), ".%03d", i);
    name = xpar_derive_name(o.output_prefix, suf);
#else
    xpar_asprintf(&name, "%s.xpa.%03d", o.output_prefix, i);
#endif
    xpar_stat_t st;
    int exists = xpar_stat_path(name, &st);
    if (exists == 0 && (st.size || st.is_dir) && !o.force)
      FATAL("Output file `%s' exists and is not empty.", name);
    out[i] = xpar_open(name,
      XPAR_O_WRITE | XPAR_O_CREATE | XPAR_O_TRUNCATE);
    if (!out[i]) FATAL_PERROR("fopen");
    xpar_free(name);
  )
  u8 ** shards = xpar_malloc(MAX_TOTAL_SHARDS * sizeof(u8 *));
  sz shard_size = (size + o.dshards - 1) / o.dshards;
  if (shard_size <= 128)
    FATAL("Input file too small to be sharded with the given parameters.");
  Fi(o.dshards - 1, shards[i] = buf + i * shard_size);
  /*  last shard: zeroed scratch buffer so tail padding is deterministic
      and heap contents never reach disk or the MAC domain.  */
  shards[o.dshards - 1] = xpar_malloc(shard_size);
  xpar_memset(shards[o.dshards - 1], 0, shard_size);
  if (size > (o.dshards - 1) * shard_size)
    xpar_memcpy(shards[o.dshards - 1], buf + (o.dshards - 1) * shard_size,
      size - (o.dshards - 1) * shard_size);
  Fi0(o.dshards + o.pshards, o.dshards, shards[i] = xpar_malloc(shard_size));
  rs * r = rs_init(o.dshards, o.pshards);
  rs_encode(r, shards, shard_size);
  rs_destroy(r);
  Fi(o.dshards + o.pshards,
    u8 hdr[SHARD_HEADER_BLAKE2B_SIZE];
    u8 eos[SHARD_EOS_BLAKE2B_SIZE];
    sz hs = pack_shard_header(hdr, "XPAS", o.dshards, o.pshards, (u8) i,
                              size, shards[i], shard_size,
                              o.integrity, o.auth_key, o.auth_keylen);
    sz es = pack_eos_marker(eos, hdr, o.integrity,
                            o.auth_key, o.auth_keylen);
    xpar_xwrite(out[i], hdr, hs);
    xpar_xwrite(out[i], shards[i], shard_size);
    xpar_xwrite(out[i], eos, es);
  )
  Fi(o.dshards + o.pshards, xpar_xclose(out[i]));
  Fi0(o.dshards + o.pshards, o.dshards, xpar_free(shards[i]));
  xpar_free(shards[o.dshards - 1]);  xpar_free(out);  xpar_free(shards);
}

void sharded_encode(sharded_encoding_options_t o) {
  if (!o.no_map) {
    #if defined(XPAR_ALLOW_MAPPING)
    xpar_mmap map = xpar_map(o.input_name);
    if (map.map) {
      do_sharded_encode(o, map.map, map.size);
      xpar_unmap(&map);
      return;
    }
    #endif
  }
  xpar_file * in = xpar_open(o.input_name, XPAR_O_READ);
  if (!in) FATAL_PERROR("fopen");
  if (!xpar_is_seekable(in)) FATAL("Input not seekable.");
  xpar_seek(in, 0, XPAR_SEEK_END);
  i64 size_raw = xpar_tell(in);
  if (size_raw < 0) FATAL_PERROR("ftell");
  sz size = (sz) size_raw;
  xpar_seek(in, 0, XPAR_SEEK_SET);
  u8 * buffer = xpar_malloc(size);
  if (xpar_xread(in, buffer, size) != size) FATAL("Short read.");
  xpar_close(in);
  do_sharded_encode(o, buffer, size);
  xpar_free(buffer);
}
void sharded_decode(sharded_decoding_options_t opt) {
  sharded_hv_result_t * res =
    xpar_malloc(MAX_TOTAL_SHARDS * sizeof(sharded_hv_result_t));
  if (opt.n_input_shards > MAX_TOTAL_SHARDS)
    FATAL(
      "Too many input shards. While many of them may be wrong and\n"
      "subsequently discarded, this functionality is not implemented\n"
      "yet. Please throw away some of the input shards and try again.\n"
    );
  Fi(opt.n_input_shards,
    res[i] = validate_shard_header(opt.input_files[i], opt, "XPAS");
    if (!res[i].valid) {
      if (!opt.quiet)
        xpar_fprintf(xpar_stderr,
          "Invalid shard header in `%s', skipping.\n", opt.input_files[i]);
      if (!opt.force) { xpar_free(res); xpar_exit(1); }
    }
  )
  /*  Consensus voting.  */
  u8 consensus_dshards, consensus_pshards;
  sz consensus_size, consensus_shard_size;
  {
    u8 b[MAX_TOTAL_SHARDS];
    Fi(opt.n_input_shards, b[i] = res[i].dshards);
    consensus_dshards = *(u8 *) most_frequent(b, opt.n_input_shards, 1);
    Fi(opt.n_input_shards, b[i] = res[i].pshards);
    consensus_pshards = *(u8 *) most_frequent(b, opt.n_input_shards, 1);
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
  if ((u64) consensus_size
      > (u64) consensus_dshards * (u64) consensus_shard_size)
    FATAL("Header total_size (%zu) exceeds %u data shards of %zu bytes.",
          consensus_size, consensus_dshards, consensus_shard_size);
  /*  Kick out shards that don't match the consensus.  */
  Fi(opt.n_input_shards,
    if (res[i].dshards != consensus_dshards
     || res[i].pshards != consensus_pshards
     || res[i].total_size != consensus_size
     || res[i].shard_size != consensus_shard_size
     || res[i].shard_number >= consensus_dshards + consensus_pshards) {
      res[i].valid = false;
      if (!opt.quiet)
        xpar_fprintf(xpar_stderr,
          "Shard `%s' does not match the consensus, skipping.\n",
          opt.input_files[i]);
      if (!opt.force) { xpar_free(res); xpar_exit(1); }
    }
  )
  /*  Check if we have a duplicate of any shard.  */
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
  /*  Free the invalid buffers, compact the valid ones.  */
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
  /*  Log some information.  */
  if (opt.verbose) {
    xpar_fprintf(xpar_stderr,
      "Sharding consensus: %u data shards, %u parity shards\n",
      consensus_dshards, consensus_pshards);
    xpar_fprintf(xpar_stderr,
      "Sharding consensus: %zu bytes total, %zu bytes per shard\n",
      consensus_size, consensus_shard_size);
    xpar_fprintf(xpar_stderr, "Valid shards:\n");
    Fi(n_valid_shards,
      xpar_fprintf(xpar_stderr,
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
  xpar_file * out = xpar_open(opt.output_file,
    XPAR_O_WRITE | XPAR_O_CREATE | XPAR_O_TRUNCATE);
  if (!out) FATAL_PERROR("fopen");
  if (n_valid_shards == consensus_dshards + consensus_pshards) {
    Fi(consensus_dshards,
      sz w = MIN(consensus_size, consensus_shard_size);
      xpar_xwrite(out, res[i].buf + res[i].hdr_size, w);
      consensus_size -= w)
    Fi(n_valid_shards, unmap_shard(&res[i]));
    xpar_xclose(out);
    xpar_free(res);
    return;
  }
  rs * r = rs_init(consensus_dshards, consensus_pshards);
  u8 * buffers[MAX_TOTAL_SHARDS] = { NULL }, pres[MAX_TOTAL_SHARDS] = { 0 };
  Fi(n_valid_shards,
    buffers[res[i].shard_number] = res[i].buf + res[i].hdr_size,
    pres[res[i].shard_number] = 1)
  Fi(consensus_dshards + consensus_pshards, if (!buffers[i]) {
    buffers[i] = xpar_malloc(consensus_shard_size);
    xpar_memset(buffers[i], 0, consensus_shard_size);
  })
  if (!rs_correct(r, buffers, pres, consensus_shard_size))
    FATAL("Failed to correct the data.");
  Fi(consensus_dshards,
    sz w = MIN(consensus_size, consensus_shard_size);
    xpar_xwrite(out, buffers[i], w);
    consensus_size -= w)
  xpar_xclose(out);  rs_destroy(r);
  Fi(n_valid_shards, unmap_shard(&res[i]));
  Fi(consensus_dshards + consensus_pshards,
    if (!pres[i]) xpar_free(buffers[i]));
  xpar_free(res);
}
/*  Dry-run of sharded_decode; returns invalid-or-missing count.
    Exits 1 when unrecoverable.  */
void sharded_test(sharded_decoding_options_t opt) {
  sharded_hv_result_t * res = xpar_malloc(MAX_TOTAL_SHARDS * sizeof(*res));
  if (opt.n_input_shards > MAX_TOTAL_SHARDS) {
    if (!opt.quiet)
      xpar_fprintf(xpar_stderr,
        "Too many input shards (max %d).\n", MAX_TOTAL_SHARDS);
    xpar_free(res); xpar_exit(1);
  }
  Fi(opt.n_input_shards,
    res[i] = validate_shard_header(opt.input_files[i], opt, "XPAS");
    if (!res[i].valid && !opt.quiet)
      xpar_fprintf(xpar_stderr, "Invalid shard `%s'.\n", opt.input_files[i]);
  )
  int n_valid = 0;
  Fi(opt.n_input_shards, if (res[i].valid) n_valid++);
  u8 consensus_dshards = 0, consensus_pshards = 0;
  sz consensus_shard_size = 0;
  if (n_valid) {
    u8 b[MAX_TOTAL_SHARDS]; int n = 0;
    Fi(opt.n_input_shards, if (res[i].valid) b[n++] = res[i].dshards);
    consensus_dshards = *(u8 *) most_frequent(b, n, 1);
    n = 0;
    Fi(opt.n_input_shards, if (res[i].valid) b[n++] = res[i].pshards);
    consensus_pshards = *(u8 *) most_frequent(b, n, 1);
    sz ss[MAX_TOTAL_SHARDS]; n = 0;
    Fi(opt.n_input_shards, if (res[i].valid) ss[n++] = res[i].shard_size);
    consensus_shard_size =
      *(sz *) most_frequent((u8 *) ss, n, sizeof(sz));
  }
  if (n_valid < consensus_dshards) {
    if (!opt.quiet)
      xpar_fprintf(xpar_stderr,
        "Only %d valid shards, need %u to recover.\n",
        n_valid, consensus_dshards);
    Fi(opt.n_input_shards, if (res[i].buf) unmap_shard(&res[i]));
    xpar_free(res);
    xpar_exit(1);
  }
  /*  Verify reconstruction if any shard was invalid.  */
  int n_total = consensus_dshards + consensus_pshards;
  int bad = n_total - n_valid;
  if (n_valid < n_total) {
    rs * r = rs_init(consensus_dshards, consensus_pshards);
    u8 * buffers[MAX_TOTAL_SHARDS] = { NULL };
    u8 pres[MAX_TOTAL_SHARDS] = { 0 };
    Fi(opt.n_input_shards, if (res[i].valid &&
        res[i].shard_number < consensus_dshards + consensus_pshards) {
      buffers[res[i].shard_number] = res[i].buf + res[i].hdr_size;
      pres[res[i].shard_number] = 1;
    })
    Fi(n_total, if (!buffers[i]) {
      buffers[i] = xpar_malloc(consensus_shard_size);
      xpar_memset(buffers[i], 0, consensus_shard_size);
    })
    if (!rs_correct(r, buffers, pres, consensus_shard_size)) {
      if (!opt.quiet)
        xpar_fprintf(xpar_stderr, "RS reconstruction failed.\n");
      bad++;
    }
    rs_destroy(r);
    Fi(n_total, if (!pres[i]) xpar_free(buffers[i]));
  }
  if (opt.verbose)
    xpar_fprintf(xpar_stderr,
      "Checked %zu shards, %d valid (need %u), %d bad.\n",
      opt.n_input_shards, n_valid, consensus_dshards, bad);
  Fi(opt.n_input_shards, if (res[i].buf) unmap_shard(&res[i]));
  xpar_free(res);
  if (bad) xpar_exit(1);
}