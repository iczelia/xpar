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

/*  xpar: shared generation-chain reader interface.  */
#ifndef XPAR_OPS_CHAIN_H
#define XPAR_OPS_CHAIN_H

#include "ops.h"
#include "armour.h"
#include "container.h"
#include "manifest.h"

#define XPAR_GEN_NONE 0xFFFFFFFFu

typedef struct {
  char * path;
  u8 * data;
  sz len;
  u32 gen, volume_index, volume_kind;
  u64 recovery_first, recovery_count;
  bool armoured_file, armoured_crit, has_volh;
  const u8 * layt_body;
  sz layt_len;
  u8 set_id[XPAR_SET_ID_LEN];
} xpar_chain_vol;

typedef struct {
  u8 set_id[XPAR_SET_ID_LEN];
  xpar_setd sd;
  u32 parent;
  bool parent_missing;
  u64 recovery_count, recovery_top;
  u32 vol_count;
  const u8 * layt_body;
  sz layt_len;
} xpar_chain_gen;

typedef struct {
  xpar_chain_vol * vol;
  u32 vol_count;
  xpar_chain_gen * gen;
  u32 gen_count, head;
  bool forked;
  u8 ** blob;
  u32 blob_count;
  xpar_critset crit;
  xpar_key key;
  u8 master[XPAR_BLAKE3_KEY_LEN];
  bool key_loaded, authenticated, auth_only;
  char * base, * dir;
} xpar_chain;

void xpar_gchain_load(const xpar_options *, xpar_chain *);
void xpar_gchain_free(xpar_chain *);
u32 xpar_gchain_select(const xpar_chain *, const xpar_genref *);
void xpar_gchain_genref(const xpar_chain *, u32, xpar_genref *,
                        char id_text[XPAR_SET_ID_LEN * 2 + 1]);
void xpar_gchain_manifest(const xpar_chain *, u32, xpar_manifest *, u32 **);
u32 xpar_gchain_posix(const xpar_chain *, u32, xpar_posix_rec **);
void xpar_gchain_posix_free(xpar_posix_rec *, u32);

typedef struct {
  u8 symbol_bits;
  u32 poly, n, k, fcr, prim;
  u64 depth, plain_length, armoured_length, stream_offset, stream_length;
} xpar_arm_prologue;

bool xpar_garm_prologue(const u8 *, sz, xpar_arm_prologue *, int *);

/*  Three copies of the 96-byte prologue, each with its 32 GF(2^8) parity
    bytes, which is how an armoured volume starts.  */
void xpar_garm_write_prologue(xpar_file *, const xpar_armour_params *,
                              u64 plain_length, u64 armoured_length,
                              u64 stream_offset, u64 stream_length);
void xpar_garm_write_plain(const char *, const xpar_armour_params *,
                           const u8 *, u64, u64, u64);
void xpar_garm_write_patched(const char *, const xpar_armour_params *,
                             const u8 *, u64, u64, u64, xpar_file *,
                             const u64 *, u64, u64);
void xpar_garm_write_inserted(const char *, const xpar_armour_params *,
                              const u8 *, u64, u64, const u8 *, u64,
                              u64, u64);
i64 xpar_gchain_gen_of(const xpar_chain *, u64, u64);
void xpar_gchain_deps(const xpar_chain *, const xpar_manifest *, const u32 *,
                      u64 *, u64 *);

#endif
