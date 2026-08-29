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
  /*  Wrapped recovery and table packets were found.  */
  bool wrap_rcvs, wrap_tab;
  xpar_armour_params wrap_rcvs_ap, wrap_tab_ap;
  u64 armg_disk, armg_plain;
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

/*  Count on-disk generations lacking readable descriptors.  */
u32 xpar_gen_unreadable(const xpar_setref * ref, const u32 * have,
                        u32 have_count, char * const * read, u32 read_count,
                        u32 * first);

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
  /*  On-disk members lacking a readable generation.  */
  u32 lost_count, lost_first;
  /*  Packets failing authentication with the loaded key.  */
  u64 auth_failed;
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

/*  Detect an archive from any valid prologue copy.  */
bool xpar_garm_is_archive(const u8 *, sz);

/*  How many of the three stored prologue copies still check out.  */
u32 xpar_garm_prologue_copies(const u8 *, sz);

/*  Read critical-group armour from ARMG; false when stored plain.  */
bool xpar_gchain_crit_armour(const xpar_chain *, u32, xpar_armour_params *);

/*  Return the region code for wrapped recovery or table packets.  */
bool xpar_gchain_wrap_armour(const xpar_chain *, u32, bool rcvs,
                             xpar_armour_params *);

/*  Bytes a generation's ARMG packets occupy and the plaintext they hold. */
void xpar_gchain_armour_bytes(const xpar_chain *, u32, u64 * disk,
                              u64 * plain);

/*  Three copies of the 96-byte prologue, each with its 32 GF(2^8) parity
    bytes, which is how an armoured volume starts.  */
void xpar_garm_write_prologue(xpar_file *, const xpar_armour_params *,
                              u64 plain_length, u64 armoured_length,
                              u64 stream_offset, u64 stream_length);
void xpar_garm_write_patched(const char *, const xpar_armour_params *,
                             const u8 *, u64, u64, u64, xpar_file *,
                             const u64 *, u64, u64);
void xpar_garm_write_inserted(const char *, const xpar_armour_params *,
                              const u8 *, u64, u64, const u8 *, u64,
                              u64, u64);
i64 xpar_gchain_gen_of(const xpar_chain *, u64, u64);
/*  Report superseded slices and users of a generation.  */
void xpar_gchain_superseded(const xpar_chain *, const xpar_manifest *, u64 *);
u64 xpar_gchain_users(const xpar_chain *, const xpar_manifest *, u32);

void xpar_gchain_deps(const xpar_chain *, const xpar_manifest *, const u32 *,
                      u64 *, u64 *);

#endif
