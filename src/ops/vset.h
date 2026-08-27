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

/*  xpar: validated volume-set reader and staged-output checks.  */
#ifndef XPAR_OPS_VSET_H
#define XPAR_OPS_VSET_H

#include "ops.h"
#include "container.h"
#include "json.h"
#include "armour.h"
#include "manifest.h"
#include "slice.h"

typedef struct xpar_vset xpar_vset;

xpar_vset * xpar_vset_open(const xpar_options *);
void xpar_vset_close(xpar_vset *);
int xpar_vset_check(xpar_vset *, const xpar_options *, xpar_json *);
void xpar_vset_report(const xpar_vset *, const xpar_options *, int);
void xpar_vset_json_set(const xpar_vset *, xpar_json *);
void xpar_vset_json_summary(const xpar_vset *, xpar_json *, int);
bool xpar_vset_read(xpar_vset *, u64, u8 *, u64);
void xpar_vset_mark_superseded(xpar_vset *, const xpar_vset *);
bool xpar_vset_bind_sources(xpar_vset *, const xpar_manifest *);

const xpar_setd * xpar_vset_setd(const xpar_vset *);
const xpar_geom * xpar_vset_geom(const xpar_vset *);
const xpar_geom * xpar_vset_egeom(const xpar_vset *);
const xpar_manifest * xpar_vset_manifest(const xpar_vset *);
const xpar_occindex * xpar_vset_occ(const xpar_vset *);
const xpar_tags * xpar_vset_tags(const xpar_vset *);
const xpar_layt * xpar_vset_layt(const xpar_vset *);
const xpar_erasures * xpar_vset_erasures(const xpar_vset *);
const char * xpar_vset_dir(const xpar_vset *);
const u8 * xpar_vset_id(const xpar_vset *);
const xpar_key * xpar_vset_key(const xpar_vset *);
bool xpar_vset_authenticated(const xpar_vset *);
const u8 * xpar_vset_volume(const xpar_vset *, u32, u64 *);
const char * xpar_vset_volume_path(const xpar_vset *, u32);
const u8 * xpar_vset_rcvs(const xpar_vset *, u64, u64 *);
bool xpar_vset_armoured(const xpar_vset *, const u8 **, u64 *, u64 *,
                        xpar_armour_params *, const char **);

/*  Rewrite damaged named split volumes from intact substitutes.  */
bool xpar_vset_rewrite_substituted(xpar_vset *, const char ** reason);

/*  Reconstruct one bare split data-volume range from the other data and
    recovery slices, writing bytes relative to the start of the volume.
    The destination must support pread as well as pwrite: reconstructed
    slices are strong-gated before this returns success.  */
bool xpar_vset_recover_data(xpar_vset *, u64 stream_offset, u64 length,
                            u64 memory, xpar_file * destination,
                            const char ** reason);

bool xpar_vset_cell_covered(const xpar_vset *, u64);
u32 xpar_vset_have_tables(const xpar_vset *);
u32 xpar_vset_volumes(const xpar_vset *);
u64 xpar_vset_recovery(const xpar_vset *);
u64 xpar_vset_recovery_total(const xpar_vset *);
u64 xpar_vset_bad_cells(const xpar_vset *);
u64 xpar_vset_volumes_to_rewrite(const xpar_vset *);
bool xpar_vset_stream_intact(const xpar_vset *, int rc);
u64 xpar_vset_bad_slices(const xpar_vset *);
u64 xpar_vset_bad_entries(const xpar_vset *);
u64 xpar_vset_alias_bad(const xpar_vset *);
u64 xpar_vset_max_depth(const xpar_vset *);
u64 xpar_vset_bytes_read(const xpar_vset *);
u64 xpar_vset_inner_corrected(const xpar_vset *);

u64 xpar_verify_syndromes(void);
void xpar_verify_written_set(const xpar_options *, const char * index_path);
void xpar_verify_written_set_sources(const xpar_options *,
                                     const char * index_path,
                                     const xpar_manifest *);
void xpar_verify_written_set_at(const xpar_options *, const char * index_path,
                                const xpar_genref * generation);
void xpar_verify_written_archive_at(const xpar_options *, const char * path,
                                    const xpar_genref * generation);
bool xpar_verify_written_volume(const char *, const xpar_key *, const u8 *,
                                u32, u32, u64, u64, u64);
bool xpar_verify_packets_ok(const u8 *, u64, const xpar_key *);

/*  Lazily correct armour frames backing [lo, hi); report any change.  */
bool xpar_vset_armour_correct(xpar_vset *, u64 lo, u64 hi);
bool xpar_verify_next_armg(const u8 *, u64, const xpar_key *, u64 *,
                           const u8 **, u64 *);

#endif
