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

#ifndef XPAR_OPS_H
#define XPAR_OPS_H

#include "common.h"
#include "cli.h"
#include "container.h"

/*  Apply progress policy consistently across human and JSON output.  */
static inline bool xpar_progress_wanted(const xpar_options * o) {
  if (o->quiet) return false;
  if (o->progress == XPAR_PROGRESS_OFF) return false;
  if (o->progress == XPAR_PROGRESS_ON) return true;
  if (o->json) return false;
  return xpar_is_tty(xpar_stderr);
}

/*  Reject unknown critical packet types.  */
static inline void xpar_reject_unknown_critical(const xpar_scan * s) {
  if (s->skip_unsupported)
    FATAL_FORMAT("this set uses an unknown critical packet type");
}

int xpar_op_create(const xpar_options *);
int xpar_op_verify(const xpar_options *);
int xpar_op_repair(const xpar_options *);
int xpar_op_scrub(const xpar_options *);
int xpar_op_extract(const xpar_options *);
int xpar_op_recover(const xpar_options *);
int xpar_op_list   (const xpar_options *);
int xpar_op_info   (const xpar_options *);
int xpar_op_explain(const xpar_options *);
int xpar_op_addrecovery(const xpar_options *);
int xpar_op_add        (const xpar_options *);
int xpar_op_consolidate(const xpar_options *);
int xpar_op_prune      (const xpar_options *);
int xpar_op_undo(const xpar_options *);
int xpar_op_recover_prologue(const xpar_options *);
int xpar_op_benchmark(const xpar_options *);

/*  Re-encode volumes containing missing recovery exponents; `dry` only
    counts the slices and volumes a real run would rewrite.  */
u64 xpar_gen_regen_recovery(const xpar_options *, u64 * volumes,
                            const char ** reason, bool dry);
/*  Recreate missing index volumes from the surviving critical packets.  */
u64 xpar_gen_regen_index(const xpar_options *, u64 * volumes,
                         const char ** reason, bool dry);
/*  Remove or empty a spent undo journal; false when it still replays.  */
bool xpar_journal_drop(const char * path);
/*  Recover a maintenance journal.  */
int xpar_maint_recover(const char * journal, bool quiet);
/*  Find a journal for ARG and return its operation through OP.  */
char * xpar_maint_pending(const char * arg, const char ** op);
/*  Rewrite critical packets superseded by the index.  */
u64 xpar_gen_rewrite_stale(const xpar_options *, u64 * volumes,
                           const char ** reason, bool dry);
char * xpar_spool_stdin(const xpar_options *);
char * xpar_publish_spooled_stdin(const xpar_options *, const char *);

#endif
