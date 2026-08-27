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

/*  Apply progress policy consistently across verbs.  */
static inline bool xpar_progress_wanted(const xpar_options * o) {
  if (o->quiet || o->json) return false;
  if (o->progress == XPAR_PROGRESS_ON) return true;
  if (o->progress == XPAR_PROGRESS_OFF) return false;
  return xpar_is_tty(xpar_stderr);
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
char * xpar_spool_stdin(const xpar_options *);
char * xpar_publish_spooled_stdin(const xpar_options *, const char *);

#endif
