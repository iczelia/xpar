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

/*  Planning interface for codec, column, cell, and memory decisions.
    Plans are computed without allocation or file access and retain enough
    information to explain rejected candidates.  */

#ifndef XPAR_PLAN_H
#define XPAR_PLAN_H

#include "common.h"
#include "xpar2.h"
#include "slice.h"

typedef enum {
  XPAR_PLAN_OK = 0,
  XPAR_PLAN_NO_FIT,        /*  Nothing fits the memory budget.  */
  XPAR_PLAN_BAD_GEOMETRY,  /*  S, R or Z outside what the field admits.  */
  XPAR_PLAN_TOO_MANY_CELLS,/*  ceil(Z/Y) past the format's cell bound.  */
  XPAR_PLAN_NO_CODEC       /*  No codec can express these parameters.  */
} xpar_plan_status;

const char * xpar_plan_reason(xpar_plan_status);
u64 xpar_plan_default_memory(void);

typedef struct {
  u64  stream_length;      /*  L, total bytes to protect.  */
  u64  memory_budget;      /*  -m; 0 derives one from physical memory.  */
  u64  slice_size;         /*  -s; 0 lets the geometry heuristic choose.  */
  u64  slice_count;        /*  -b; 0 likewise.  */
  u64  recovery_slices;    /*  R, already resolved from the -r grammar.  */
  u32  cell_bytes;         /*  Y; 0 lets rule (d) choose.  */
  u32  column_chunk;       /*  --fft-chunk; 0 lets rules (a) and (b) choose.  */
  u8   field_log2;         /*  0 for auto.  */
  u8   codec;              /*  0xFF for auto.  */
  u8   layout;             /*  XPAR_LAYOUT_*  */
  u32  armour_frame;       /*  Bytes per armour frame, or 0 when unarmoured.  */
  bool rotational;         /*  Target device.  */
  bool streaming;          /*  Input is a pipe, so the codec must stream.  */
  int  threads;
} xpar_plan_req;

typedef struct {
  u8           codec;
  u8           field_log2;
  bool         feasible;
  const char * why;          /*  NULL when feasible.  */
  u64          encode_work;  /*  Region multiply-accumulate bytes.  */
  u64          working_set;
  u32          column_chunk;
  u64          passes;
} xpar_plan_cand;

#define XPAR_PLAN_MAX_CAND 4

typedef struct {
  xpar_geom geom;            /*  Z, S, Y and the derived cell counts.  */
  u64 recovery_slices;       /*  R  */
  u8  field_log2;
  u8  codec;
  u32 column_chunk;          /*  C, a memory-tiling choice only.  */
  u64 passes;
  u64 encode_work;

  /*  Footprint, itemised because a refusal has to say which part did not
      fit before a user can act on it.  */
  u64 mem_codec;
  u64 mem_readahead;
  u64 mem_stage;
  u64 mem_total;

  u32 dedup_target_chunk;    /*  0 when dedup is off or whole-file.  */
  int threads;

  xpar_plan_cand cand[XPAR_PLAN_MAX_CAND];
  int            cand_count;
} xpar_plan;

/*  Select a plan without allocation or I/O.  */
xpar_plan_status xpar_plan_make(const xpar_plan_req *, xpar_plan * out);

/*  Compute the decode-side footprint for an existing set.  */
xpar_plan_status xpar_plan_for_repair(const xpar_setd *, u64 recovery_slices,
                                      u64 memory_budget, int threads,
                                      xpar_plan * out);

/*  Calibrated matrix/FFT lost-slice crossover used by the verbose plan.  */
u64 xpar_plan_repair_crossover(u8 fft_codec, u8 field_log2, u64 slices,
                               u64 recovery_slices, u64 slice_size);

void xpar_plan_print(const xpar_plan *, xpar_file * out, bool verbose);

/*  Explain how to resolve XPAR_PLAN_NO_FIT.  */
void xpar_plan_explain_no_fit(const xpar_plan_req *, char * buf, sz cap);

#endif
