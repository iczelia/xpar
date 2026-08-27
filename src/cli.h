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

/*  Parsed command-line interface.  */

#ifndef XPAR_CLI_H
#define XPAR_CLI_H

#include "xpar2.h"

typedef enum {
  XPAR_VERB_NONE = 0,
  XPAR_VERB_CREATE,      XPAR_VERB_VERIFY,      XPAR_VERB_REPAIR,
  XPAR_VERB_SCRUB,       XPAR_VERB_EXTRACT,     XPAR_VERB_RECOVER,
  XPAR_VERB_ADDRECOVERY, XPAR_VERB_ADD,         XPAR_VERB_CONSOLIDATE,
  XPAR_VERB_PRUNE,       XPAR_VERB_LIST,        XPAR_VERB_INFO,
  XPAR_VERB_EXPLAIN,     XPAR_VERB_UNDO,        XPAR_VERB_RECOVER_PROLOGUE,
  XPAR_VERB_BENCHMARK
} xpar_verb;

const char * xpar_verb_name(xpar_verb v);

/*  Option value spaces.  */

#define XPAR_CLI_AUTO (-1)   /*  Any enumerated field with an auto mode.  */

enum { XPAR_ARMOUR_NONE = 0, XPAR_ARMOUR_METADATA, XPAR_ARMOUR_ALL };
enum { XPAR_VOLS_LADDER = 0, XPAR_VOLS_EQUAL, XPAR_VOLS_FIXED };
enum { XPAR_COLOR_AUTO = 0, XPAR_COLOR_ALWAYS, XPAR_COLOR_NEVER };
enum { XPAR_PROGRESS_AUTO = 0, XPAR_PROGRESS_ON, XPAR_PROGRESS_OFF };
enum { XPAR_RESYNC_OFF = 0, XPAR_RESYNC_AUTO, XPAR_RESYNC_ALWAYS };
enum { XPAR_RESCAN_STAT = 0, XPAR_RESCAN_HASH, XPAR_RESCAN_NONE };
enum { XPAR_SCOPE_GENERATION = 0, XPAR_SCOPE_CHAIN };
enum { XPAR_OWNERMAP_NAME = 0, XPAR_OWNERMAP_NUMERIC };

enum { XPAR_DEST_DEFAULT = 0, XPAR_DEST_IN_PLACE, XPAR_DEST_TO,
       XPAR_DEST_BACKUP };

#define XPAR_PRES_MTIME     (1u << 0)
#define XPAR_PRES_ATIME     (1u << 1)
#define XPAR_PRES_CTIME     (1u << 2)
#define XPAR_PRES_BTIME     (1u << 3)
#define XPAR_PRES_MODE      (1u << 4)
#define XPAR_PRES_SETID     (1u << 5)
#define XPAR_PRES_ATTRS     (1u << 6)
#define XPAR_PRES_OWNER     (1u << 7)
#define XPAR_PRES_XATTR     (1u << 8)
#define XPAR_PRES_XATTR_ALL (1u << 9)
#define XPAR_PRES_LINKS     (1u << 10)

#define XPAR_PRES_TIMES (XPAR_PRES_MTIME | XPAR_PRES_ATIME | \
                         XPAR_PRES_CTIME | XPAR_PRES_BTIME)
#define XPAR_PRES_ALL   (XPAR_PRES_TIMES | XPAR_PRES_MODE | \
                         XPAR_PRES_SETID | XPAR_PRES_ATTRS | \
                         XPAR_PRES_OWNER | XPAR_PRES_XATTR | \
                         XPAR_PRES_XATTR_ALL | XPAR_PRES_LINKS)

#define XPAR_PRES_DEFAULT (XPAR_PRES_MTIME | XPAR_PRES_MODE | \
                           XPAR_PRES_ATTRS | XPAR_PRES_LINKS)

/*  Host metadata omitted by default under --reproducible.  */
#define XPAR_PRES_HOSTDEP (XPAR_PRES_TIMES | XPAR_PRES_OWNER | \
                           XPAR_PRES_XATTR | XPAR_PRES_XATTR_ALL)

typedef enum {
  XPAR_R_NONE = 0,
  XPAR_R_COUNT,     /*  `-r 100`: recovery slices.  */
  XPAR_R_PERCENT,   /*  `-r 15.7%`: of S, rounded to nearest, min 1.  */
  XPAR_R_BYTES,     /*  `-r 2.5M`: absolute, ceil(bytes / Z) slices.  */
  XPAR_R_TIMES      /*  `-r 1x`: S * x slices.  */
} xpar_rkind;

typedef struct {
  xpar_rkind kind;
  u64 count;   /*  COUNT: slices. BYTES: bytes. Zero otherwise.  */
  f64 factor;  /*  PERCENT: the percentage. TIMES: the multiplier.  */
} xpar_rspec;

typedef struct {
  bool by_id;         /*  true: id_prefix identifies it, not number.  */
  u64  number;
  char * id_prefix;   /*  Lower-case hex prefix of the 16-byte set_id.  */
} xpar_genref;

typedef struct {
  char ** vol;    /*  Candidate index-volume paths, in a stable order.  */
  u32     count;
  char *  base;   /*  Derived base name, or NULL for a directory scan.  */
  char *  dir;    /*  The scanned directory, or NULL.  */
} xpar_setref;

/*  The parsed command line.  */

typedef struct {
  xpar_verb verb;
  bool bare_set;         /*  From the `xpar <set-file>` shorthand.  */

  int  verbose;          /*  Repeat count: -vv is 2.  */
  bool quiet, force, json, reproducible;
  int  jobs;             /*  0: one per core, decided by the runner.  */
  u64  memory;           /*  0: derived from physical memory.  */
  int  progress;         /*  XPAR_PROGRESS_*  */
  int  color;            /*  XPAR_COLOR_*  */
  bool human_stderr;     /*  --json moves human output off stdout.  */
  char * simd;           /*  --simd tier name; NULL means auto.  */

  char *  set;
  char ** paths;
  u32     path_count;
  bool    from_stdin;    /*  A lone `-` among the inputs.  */
  xpar_setref set_ref;   /*  Resolved for every verb that takes a set.  */

  /*  create, and by inheritance add and consolidate.  */
  char * output;
  xpar_rspec recovery, max_recovery;
  u64  min_recovery;
  u64  slice_size;       /*  -s; excludes -b.  */
  u64  slices;           /*  -b; excludes -s.  */
  u64  cell_bytes;       /* --cell; 0 selects automatic Y. */
  int  layout;           /*  XPAR_LAYOUT_*  */
  bool layout_given;     /*  Explicit --layout.  */
  int  codec;            /*  XPAR_CODEC_* or XPAR_CLI_AUTO.  */
  int  field;            /*  8, 16 or XPAR_CLI_AUTO.  */
  int  align;            /*  XPAR_ALIGN_*  */
  int  slice_tag;        /*  0, 8 or 16 bytes.  */
  int  armour;           /*  XPAR_ARMOUR_*  */
  int  armour_field;     /*  8, 16 or XPAR_CLI_AUTO.  */
  u32  armour_t;         /*  0 when unset; excludes armour_pct.  */
  f64  armour_pct;       /*  0 when unset; 0 < p <= 50.  */
  u64  burst;            /*  0 when unset; excludes depth.  */
  u32  depth;            /*  0 when unset; the planner's default is 1.  */
  int  volumes;          /*  XPAR_VOLS_*  */
  u32  volume_count;     /*  --volumes=N.  */
  int  dedup;            /*  XPAR_DEDUP_*  */
  u64  dedup_chunk, dedup_memory, dedup_max_refs;
  u32  preserve;         /*  XPAR_PRES_* mask.  */
  u32  preserve_explicit;/*  Positive tokens named on the command line.  */
  u32  require;          /*  XPAR_PRES_* mask; 0 means degrade.  */
  bool recurse, follow_symlinks, labels;
  bool auth_only, no_verify_after, spool;
  char ** exclude;  u32 exclude_count;
  char ** include;  u32 include_count;
  char * base_dir;
  char * auth_key;
  char * spool_dir;      /*  NULL with spool set: the host's temp dir.  */
  char * stdin_name;     /*  Required manifest path for a `-' input.  */

  /*  verify.  */
  bool fast, strong, resync_exhaustive;
  int  resync;           /*  XPAR_RESYNC_*  */
  u32  resync_step;
  u64  resync_window;
  char * scan_dir;

  /*  repair.  */
  int  dest;             /*  XPAR_DEST_*  */
  char * to_dir;         /*  --to, shared with extract and recover.  */
  bool paranoid, keep_journal, no_journal, dry_run, exit_on_change;
  /*  Internal to a --chain walk: repair generation N's code against the
      effective head manifest, whose extents identify current file offsets.  */
  bool repair_head_set;
  char * repair_head_id;
  /*  Private root containing materialised ancestor generations during an
      owned-layout `repair --chain --to` walk. Not a command-line option.  */
  char * repair_chain_stage;
  bool chain_metadata_only; /*  Internal: do not scan recovery payloads.  */

  /*  scrub.  */
  bool deep, rewrite, rebuild_cells;

  /*  extract.  */
  bool to_stdout, mangle;
  int  owner_map;        /*  XPAR_OWNERMAP_*  */

  /*  recover.  */
  char * volume_name;    /*  Test this first: NULL means --volume was a
                             number, and volume_index holds it.  */
  u64  volume_index;
  bool volume_given;     /*  Numeric zero is a valid LAYT index.  */

  /*  add.  */
  int  rescan;           /*  XPAR_RESCAN_*  */
  int  dedup_scope;      /*  XPAR_SCOPE_*  */
  bool allow_missing;

  /*  consolidate.  */
  bool replace;

  /*  prune and every verb that selects a generation.  */
  xpar_genref * gens;    /*  --generation, repeatable for prune.  */
  u32  gen_count;
  xpar_genref before;    /*  --before; valid when have_before.  */
  bool have_before, chain;

  /*  info.  */
  bool deps;

  /*  list and benchmark reporting selectors.  */
  bool list_links, list_dedup, benchmark_tiers;
} xpar_options;

/*  Entry points.  */

void xpar_cli_parse(int argc, char ** argv, xpar_options * o);
void xpar_cli_free (xpar_options * o);

void xpar_cli_help(xpar_verb v);   /*  XPAR_VERB_NONE: the verb list.  */
void xpar_cli_version(void);

/*  Resolve a set argument to its candidate index volumes.  */
void xpar_cli_resolve_set(const char * arg, xpar_setref * out);
void xpar_setref_free(xpar_setref * s);

/*  Size and redundancy grammars.  */
int xpar_cli_parse_size(const char * s, u64 * out);
int xpar_cli_parse_recovery(const char * s, xpar_rspec * out);

#endif
