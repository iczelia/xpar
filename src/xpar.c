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

/*  Entry point: initialise, parse, dispatch, return.  */

#include "common.h"
#include "cli.h"
#include "kernel/blake3.h"
#include "kernel/crc32c.h"
#include "kernel/gf.h"
#include "ops/ops.h"
#include "platform/port-cpu.h"

typedef int (* xpar_op_fn)(const xpar_options *);

typedef struct {
  xpar_verb  verb;
  xpar_op_fn fn;
} xpar_dispatch;

static const xpar_dispatch dispatch[] = {
  { XPAR_VERB_CREATE,            xpar_op_create           },
  { XPAR_VERB_VERIFY,            xpar_op_verify           },
  { XPAR_VERB_REPAIR,            xpar_op_repair           },
  { XPAR_VERB_SCRUB,             xpar_op_scrub            },
  { XPAR_VERB_EXTRACT,           xpar_op_extract          },
  { XPAR_VERB_RECOVER,           xpar_op_recover          },
  { XPAR_VERB_ADDRECOVERY,       xpar_op_addrecovery      },
  { XPAR_VERB_ADD,               xpar_op_add              },
  { XPAR_VERB_CONSOLIDATE,       xpar_op_consolidate      },
  { XPAR_VERB_PRUNE,             xpar_op_prune            },
  { XPAR_VERB_LIST,              xpar_op_list             },
  { XPAR_VERB_INFO,              xpar_op_info             },
  { XPAR_VERB_EXPLAIN,           xpar_op_explain          },
  { XPAR_VERB_UNDO,              xpar_op_undo             },
  { XPAR_VERB_RECOVER_PROLOGUE,  xpar_op_recover_prologue },
  { XPAR_VERB_BENCHMARK,         xpar_op_benchmark        }
};

static void apply_simd(const char * want) {
  int t;

  if (!want || !xpar_strcmp(want, "auto")) return;

  t = xpar_cpu_tier_find(want);
  if (t < 0) {
    int i, n = xpar_cpu_tier_count();
    xpar_fprintf(xpar_stderr, "xpar: unknown SIMD tier '%s'.\n", want);
    xpar_fputs("xpar: this build has:", xpar_stderr);
    for (i = 0; i < n; i++) {
      const xpar_cpu_tier * ct = xpar_cpu_tier_at(i);
      xpar_fprintf(xpar_stderr, " %s%s", ct->name,
                   xpar_cpu_tier_usable(i) ? "" : " (unsupported here)");
    }
    xpar_fputs("\n", xpar_stderr);
    xpar_exit(XPAR_EXIT_USAGE);
  }
  if (!xpar_cpu_tier_usable(t))
    FATAL("SIMD tier '%s' is unavailable on this machine.", want);

  xpar_cpu_force(xpar_cpu_tier_at(t)->need);
  xpar_gf_use_default_tier();
  (void) xpar_gf_use_tier_name(want);
}

int xpar_main(int argc, char ** argv) {
  xpar_options o;
  sz i;
  int rc;

  xpar_gf_init();
  xpar_crc32c_init();

  xpar_cli_parse(argc, argv, &o);
  /*  XPAR_SIMD forces a tier process-wide; --simd takes precedence.  */
  apply_simd(o.simd ? o.simd : xpar_getenv("XPAR_SIMD"));

  for (i = 0; i < ARRAY_LEN(dispatch); i++)
    if (dispatch[i].verb == o.verb) {
      rc = dispatch[i].fn(&o);
      xpar_cli_free(&o);
      return rc;
    }

  FATAL_CODE(XPAR_EXIT_INTERNAL,
             "internal: verb %d has no implementation in the dispatch "
             "table.", (int) o.verb);
}
