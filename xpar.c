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

#include "config.h"
#include "common.h"
#include "jmode.h"
#include "vmode.h"
#include "lmode.h"
#include "cmode.h"
#include "platform.h"
#include "yarg.h"

#include <limits.h>
#include <assert.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(XPAR_OPENMP)
#include <omp.h>
#endif

// ============================================================================
//  Command-line stub.
// ============================================================================
static void version() {
  printf(
    "xpar %d.%d. Copyright (C) by Kamila Szewczyk, 2022-2025.\n"
    "Licensed under the terms of GNU GPL version 3, available online at:\n"
    " <https://www.gnu.org/licenses/gpl-3.0.en.html>\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.\n",
    XPAR_MAJOR, XPAR_MINOR
  );
}
static void help() {
  printf(
    "xpar - an error/erasure code system guarding data integrity.\n"
    "Usage (joint mode):\n"
    "  xpar -Je/-Jd [...] <in>                    (adds/removes .xpa)\n"
    "  xpar -Je/-Jd [...] <in> <out>              (produces <out>)\n"
    "Usage (sharded mode):\n"
    "  xpar -[SWLC]e [...] <in>                   (produces <in>.xpa.XXX)\n"
    "  xpar -[SWLC]e --out-prefix=# [...] <in>    (produces #.xpa.XXX)\n"
    "  xpar -[SWLC]d [...] <out> <in>.001 ...     (produces <out>)\n"
    "\n"
    "Mode selection:\n"
    "  -J,   --joint        use the joint mode (default)\n"
    "  -W,   --van-sharded  use the (Vandermonde) sharded mode\n"
    "  -L,   --fft-sharded  use the (FFT) sharded mode\n"
    "  -C,   --cau-sharded  use the (Cauchy) sharded mode\n"
    "  -e,   --encode       add parity bits to a specified file\n"
    "  -d,   --decode       recover the original data\n"
    "Options:\n"
    "  -h,   --help         display an usage overview\n"
    "  -f,   --force        force operation: ignore errors, overwrite files\n"
    "  -v,   --verbose      verbose mode (display more information)\n"
    "  -q,   --quiet        quiet mode (display less information)\n"
    "  -V,   --version      display version information\n"
#if defined(XPAR_ALLOW_MAPPING)
    "        --no-mmap      unconditionally disable memory mapping\n"
#endif
#if defined(XPAR_OPENMP)
    "  -j #, --jobs=#       set the number of threads to use\n"
#endif
    "Joint mode only:\n"
    "  -c,   --stdout       force writing to standard output\n"
    "  -i #, --interlace=#  change the interlacing setting (1,2,3)\n"
    "Sharded mode encoding options:\n"
    "        --dshards=#    set the number of data shards (< 128)\n"
    "        --pshards=#    set the number of parity shards (< 64)\n"
    "        --out-prefix=# set the output file prefix\n"
    "\n"
    "In joint mode:\n"
    "Interlacing transposes the input data into larger blocks allowing\n"
    "the codes to accomplish higher burst error correction capabilities.\n"
    "The default interlacing factor is 1, which means no interlacing.\n"
    "The interlacing factor of two allows correction of 4080 contiguous\n"
    "errors in a 65025 byte block.\n"
    "\n"
    "Report bugs to: https://github.com/iczelia/xpar\n"
    "Or contact the author: Kamila Szewczyk <k@iczelia.net>\n"
  );
}
enum mode_t { MODE_NONE, MODE_ENCODING, MODE_DECODING };
enum { FLAG_NO_MMAP = CHAR_MAX + 1, FLAG_DSHARDS, FLAG_PSHARDS,
        FLAG_OUT_PREFIX };
static const yarg_options opt[] = {
  { 'V', no_argument, "version" },
  { 'v', no_argument, "verbose" },
  { 'J', no_argument, "joint" },
  { 'W', no_argument, "van-sharded" },
  { 'L', no_argument, "fft-sharded" },
  { 'C', no_argument, "cau-sharded" },
#if defined(XPAR_OPENMP)
  { 'j', required_argument, "jobs" },
#endif
  { 'c', no_argument, "stdout" },
  { 'q', no_argument, "quiet" },
  { 'h', no_argument, "help" },
  { 'e', no_argument, "encode" },
  { 'd', no_argument, "decode" },
  { 'f', no_argument, "force" },
  { FLAG_DSHARDS, required_argument, "dshards" },
  { FLAG_PSHARDS, required_argument, "pshards" },
  { FLAG_OUT_PREFIX, required_argument, "out-prefix" },
#if defined(XPAR_ALLOW_MAPPING)
  { FLAG_NO_MMAP, no_argument, "no-mmap" },
#endif
  { 'i', required_argument, "interlacing" },
  { 0, 0, NULL }
};
int main(int argc, char * argv[]) {
  jmode_gf256_gentab(0x87);  smode_gf256_gentab(0x87);  lmode_gentab();
  cmode_gf256_gentab(0x87);  platform_init();
  yarg_settings settings = { .style = YARG_STYLE_UNIX, .dash_dash = true };
  bool verbose = false, quiet = false, force = false, force_stdout = false;
  bool no_map = false, joint = false, sharded = false, log_sharded = false;
  int mode = MODE_NONE, interlacing = -1, dshards = -1, pshards = -1;
  int jobs = -1;
  const char * out_prefix = NULL;
  yarg_result * res = yarg_parse(argc, argv, opt, settings);
  if (res->error) { fputs(res->error, stderr); exit(1); }
  for (int i = 0; i < res->argc; i++) {
    yarg_option o = res->args[i];
    switch (o.opt) {
      case 'V': version(); return 0;
      case 'j':
        if (jobs != -1) goto conflict;  jobs = atoi(o.arg); break;
      case 'J':
        if (sharded || log_sharded) goto conflict;  joint = true; break;
      case 'W':
        if (joint || log_sharded) goto conflict;  sharded = true; break;
      case 'L':
        if (joint || sharded) goto conflict;  log_sharded = true; break;
      case 'v':
        if (quiet) goto conflict;  verbose = true; break;
      case 'q':
        if (verbose) goto conflict;  quiet = true; break;
      case 'h': help(); return 0;
      case 'e':
        if(mode == MODE_NONE) mode = MODE_ENCODING;
        else goto opmode_conflict;  break;
      case 'd':
        if(mode == MODE_NONE) mode = MODE_DECODING;
        else goto opmode_conflict;  break;
      case 'f': force = true; break;
      case 'c': force_stdout = true; break;
      case FLAG_NO_MMAP: no_map = true; break;
      case 'i':
        interlacing = atoi(o.arg);
        if (interlacing < 1 || interlacing > 3)
          FATAL("Invalid interlacing factor.");
        break;
      case FLAG_DSHARDS:
        dshards = atoi(o.arg);
        if (dshards < 1 || dshards >= MAX_DATA_SHARDS)
          FATAL("Invalid number of data shards.");
        break;
      case FLAG_PSHARDS:
        pshards = atoi(o.arg);
        if (pshards < 1 || pshards >= MAX_PARITY_SHARDS)
          FATAL("Invalid number of parity shards.");
        break;
      case FLAG_OUT_PREFIX: out_prefix = o.arg; break;
      default: exit(1); break;
      conflict: FATAL("Conflicting options.");
      opmode_conflict: FATAL("Multiple operation modes specified.");
    }
  }
#if defined(XPAR_OPENMP)
  if (jobs > 0) omp_set_num_threads(jobs);
#endif
  if (mode == MODE_NONE)
    FATAL("No operation mode specified.");
  if (!joint && !sharded && !log_sharded) joint = true;
  if (joint) {
    if (dshards != -1 || pshards != -1 || out_prefix)
      FATAL("Sharded mode options in joint mode.");
    if (interlacing == -1) interlacing = 1;
    char * f1 = NULL, * f2 = NULL;
    switch (res->pos_argc) {
      case 0: break;
      case 1: f1 = res->pos_args[0]; break;
      case 2: f1 = res->pos_args[0], f2 = res->pos_args[1]; break;
      default: FATAL("Too many positional arguments.");
    }
    char * input_file = NULL, * output_file = NULL;
    if (f1) switch(mode) {
      case MODE_ENCODING:
        if (!f2) {
          input_file = f1;
          if (force_stdout) output_file = NULL;
          else asprintf(&output_file, "%s.xpa", f1);
        } else input_file = f1, output_file = f2;
        break;
      case MODE_DECODING:
        if (!f2) {
          input_file = f1;
          if (force_stdout) output_file = NULL;
          else {
            sz len = strlen(f1);
            if (len < 5 || strcmp(f1 + len - 4, ".xpa"))
              FATAL("Unknown file type.");
            output_file = strndup(f1, len - 4);
          }
        } else input_file = f1, output_file = f2;
        break;
    }
    joint_options_t options = {
      .input_name = input_file, .output_name = output_file,
      .interlacing = interlacing,
      .force = force, .quiet = quiet, .verbose = verbose,
      .no_map = no_map
    };
    volatile struct timeval start, end;
    gettimeofday((struct timeval *) &start, NULL);
    switch(mode) {
      case MODE_ENCODING: do_joint_encode(options); break;
      case MODE_DECODING: do_joint_decode(options); break;
    }
    gettimeofday((struct timeval *) &end, NULL);
    if (verbose) {
      double elapsed = (end.tv_sec - start.tv_sec) +
        (end.tv_usec - start.tv_usec) / 1000000.0;
      printf("Elapsed time: %.6f seconds.\n", elapsed);
    }
    if (output_file != f2) free(output_file);
  } else if(sharded || log_sharded) {
    if (interlacing != -1 || force_stdout)
      FATAL(sharded ? "Joint mode options in sharded mode."
                    : "Joint mode options in log-sharded mode.");
    volatile struct timeval start, end;
    gettimeofday((struct timeval *) &start, NULL);
    switch(mode) {
      case MODE_ENCODING: {
        if (dshards == -1 || pshards == -1)
          FATAL("Number of data and parity shards not specified.");
        if (res->pos_argc == 0) FATAL("No input file specified.");
        const char * input_file = res->pos_args[0];
        if (res->pos_argc > 1) FATAL("Too many positional arguments.");
        if (!out_prefix) out_prefix = input_file;
        sharded_encoding_options_t opt = {
          .input_name = input_file, .output_prefix = out_prefix,
          .dshards = dshards, .pshards = pshards,
          .force = force, .quiet = quiet, .verbose = verbose,
          .no_map = no_map
        };
        if (log_sharded) log_sharded_encode(opt); else sharded_encode(opt);
        break;
      }
      case MODE_DECODING: {
        if (res->pos_argc == 0) FATAL("No output file specified.");
        const char * output_file = res->pos_args[0];
        if (res->pos_argc < 2) FATAL("No input shards specified.");
        sharded_decoding_options_t opt = {
          .output_file = output_file,
          .input_files = (const char **) res->pos_args + 1,
          .force = force, .quiet = quiet, .verbose = verbose,
          .no_map = no_map, .n_input_shards = res->pos_argc - 1
        };
        if (log_sharded) log_sharded_decode(opt); else sharded_decode(opt);
        break;
      }
    }
    gettimeofday((struct timeval *) &end, NULL);
    if (verbose) {
      double elapsed = (end.tv_sec - start.tv_sec) +
        (end.tv_usec - start.tv_usec) / 1000000.0;
      printf("Elapsed time: %.6f seconds.\n", elapsed);
    }
  }
  yarg_destroy(res);
}
