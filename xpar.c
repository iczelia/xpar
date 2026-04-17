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

#include "config.h"
#include "common.h"
#include "jmode.h"
#ifdef HAVE_BLAKE2B
  #include "blake2b.h"
#endif
#include "vmode.h"
#include "lmode.h"
#include "yarg.h"

#include <limits.h>

/*  -----------------------------------------------------------------------
  Command-line stub.  */
static void version() {
  xpar_fprintf(xpar_stdout, 
    "xpar %d.%d. Copyright (C) by Kamila Szewczyk, 2022-2026.\n"
    "Licensed under the terms of GNU GPL version 3, available online at:\n"
    " <https://www.gnu.org/licenses/gpl-3.0.en.html>\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.\n",
    XPAR_MAJOR, XPAR_MINOR
  );
}
static void help() {
  xpar_fprintf(xpar_stdout, 
    "xpar - an error/erasure code system guarding data integrity.\n"
    "Usage (joint mode):\n"
    "  xpar -Je/-Jd [...] <in>                    (adds/removes .xpa)\n"
    "  xpar -Je/-Jd [...] <in> <out>              (produces <out>)\n"
    "Usage (sharded mode):\n"
    "  xpar -[SWL]e [...] <in>                   (produces <in>.xpa.XXX)\n"
    "  xpar -[SWL]e --out-prefix=# [...] <in>    (produces #.xpa.XXX)\n"
    "  xpar -[SWL]d [...] <out> <in>.001 ...     (produces <out>)\n"
    "\n"
    "Mode selection:\n"
    "  -J,   --joint        use the joint mode (default)\n"
    "  -W,   --van-sharded  use the (Vandermonde) sharded mode\n"
    "  -L,   --fft-sharded  use the (FFT) sharded mode\n"
    "  -e,   --encode       add parity bits to a specified file\n"
    "  -d,   --decode       recover the original data\n"
    "  -t,   --test         verify integrity without writing output\n"
    "Options:\n"
    "  -h,   --help         display an usage overview\n"
    "  -f,   --force        force operation: ignore errors, overwrite files\n"
    "  -v,   --verbose      verbose mode (display more information)\n"
    "  -q,   --quiet        quiet mode (display less information)\n"
    "  -V,   --version      display version information\n"
#if defined(XPAR_ALLOW_MAPPING)
    "        --no-mmap      unconditionally disable memory mapping\n"
#endif
    "  -j #, --jobs=#       set the number of threads to use\n"
    "Joint mode only:\n"
    "  -c,   --stdout       force writing to standard output\n"
    "  -i #, --interlace=#  change the interlacing setting (1,2,3)\n"
    "  -s,   --systematic   store parity only; caller keeps the original\n"
    "  -H #, --integrity=#  crc32c (default) or blake2b (128-bit tag)\n"
    "        --auth=<file>  key file for keyed-MAC mode (implies -H blake2b)\n"
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
enum mode_t { MODE_NONE, MODE_ENCODING, MODE_DECODING, MODE_TESTING };
enum { FLAG_NO_MMAP = CHAR_MAX + 1, FLAG_DSHARDS, FLAG_PSHARDS,
        FLAG_OUT_PREFIX, FLAG_AUTH };
static const yarg_options opt[] = {
  { 'V', no_argument, "version" },
  { 'v', no_argument, "verbose" },
  { 'J', no_argument, "joint" },
  { 'W', no_argument, "van-sharded" },
  { 'L', no_argument, "fft-sharded" },
  { 'j', required_argument, "jobs" },
  { 'c', no_argument, "stdout" },
  { 'q', no_argument, "quiet" },
  { 'h', no_argument, "help" },
  { 'e', no_argument, "encode" },
  { 'd', no_argument, "decode" },
  { 't', no_argument, "test" },
  { 'f', no_argument, "force" },
  { FLAG_DSHARDS, required_argument, "dshards" },
  { FLAG_PSHARDS, required_argument, "pshards" },
  { FLAG_OUT_PREFIX, required_argument, "out-prefix" },
#if defined(XPAR_ALLOW_MAPPING)
  { FLAG_NO_MMAP, no_argument, "no-mmap" },
#endif
  { 'i', required_argument, "interlacing" },
  { 's', no_argument, "systematic" },
  { 'H', required_argument, "integrity" },
  { FLAG_AUTH, required_argument, "auth" },
  { 0, 0, NULL }
};
int xpar_main(int argc, char ** argv) {
  jmode_gf256_gentab(0x87);  smode_gf256_gentab(0x87);  lmode_gentab();
  yarg_settings settings = { .style = YARG_STYLE_UNIX, .dash_dash = true };
  bool verbose = false, quiet = false, force = false, force_stdout = false;
  bool no_map = false, joint = false, sharded = false, log_sharded = false;
  bool systematic = false;
  int mode = MODE_NONE, interlacing = -1, dshards = -1, pshards = -1;
  int jobs = -1, integrity = INTEGRITY_CRC32C;
  const char * out_prefix = NULL;
  const char * auth_keyfile = NULL;
  yarg_result * res = yarg_parse(argc, argv, opt, settings);
  if (res->error) { xpar_fputs(res->error, xpar_stderr); xpar_exit(1); }
  for (int i = 0; i < res->argc; i++) {
    yarg_option o = res->args[i];
    switch (o.opt) {
      case 'V': version(); return 0;
      case 'j':
        if (jobs != -1) goto conflict;
        { i64 v; if (xpar_parse_i64(o.arg, &v) || v < 0 || v > 1024)
          FATAL("Invalid -j argument."); jobs = (int) v; } break;
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
        if (mode == MODE_NONE) mode = MODE_ENCODING;
        else goto opmode_conflict;  break;
      case 'd':
        if (mode == MODE_NONE) mode = MODE_DECODING;
        else goto opmode_conflict;  break;
      case 't':
        if (mode == MODE_NONE) mode = MODE_TESTING;
        else goto opmode_conflict;  break;
      case 'f': force = true; break;
      case 'c': force_stdout = true; break;
      case FLAG_NO_MMAP: no_map = true; break;
      case 'i': {
        i64 v;
        if (xpar_parse_i64(o.arg, &v) || v < 1 || v > 3)
          FATAL("Invalid interlacing factor.");
        interlacing = (int) v;
        break;
      }
      case FLAG_DSHARDS: {
        i64 v;
        if (xpar_parse_i64(o.arg, &v) || v < 1 || v >= MAX_DATA_SHARDS)
          FATAL("Invalid number of data shards.");
        dshards = (int) v;
        break;
      }
      case FLAG_PSHARDS: {
        i64 v;
        if (xpar_parse_i64(o.arg, &v) || v < 1 || v >= MAX_PARITY_SHARDS)
          FATAL("Invalid number of parity shards.");
        pshards = (int) v;
        break;
      }
      case FLAG_OUT_PREFIX: out_prefix = o.arg; break;
      case 's': systematic = true; break;
      case 'H':
        if (!xpar_strcmp(o.arg, "crc32c")) integrity = INTEGRITY_CRC32C;
        else if (!xpar_strcmp(o.arg, "blake2b")) {
#ifdef HAVE_BLAKE2B
          integrity = INTEGRITY_BLAKE2B;
#else
          FATAL("BLAKE2b support was disabled at configure time.");
#endif
        } else FATAL("Unknown integrity algorithm (use crc32c or blake2b).");
        break;
      case FLAG_AUTH:
#ifdef HAVE_BLAKE2B
        auth_keyfile = o.arg;
        integrity = INTEGRITY_BLAKE2B;
#else
        FATAL("BLAKE2b support was disabled at configure time "
              "(--auth requires it).");
#endif
        break;
      default: xpar_exit(1); break;
      conflict: FATAL("Conflicting options.");
      opmode_conflict: FATAL("Multiple operation modes specified.");
    }
  }
  if (jobs > 0) xpar_set_num_threads(jobs);
  if (mode == MODE_NONE)
    FATAL("No operation mode specified.");
  if (!joint && !sharded && !log_sharded) joint = true;
  if (systematic) {
    if (!joint) FATAL("--systematic only applies to joint mode.");
    if (interlacing != -1)
      FATAL("--systematic does not support interlacing.");
    interlacing = 4;
  }
#ifdef HAVE_BLAKE2B
  u8 auth_key[BLAKE2B_KEYBYTES];
#else
  u8 auth_key[1];
#endif
  sz auth_keylen = 0;
#ifdef HAVE_BLAKE2B
  if (auth_keyfile) {
    xpar_file * k = xpar_open(auth_keyfile, XPAR_O_READ);
    if (!k) FATAL_PERROR("fopen");
    xpar_seek(k, 0, XPAR_SEEK_END);
    i64 keylen_raw = xpar_tell(k);
    if (keylen_raw < 0) FATAL_PERROR("ftell");
    auth_keylen = (sz) keylen_raw;
    xpar_seek(k, 0, XPAR_SEEK_SET);
    if (auth_keylen < 1 || auth_keylen > BLAKE2B_KEYBYTES)
      FATAL("--auth key file must be 1..%d bytes.", BLAKE2B_KEYBYTES);
    if (xpar_xread(k, auth_key, auth_keylen) != auth_keylen)
      FATAL("Short read on key file.");
    xpar_close(k);
  }
#endif
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
          else output_file = xpar_derive_name(f1, ".xpa");
        } else input_file = f1, output_file = f2;
        break;
      case MODE_DECODING:
      case MODE_TESTING:
        if (!f2) {
          if (systematic) {
            input_file = f1;
            output_file = xpar_derive_name(f1, ".xpa");
          } else {
            input_file = f1;
            if (mode == MODE_DECODING && !force_stdout) {
              sz len = xpar_strlen(f1);
              if (len < 5 || xpar_strcmp(f1 + len - 4, ".xpa"))
                FATAL("Unknown file type.");
              output_file = xpar_strndup(f1, len - 4);
            }
          }
        } else input_file = f1, output_file = f2;
        break;
    }
    joint_options_t options = {
      .input_name = input_file, .output_name = output_file,
      .interlacing = interlacing,
      .integrity = integrity,
      .auth_key = auth_keylen ? auth_key : NULL,
      .auth_keylen = auth_keylen,
      .force = force, .quiet = quiet, .verbose = verbose,
      .no_map = no_map
    };
    u64 t_start = xpar_usec_now();
    switch(mode) {
      case MODE_ENCODING: do_joint_encode(options); break;
      case MODE_DECODING: do_joint_decode(options); break;
      case MODE_TESTING:  do_joint_test(options);   break;
    }
    u64 t_end = xpar_usec_now();
    if (verbose) {
      u64 us = t_end - t_start;
      xpar_fprintf(xpar_stdout, "Elapsed time: %llu.%06llu seconds.\n",
                   (unsigned long long)(us / 1000000ULL),
                   (unsigned long long)(us % 1000000ULL));
    }
    if (output_file != f2) xpar_free(output_file);
  } else if (sharded || log_sharded) {
    if (interlacing != -1 || force_stdout)
      FATAL(sharded ? "Joint mode options in sharded mode."
                    : "Joint mode options in log-sharded mode.");
    u64 t_start = xpar_usec_now();
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
          .no_map = no_map,
          .integrity = integrity,
          .auth_key = auth_keylen ? auth_key : NULL,
          .auth_keylen = auth_keylen
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
          .no_map = no_map, .n_input_shards = res->pos_argc - 1,
          .auth_key = auth_keylen ? auth_key : NULL,
          .auth_keylen = auth_keylen
        };
        if (log_sharded) log_sharded_decode(opt); else sharded_decode(opt);
        break;
      }
      case MODE_TESTING: {
        if (res->pos_argc == 0) FATAL("No input shards specified.");
        sharded_decoding_options_t opt = {
          .output_file = NULL,
          .input_files = (const char **) res->pos_args,
          .force = force, .quiet = quiet, .verbose = verbose,
          .no_map = no_map, .n_input_shards = res->pos_argc,
          .auth_key = auth_keylen ? auth_key : NULL,
          .auth_keylen = auth_keylen
        };
        if (log_sharded) log_sharded_test(opt); else sharded_test(opt);
        break;
      }
    }
    u64 t_end = xpar_usec_now();
    if (verbose) {
      u64 us = t_end - t_start;
      xpar_fprintf(xpar_stdout, "Elapsed time: %llu.%06llu seconds.\n",
                   (unsigned long long)(us / 1000000ULL),
                   (unsigned long long)(us % 1000000ULL));
    }
  }
  yarg_destroy(res);
  return 0;
}
