/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; version 3 of the License only.  */

#include "t_harness.h"
#include "kernel/crc32c.h"
#include "kernel/gf.h"

int xpar_main(int argc, char ** argv) {
  int i;

  for (i = 1; i < argc; i++) {
    if (!xpar_strncmp(argv[i], "--report=", 9) &&
        !xt_open_report(argv[i] + 9)) {
      xpar_fprintf(xpar_stderr, "t_suite: cannot open report '%s'\n",
                   argv[i] + 9);
      return 1;
    }
  }

  xt_level_from_env(xpar_getenv("XPAR_TEST_LEVEL"));
  xt_trace_from_env(xpar_getenv("XPAR_TEST_TRACE"));
  xpar_gf_init();
  xpar_crc32c_init();

  xt_run_unit();
  xt_run_codec();
  xt_run_central();
  xt_run_functional(argc, argv);

  return xt_finish("t_suite");
}
