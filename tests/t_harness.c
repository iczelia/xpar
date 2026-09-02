/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; version 3 of the License only.  */

#include "t_harness.h"

int xt_level = 1;
int xt_tracing;
u64 xt_checks, xt_failures;
const char * xt_section = "(none)";
xpar_file * xt_report_file;

bool xt_open_report(const char * path) {
  xt_report_file = xpar_open(path, XPAR_O_WRONLY | XPAR_O_CREAT | XPAR_O_TRUNC);
  return xt_report_file != NULL;
}

void xt_close_report(void) {
  if (xt_report_file) (void) xpar_close(xt_report_file);
  xt_report_file = NULL;
}
