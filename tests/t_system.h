/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; version 3 of the License only.  */

#ifndef XPAR_T_SYSTEM_H
#define XPAR_T_SYSTEM_H

#include "t_harness.h"
#include "platform/port-fs.h"

#define XT_PATH_MAX 1024

typedef struct {
  char * cwd;
  char * root;
  char * seed;
  char * xpar;
  char out[XT_PATH_MAX];
  char err[XT_PATH_MAX];
  u32 case_no;
  int keep;
  bool target_link_identity;
} xt_context;

bool xt_context_init(xt_context *, int argc, char ** argv);
void xt_context_free(xt_context *);

bool xt_path(char *, sz, const char *, const char *);
bool xt_remove_tree(const char *);
bool xt_mkdir(const char *);
bool xt_case_begin(xt_context *, char *, sz, const char *);

bool xt_write_pattern(const char *, u64 bytes, u64 seed);
bool xt_copy_file(const char *, const char *);
bool xt_copy_tree(const char *, const char *);
bool xt_files_equal(const char *, const char *);
bool xt_trees_equal(const char *, const char *);
bool xt_file_size(const char *, u64 *);
bool xt_damage(const char *, u64 offset, u64 bytes, u64 seed);
bool xt_truncate(const char *, u64 bytes);
bool xt_file_contains(const char *, const char *);
bool xt_file_contains_ci(const char *, const char *);
bool xt_json_u64(const char *, const char *, u64 *);
bool xt_json_string(const char *, const char *, const char *);
void xt_dump_file(const char *, const char *);
bool xt_find_file(const char *, const char * prefix, const char * suffix,
                  char *, sz);

int xt_run_xpar(xt_context *, const char * cwd,
                const char * const * args);
void xt_run_recovery(xt_context *);

#endif
