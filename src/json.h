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

/*  Streaming JSON Lines output.  */

#ifndef XPAR_JSON_H
#define XPAR_JSON_H

#include "common.h"

#define XPAR_JSON_SCHEMA 1

typedef struct {
  xpar_file * out;
  u64         start_usec;
  bool        enabled;
  bool        in_object;
  bool        need_comma;
} xpar_json;

void xpar_json_init(xpar_json *, xpar_file * out, bool enabled);

void xpar_json_begin(xpar_json *, const char * type);
void xpar_json_end  (xpar_json *);

void xpar_json_str (xpar_json *, const char * key, const char * val);
void xpar_json_strn(xpar_json *, const char * key, const char * val, sz n);
void xpar_json_u64 (xpar_json *, const char * key, u64 val);
void xpar_json_i64 (xpar_json *, const char * key, i64 val);
void xpar_json_bool(xpar_json *, const char * key, bool val);
void xpar_json_null(xpar_json *, const char * key);

void xpar_json_hex(xpar_json *, const char * key, const u8 * p, sz n);

void xpar_json_progress(xpar_json *, u64 done, u64 total, u64 rate_bps);

void xpar_json_progress_sink(void * user, u64 done, u64 total, u64 rate_bps);
void xpar_json_summary (xpar_json *, const char * status, int exit_code);

#endif
