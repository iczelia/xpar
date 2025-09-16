/*
   Copyright (C) 2022-2024 Kamila Szewczyk

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

#ifndef _SHARDING_H_
#define _SHARDING_H_

#include "common.h"

// ============================================================================
//  General shared (Vandermonde, Log-time, Gao) mode encoding and decoding
//  utilities.
// ============================================================================
#define MAX_DATA_SHARDS 128
#define MAX_PARITY_SHARDS 64
#define MAX_TOTAL_SHARDS (MAX_DATA_SHARDS + MAX_PARITY_SHARDS)

typedef struct {
  const char * input_name, * output_prefix;
  bool force, quiet, verbose, no_map;
  u8 dshards, pshards;
} sharded_encoding_options_t;

typedef struct {
  const char * output_file, ** input_files;
  bool force, quiet, verbose, no_map;
  sz n_input_shards;
} sharded_decoding_options_t;

#endif
