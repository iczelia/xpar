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

#ifndef _SHARDING_H_
#define _SHARDING_H_

#include "common.h"
#include "platform.h"

#include <assert.h>

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

#define SHARD_HEADER_SIZE 19
typedef struct {
  bool valid, mapped; u32 crc; u8 * buf;
  u8 shard_number, dshards, pshards;
  sz shard_size, total_size;
#if defined(XPAR_ALLOW_MAPPING)
  mmap_t map;
#endif
} sharded_hv_result_t;
static void unmap_shard(sharded_hv_result_t * res) {
  if (res->mapped && res->buf) {
    #if defined(XPAR_ALLOW_MAPPING)
    xpar_unmap(&res->map);
    res->buf = NULL;  res->mapped = false;
    #endif
  } else {
    free(res->buf);  res->buf = NULL;
  }
}
static u8 * most_frequent(u8 * tab, sz nmemb, sz size) {
  assert(size < 16);  u8 tmp[16]; sz i, j;
  Fi0(nmemb, 1,
    for (j = i;
         j && memcmp(&tab[j * size], &tab[(j - 1) * size], size) < 0;
         j--) {
      memcpy(tmp, tab + j * size, size);
      memcpy(tab + j * size, tab + (j - 1) * size, size);
      memcpy(tab + (j - 1) * size, tmp, size);
    }
  )
  u8 * best = tab;  sz best_count = 1, current_count = 1;
  Fi0(nmemb, 1,
    if (!memcmp(tab + i * size, tab + (i - 1) * size, size))
      current_count++;
    else {
      if (current_count > best_count)
        best = tab + (i - 1) * size, best_count = current_count;
      current_count = 1;
    }
  )
  return best;
}

#endif
