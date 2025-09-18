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
#include "crc32c.h"

#include <assert.h>

// ============================================================================
//  General shared (Vandermonde, Log-time) mode encoding and decoding
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

static sharded_hv_result_t validate_shard_header(const char * file_name,
    sharded_decoding_options_t opt, const char * hdr) {
  sharded_hv_result_t res;  memset(&res, 0, sizeof(res));
  if (!opt.no_map) {
    #if defined(XPAR_ALLOW_MAPPING)
    mmap_t map = xpar_map(file_name);
    if (map.map) {
      if (map.size < SHARD_HEADER_SIZE) {
        xpar_unmap(&map);  return res;
      }
      if (memcmp(map.map, hdr, 4)) {
        xpar_unmap(&map);  return res;
      }
      Fi(4, res.crc |= ((sz) map.map[4 + i]) << (24 - 8 * i))
      res.dshards = map.map[8];
      res.pshards = map.map[9];
      res.shard_number = map.map[10];
      Fi(8, res.total_size |= ((sz) map.map[11 + i]) << (56 - 8 * i))
      if (SIZEOF_SIZE_T == 4) {
        if (map.map[11] || map.map[12] || map.map[13] || map.map[14]) {
          if(!opt.quiet)
            fprintf(stderr,
              "Shard `%s' is too large for 32-bit architectures.\n",
              file_name);
          xpar_unmap(&map);  return res;
        }
      }
      res.shard_size = map.size - SHARD_HEADER_SIZE;
      // Check the CRC.
      u32 crc = crc32c(map.map + SHARD_HEADER_SIZE, res.shard_size);
      if (crc != res.crc) {
        xpar_unmap(&map);  return res;
      }
      res.valid = true;  res.mapped = true;
      res.buf = map.map;   res.map = map;  return res;
    }
    #endif
  }
  FILE * in = fopen(file_name, "rb");
  if (!in) return res;
  fseek(in, 0, SEEK_END);
  sz size = ftell(in);
  fseek(in, 0, SEEK_SET);
  if (size < SHARD_HEADER_SIZE) {
    fclose(in);  return res;
  }
  u8 * buffer = (u8 *) xmalloc(size); // LSP doesn't know it's C.
  if (xfread(buffer, size, in) != size) {
    free(buffer);  fclose(in);  return res;
  }
  fclose(in);
  if (memcmp(buffer, hdr, 4)) {
    free(buffer);  return res;
  }
  Fi(4, res.crc |= ((sz) buffer[4 + i]) << (24 - 8 * i));
  res.dshards = buffer[8];
  res.pshards = buffer[9];
  res.shard_number = buffer[10];
  Fi(8, res.total_size |= ((sz) buffer[11 + i]) << (56 - 8 * i));
  if (SIZEOF_SIZE_T == 4) {
    if (buffer[11] || buffer[12] || buffer[13] || buffer[14]) {
      if(!opt.quiet)
        fprintf(stderr,
          "Shard `%s' is too large for 32-bit architectures.\n",
          file_name);
      free(buffer);  return res;
    }
  }
  res.shard_size = size - SHARD_HEADER_SIZE;
  u32 crc = crc32c(buffer + SHARD_HEADER_SIZE, res.shard_size);
  if (crc != res.crc) {
    free(buffer);  return res;
  }
  res.buf = buffer; res.valid = true;  return res;
}

#endif
