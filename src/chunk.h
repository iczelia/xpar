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

#ifndef XPAR_CHUNK_H
#define XPAR_CHUNK_H

#include "common.h"
#include "xpar2.h"

typedef bool (*xpar_chunk_emit)(void *, u64 file_offset, u32 length,
                                const u8 hash[16]);

/*  Deterministic Gear chunks; false if the file cannot be opened.  */
bool xpar_chunk_file(const char * path, u64 average,
                     xpar_chunk_emit emit, void * user,
                     u8 content_hash[32], u8 prefix_hash[16]);

typedef struct {
  u8  hash[16];
  u64 stream_offset;
  u32 length;
  u64 refs;
  u8  trust;
} xpar_chunk_slot;

#ifndef XPAR_CHUNK_INDEX_TYPEDEF
#define XPAR_CHUNK_INDEX_TYPEDEF
typedef struct xpar_chunk_index xpar_chunk_index;
#endif

struct xpar_chunk_index {
  xpar_chunk_slot * slot;
  u32 capacity, count;
  u64 max_bytes;
};

bool xpar_chunk_index_init(xpar_chunk_index *, u64 max_bytes);
void xpar_chunk_index_free(xpar_chunk_index *);
xpar_chunk_slot * xpar_chunk_index_find(xpar_chunk_index *,
                                        const u8 hash[16], u32 length);
bool xpar_chunk_index_put(xpar_chunk_index *, const u8 hash[16], u32 length,
                          u64 stream_offset);

bool xpar_chunk_cache_load(const char * path,
                           const u8 set_id[XPAR_SET_ID_LEN], u64 average,
                           xpar_chunk_index *);
bool xpar_chunk_cache_write(const char * path,
                            const u8 set_id[XPAR_SET_ID_LEN], u64 average,
                            const xpar_chunk_index *);
bool xpar_chunk_cache_rebind(const char * path,
                             const u8 set_id[XPAR_SET_ID_LEN]);

#endif
