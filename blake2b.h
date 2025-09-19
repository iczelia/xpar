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

/*  BLAKE2b; adapted from the reference C (CC0, Samuel Neves).  */

#ifndef _BLAKE2B_H_
#define _BLAKE2B_H_

#include "common.h"

#define BLAKE2B_BLOCKBYTES 128
#define BLAKE2B_OUTBYTES    64
#define BLAKE2B_KEYBYTES    64

typedef struct {
  u64 h[8];
  u64 t[2];
  u64 f[2];
  u8 buf[BLAKE2B_BLOCKBYTES];
  sz buflen;
  sz outlen;
} blake2b_state;

int blake2b_init(blake2b_state * restrict s, sz outlen);
int blake2b_init_key(blake2b_state * restrict s, sz outlen,
                     const void * restrict key, sz keylen);
int blake2b_update(blake2b_state * restrict s,
                   const void * restrict in, sz inlen);
int blake2b_final(blake2b_state * restrict s, void * restrict out, sz outlen);
int blake2b(void * restrict out, sz outlen,
            const void * restrict in, sz inlen,
            const void * restrict key, sz keylen);

#endif
