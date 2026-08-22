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

/*  xpar: authenticated key-file loading interface.  */
#ifndef XPAR_AUTH_H
#define XPAR_AUTH_H

#include "container.h"

typedef enum {
  XPAR_KEYFILE_OK = 0,
  XPAR_KEYFILE_OPEN,
  XPAR_KEYFILE_EMPTY,
  XPAR_KEYFILE_READ
} xpar_keyfile_status;

xpar_keyfile_status xpar_keyfile_load(const char * path, xpar_key * key,
                                      u8 master[XPAR_BLAKE3_KEY_LEN]);

void xpar_key_forget(xpar_key * key,
                     u8 master[XPAR_BLAKE3_KEY_LEN]);

#endif
