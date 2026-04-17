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

#ifndef _CRC32C_H_
#define _CRC32C_H_

#include "common.h"

u32 crc32c(const u8 * data, sz length);

/*  Streaming CRC; init with 0xFFFFFFFF, finalize by XORing it back.  */
u32 crc32c_partial(u32 crc, const u8 * data, sz length);

#endif
