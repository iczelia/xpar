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

#ifndef _CRC32C_H_
#define _CRC32C_H_

#include "common.h"
#include "config.h"

// Expects:
// - Data to be aligned on a 32-bit boundary.
// - Length to be a multiple of 64.
u32 crc32c(u8 * data, sz length);

#endif
