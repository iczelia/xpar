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

#ifndef _LMODE_H_
#define _LMODE_H_

#include "common.h"
#include "sharding.h"

// ============================================================================
//  Linearithmic shared mode encoding and decoding.
// ============================================================================
void lmode_gentab();
void log_sharded_encode(sharded_encoding_options_t o);
void log_sharded_decode(sharded_decoding_options_t o);

#endif
