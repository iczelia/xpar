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

#ifndef _JMODE_H_
#define _JMODE_H_

#include "common.h"

// ============================================================================
//  Joint mode encoding and decoding.
// ============================================================================
typedef struct {
  const char * input_name, * output_name;
  int interlacing; // 1-3 inclusive.
  bool force, quiet, verbose, no_map;
} joint_options_t;

void jmode_gf256_gentab(u8 poly);
void do_joint_encode(joint_options_t o);
void do_joint_decode(joint_options_t o);

#endif
