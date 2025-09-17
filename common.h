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

#ifndef _COMMON_H_
#define _COMMON_H_

#include "config.h"

#if !defined(HAVE_STDINT_H)
  #error "Here, have a nickel kid. Go buy yourself a real computer."
#endif

// This often gets us asprintf.
#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

#define MiB(x) ((x) * 1024 * 1024)
#define KiB(x) ((x) * 1024)

#define FATAL(msg, ...) do { \
  fprintf(stderr, msg "\n", ##__VA_ARGS__); exit(1); \
} while (0)
#define FATAL_UNLESS(msg, cond, ...) do { \
  fprintf(stderr, msg "\n", ##__VA_ARGS__); if(cond) exit(1); \
} while (0)
#define FATAL_PERROR(func) do { perror(func); exit(1); } while (0)

#define Fi(n, ...) for (int i = 0; i < (n); i++) { __VA_ARGS__; }
#define Fj(n, ...) for (int j = 0; j < (n); j++) { __VA_ARGS__; }
#define Fk(n, ...) for (int k = 0; k < (n); k++) { __VA_ARGS__; }
#define Fi0(n, s, ...) for (int i = s; i < (n); i++) { __VA_ARGS__; }
#define Fj0(n, s, ...) for (int j = s; j < (n); j++) { __VA_ARGS__; }
#define Fk0(n, s, ...) for (int k = s; k < (n); k++) { __VA_ARGS__; }

#define MIN(a, b) ((a) < (b) ? (a) : (b))
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32;
typedef int8_t i8; typedef int16_t i16; typedef int32_t i32;
typedef size_t sz;

#endif
