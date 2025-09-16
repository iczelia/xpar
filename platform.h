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

#ifndef _PLATFORM_H_
#define _PLATFORM_H_

#include "common.h"
#include "config.h"

// ============================================================================
//  Potential shim for asprintf if not present.
// ============================================================================
#if !defined(HAVE_ASPRINTF)
  int asprintf(char **strp, const char *fmt, ...);
#endif

#if !defined(HAVE_STRNDUP)
  char * strndup(const char *s, sz n);
#endif

// ============================================================================
//  Cross-platform (Linux + Windows) file mapping interface.
// ============================================================================
#if defined(HAVE_MMAP) || defined(HAVE_CREATEFILEMAPPINGA)
  #define XPAR_ALLOW_MAPPING 1
  typedef struct { u8 * map; sz size; } mmap_t;
  mmap_t xpar_map(const char *);
  void xpar_unmap(mmap_t *);
#endif

// ============================================================================
//  Platform-specific initialization.
// ============================================================================
void platform_init(void);

// ============================================================================
//  Safe I/O and memory allocation functions: terminate upon error.
//  Also utilise platform-specific behaviours to ensure data integrity.
// ============================================================================
void xfclose(FILE * des);
void xfwrite(const void * ptr, sz size, FILE * stream);
sz xfread(void * ptr, sz size, FILE * stream);
void * xmalloc(sz size);
void notty(FILE * des);
bool is_seekable(FILE * des);

#endif
