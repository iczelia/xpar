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

#ifndef XPAR_VOLNAME_H
#define XPAR_VOLNAME_H

#include "common.h"
#include "xpar2.h"

char * xpar_vname_index   (const char * base, u32 gen);
char * xpar_vname_recovery(const char * base, u32 gen, u64 first, u64 count,
                           int wfirst, int wcount, u32 ordinal);
char * xpar_vname_data    (const char * base, u32 gen, u32 index, int width);

char * xpar_vname_label(const char * data_name);
char * xpar_vname_undo(const char * base, u32 generation);
char * xpar_vname_maint(const char * base);
char * xpar_vname_cache(const char * base);

void xpar_vname_widths(u64 max_first, u64 max_count,
                       int * wfirst, int * wcount);
bool xpar_vname_has_ext(const char * name);
bool xpar_vname_is_undo(const char * name);
bool xpar_vname_is_index (const char * name, const char * stem);
bool xpar_vname_is_member(const char * name, const char * stem);

/*  The generation a member volume name encodes, or -1 when NAME names no
    member of STEM.  */
i64 xpar_vname_gen_of(const char * name, const char * stem);

/*  Whether PATH is an output associated with BASE.  */
bool xpar_vname_is_output(const char * path, const char * base);

#endif
