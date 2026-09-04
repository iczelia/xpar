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

/*  Path names and private staging files.  */

#ifndef XPAR_PATHNAME_H
#define XPAR_PATHNAME_H

#include "common.h"

bool xpar_path_sep(char c);

const char * xpar_path_base(const char * path);

char * xpar_path_dir(const char * path);

char * xpar_path_join  (const char * dir, const char * name);
char * xpar_path_join_n(const char * dir, const char * name, u32 n);

bool xpar_path_ends_with(const char * s, const char * suffix);

/*  Normalize separators and leading "./"; the caller frees the result.  */
char * xpar_path_norm(const char * path);

/*  Whether two normalized path spellings match.  */
bool xpar_path_same(const char * a, const char * b);

/*  Escaped copies live in a ring this many slots deep.  */
#define XPAR_ESCAPE_RING 8

/*  Escape control bytes as \xNN. The rotating buffer owns the result.  */
char * xpar_name_escape(const char * s);

/*  Scan one nonempty decimal run.  */
bool xpar_scan_digits(const char * s, sz * at, sz end);

/*  Set or get the fallback volume directory.  */
void xpar_path_scan_set(const char * dir);
const char * xpar_path_scan(void);

/*  Resolve NAME beside DIR or under --scan; the caller frees it.  */
char * xpar_path_vol(const char * dir, const char * name);

/*  Maximum component length used by staging helpers.  */
#define XPAR_COMPONENT_MAX 255

xpar_file * xpar_stage_open(const char * stem, const char * dos_tag,
                            int flags, int nofollow, char ** out);

char * xpar_stage_dir(const char * stem, const char * dos_tag);

/*  A caller-designed numbered 8.3 name beside PATH.  */
char * xpar_dos_numbered(const char * path, const char * tag,
                         const char * ext, u32 number);

/*  Trim STEM's final component to leave room for a SUFFIX-byte tail.  */
char * xpar_stage_stem(const char * stem, sz suffix);

/*  Keep PATH reachable as BACKUP while replacing it.  */
int xpar_keep_aside(const char * path, const char * backup);
/*  Restore PATH and remove BACKUP.  */
int xpar_put_back(const char * path, const char * backup);

#endif
