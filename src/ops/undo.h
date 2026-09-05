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

/*  Shared on-disk journal definitions. A valid footer permits replay.
    XPAR_UNDO_CREATED marks files absent before repair.  */

#ifndef XPAR_UNDO_H
#define XPAR_UNDO_H

#define XPAR_UNDO_MAGIC    "XPARUNDO"
#define XPAR_UNDO_END      "XPARUNDN"
#define XPAR_UNDO_VER      1U
#define XPAR_UNDO_HDR      64U
#define XPAR_UNDO_REC      40U
#define XPAR_UNDO_FOOT     24U
#define XPAR_UNDO_CREATED  1U        /*  Remove a name created by repair.  */
#define XPAR_UNDO_REPLACED 2U        /*  Recreate an independent file.  */
#define XPAR_UNDO_DIRECTORY 4U       /*  CREATED names a directory.  */
#define XPAR_UNDO_FLAGS                                                    \
  (XPAR_UNDO_CREATED | XPAR_UNDO_REPLACED | XPAR_UNDO_DIRECTORY)

#endif
