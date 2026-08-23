/*  Copyright (C) 2022-2026 Kamila Szewczyk
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; version 3 of the License only.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <http://www.gnu.org/licenses/>.  */

/*  Merge stderr into stdout for COMMAND.COM and preserve the exit status.
 *
 *      RUN2.EXE XPAR.EXE verify SET.XPA > LOG.TXT                        */

#include <process.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char ** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: run2 PROGRAM [ARGUMENT...]\n");
    return 2;
  }
  if (dup2(1, 2) < 0) {
    perror("run2: dup2");
    return 2;
  }
  return spawnvp(P_WAIT, argv[1], argv + 1);
}
