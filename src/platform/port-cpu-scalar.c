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

/*  CPU backend for builds without vector kernels.  */

#include "common.h"
#include "port-cpu.h"

u32 xpar_cpu_probe(void) { return 0; }

const xpar_cpu_tier xpar_cpu_tier_table[] = {
  { "scalar", 0 },
  { NULL,     0 }
};

const int xpar_cpu_tier_table_n = 1;
