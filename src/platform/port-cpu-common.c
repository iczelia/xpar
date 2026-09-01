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

#include "common.h"
#include "port-cpu.h"

static bool g_probed = false;
static u32  g_probed_mask = 0;
static u32  g_allow = XPAR_CPU_ALL;

static u32 probed(void) {
  if (!g_probed) { g_probed_mask = xpar_cpu_probe();  g_probed = true; }
  return g_probed_mask;
}

u32 xpar_cpu_features(void) { return probed() & g_allow; }

void xpar_cpu_force(u32 allow) { (void) probed();  g_allow = allow; }

int xpar_cpu_tier_count(void) { return xpar_cpu_tier_table_n; }

const xpar_cpu_tier * xpar_cpu_tier_at(int i) {
  if (i < 0 || i >= xpar_cpu_tier_table_n) return NULL;
  return &xpar_cpu_tier_table[i];
}

int xpar_cpu_tier_find(const char * name) {
  for (int i = 0; i < xpar_cpu_tier_table_n; i++)
    if (xpar_strcmp(name, xpar_cpu_tier_table[i].name) == 0) return i;
  return -1;
}

bool xpar_cpu_tier_usable(int i) {
  const xpar_cpu_tier * t = xpar_cpu_tier_at(i);
  if (!t) return false;
  return (probed() & t->need) == t->need;
}

int xpar_cpu_tier_best(void) {
  int best = 0;
  for (int i = 0; i < xpar_cpu_tier_table_n; i++)
    if (xpar_cpu_tier_usable(i)) best = i;
  return best;
}
