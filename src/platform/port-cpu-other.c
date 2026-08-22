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

/*  Architecture-independent CPU feature caching and tier control.
    Architecture files supply only the probe and tier table.  */

#include "common.h"
#include "port-cpu.h"

/*  Probed once. The store is plain: dispatch tables are built during
    startup, before any pool exists, and a second probe on a racing thread
    would compute the same answer anyway.  */
static bool g_probed = false;
static u32  g_probed_mask = 0;
static u32  g_allow = XPAR_CPU_ALL;

static u32 probed(void) {
  if (!g_probed) { g_probed_mask = xpar_cpu_probe();  g_probed = true; }
  return g_probed_mask;
}

u32 xpar_cpu_features(void) { return probed() & g_allow; }

/*  Masks down and never up. --simd=gfni512 on a machine without AVX-512
    therefore selects whatever that machine really has, instead of building
    a dispatch table full of instructions that fault on first use.  */
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

/*  Against the probed set and not the forced one: a sweep that has just
    forced `scalar` must still be told that avx2 is runnable, or it stops
    after the first tier.  */
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

#if !(defined(__x86_64__) || defined(__i386__) ||                             \
      defined(_M_X64) || defined(_M_IX86) ||                                  \
      defined(__aarch64__) || defined(__arm__) ||                             \
      defined(_M_ARM64) || defined(_M_ARM))

/*  Every other target: MIPS, POWER, RISC-V, SPARC. The scalar kernels are
    the whole ladder and there is nothing to probe. A DOS build does not
    land here (it is i386, so port-cpu-x86.c claims it) and does not need
    to: configure skips the SIMD probes on DJGPP, so its ladder comes out
    of the x86 file with the scalar row and nothing else.  */

#if defined(__riscv)

#if defined(HAVE_ASM_HWPROBE_H) && defined(HAVE_SYS_SYSCALL_H)
  #include <asm/hwprobe.h>
  #include <sys/syscall.h>
  #include <unistd.h>
#endif

u32 xpar_cpu_probe(void) {
  u32 f = 0;
#if defined(__riscv_vector)
  f |= XPAR_CPU_RVV;
#endif
#if defined(__riscv_zvbc)
  f |= XPAR_CPU_RVVCLMUL;
#endif
#if defined(HAVE_ASM_HWPROBE_H) && defined(HAVE_SYS_SYSCALL_H) && \
    defined(__NR_riscv_hwprobe)
  { struct riscv_hwprobe p;
    p.key = RISCV_HWPROBE_KEY_IMA_EXT_0;  p.value = 0;
    if (syscall(__NR_riscv_hwprobe, &p, 1, 0, NULL, 0) == 0) {
      if (p.value & RISCV_HWPROBE_IMA_V) f |= XPAR_CPU_RVV;
      if (p.value & RISCV_HWPROBE_EXT_ZVBC) f |= XPAR_CPU_RVVCLMUL;
    }
  }
#endif
  return f;
}

const xpar_cpu_tier xpar_cpu_tier_table[] = {
  { "scalar", 0 },
#if defined(HAVE_RVV)
  { "rvv-shuffle", XPAR_CPU_RVV },
#endif
#if defined(HAVE_RVV_CLMUL)
  { "rvv-clmul", XPAR_CPU_RVV | XPAR_CPU_RVVCLMUL },
#endif
  { NULL, 0 }
};

#elif defined(__powerpc__) || defined(__powerpc64__) || defined(_ARCH_PPC)

#if defined(HAVE_GETAUXVAL) && defined(HAVE_SYS_AUXV_H)
  #include <sys/auxv.h>
#endif

#define XPAR_PPC_FEATURE_VSX 0x00000080UL

u32 xpar_cpu_probe(void) {
  u32 f = 0;
#if defined(__VSX__)
  f |= XPAR_CPU_VSX;
#endif
#if defined(HAVE_GETAUXVAL) && defined(HAVE_SYS_AUXV_H)
  if (getauxval(AT_HWCAP) & XPAR_PPC_FEATURE_VSX) f |= XPAR_CPU_VSX;
#endif
  return f;
}

const xpar_cpu_tier xpar_cpu_tier_table[] = {
  { "scalar", 0 },
#if defined(HAVE_VSX)
  { "vsx", XPAR_CPU_VSX },
#endif
  { NULL, 0 }
};

#else

u32 xpar_cpu_probe(void) { return 0; }

const xpar_cpu_tier xpar_cpu_tier_table[] = {
  { "scalar", 0 },
  { NULL,     0 }
};

#endif

const int xpar_cpu_tier_table_n =
  (int) (sizeof xpar_cpu_tier_table / sizeof xpar_cpu_tier_table[0]) - 1;

#endif
