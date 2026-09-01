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

/*  Feature probes for non-x86 and non-ARM targets.  */

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
