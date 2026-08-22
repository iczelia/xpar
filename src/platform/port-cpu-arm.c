/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License only.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.  */

/*  ARM feature detection and tier selection.  ELF hosts use AT_HWCAP;
    Darwin uses sysctl.  AArch64 always reports mandatory NEON, but probes
    optional extensions.  Unknown hosts safely report scalar features.  */

#include "common.h"
#include "port-cpu.h"

#if defined(__aarch64__) || defined(__arm__) ||                               \
    defined(_M_ARM64) || defined(_M_ARM)

#if defined(HAVE_GETAUXVAL) && defined(HAVE_SYS_AUXV_H)
  #include <sys/auxv.h>
#endif
#if defined(__APPLE__)
  #include <sys/sysctl.h>
  #include <sys/types.h>
#endif
#if defined(_WIN32)
  #include <windows.h>
#endif

/*  AT_HWCAP bit positions, from the Linux kernel: arm64 bit 1 ASIMD and
    bit 7 CRC32 (Documentation/arch/arm64/elf_hwcaps.rst), arm bit 12 NEON
    (arch/arm/include/uapi/asm/hwcap.h). They are ABI and cannot move.  */
#define XPAR_HWCAP64_ASIMD  (1UL << 1)
#define XPAR_HWCAP64_CRC32  (1UL << 7)
#define XPAR_HWCAP64_PMULL  (1UL << 4)
#define XPAR_HWCAP2_SVE2    (1UL << 1)
#define XPAR_HWCAP32_NEON   (1UL << 12)

#if defined(__APPLE__)
static bool sysctl_flag(const char * name) {
  int v = 0;
  size_t n = sizeof v;
  if (sysctlbyname(name, &v, &n, NULL, 0) != 0) return false;
  return v != 0;
}
#endif

u32 xpar_cpu_probe(void) {
  u32 f = 0;

#if defined(__aarch64__) || defined(_M_ARM64)
  f |= XPAR_CPU_NEON;
#endif
  /*  A compiler told to target the extension has already emitted it into
      every object here, so the binary cannot run without it anyway.  */
#if defined(__ARM_FEATURE_CRC32)
  f |= XPAR_CPU_ARMCRC;
#endif
#if defined(__ARM_FEATURE_CRYPTO)
  f |= XPAR_CPU_PMULL;
#endif
#if defined(__ARM_FEATURE_SVE2)
  f |= XPAR_CPU_SVE2;
#endif

#if defined(HAVE_GETAUXVAL) && defined(HAVE_SYS_AUXV_H)
  { unsigned long h = getauxval(AT_HWCAP);
  #if defined(__aarch64__)
    if (h & XPAR_HWCAP64_ASIMD) f |= XPAR_CPU_NEON;
    if (h & XPAR_HWCAP64_CRC32) f |= XPAR_CPU_ARMCRC;
    if (h & XPAR_HWCAP64_PMULL) f |= XPAR_CPU_PMULL;
  #else
    if (h & XPAR_HWCAP32_NEON)  f |= XPAR_CPU_NEON;
  #endif
  }
  #if defined(AT_HWCAP2) && defined(__aarch64__)
  if (getauxval(AT_HWCAP2) & XPAR_HWCAP2_SVE2) f |= XPAR_CPU_SVE2;
  #endif
#elif defined(__APPLE__)
  if (sysctl_flag("hw.optional.neon"))         f |= XPAR_CPU_NEON;
  if (sysctl_flag("hw.optional.armv8_crc32"))  f |= XPAR_CPU_ARMCRC;
  if (sysctl_flag("hw.optional.arm.FEAT_PMULL")) f |= XPAR_CPU_PMULL;
  if (sysctl_flag("hw.optional.arm.FEAT_SVE2"))  f |= XPAR_CPU_SVE2;
#elif defined(_WIN32)
  #if defined(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE)
  if (IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE))
    f |= XPAR_CPU_PMULL;
  #endif
  #if defined(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE)
  if (IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE))
    f |= XPAR_CPU_ARMCRC;
  #endif
#endif

  return f;
}

/*  Cumulative, like the x86 ladder: the crc tier is NEON plus the CRC32
    extension, so forcing it leaves the NEON kernels reachable.  */
const xpar_cpu_tier xpar_cpu_tier_table[] = {
  { "scalar",   0                                   },
#if defined(HAVE_NEON)
  { "neon",     XPAR_CPU_NEON                       },
#endif
#if defined(HAVE_PMULL)
  { "clmul-neon", XPAR_CPU_NEON | XPAR_CPU_PMULL    },
#endif
#if defined(HAVE_SVE2)
  { "sve2",       XPAR_CPU_SVE2                     },
#endif
#if defined(HAVE_ARM_CRC32)
  { "neon-crc", XPAR_CPU_NEON | XPAR_CPU_ARMCRC     },
#endif
  { NULL,       0                                   }
};

const int xpar_cpu_tier_table_n =
  (int) (sizeof xpar_cpu_tier_table / sizeof xpar_cpu_tier_table[0]) - 1;

#else

/*  Not an ARM target; see the note in port-cpu-x86.c.  */
typedef int xpar_cpu_arm_not_here;

#endif
