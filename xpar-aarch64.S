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

.text
.extern getauxval

#if defined(__APPLE__)
/* aarch64 MacOS */
.extern _sysctlbyname
.globl _crc32c_aarch64_cpuflags
_crc32c_aarch64_cpuflags:
  sub sp, sp, #32
  stp x29, x30, [sp, #16]
  add x29, sp, #16
  mov w8, #4
  adrp x0, .name@GOTPAGE
  ldr x0, [x0, .name@GOTPAGEOFF]
  sub x1, x29, #4
  mov x2, sp
  mov x3, xzr
  mov x4, xzr
  stur wzr, [x29, #-4]
  str x8, [sp]
  bl _sysctlbyname
  ldur w8, [x29, #-4]
  cmp w0, #0
  ccmp w8, #0, #4, eq
  cset w0, ne
  ldp x29, x30, [sp, #16]
  add sp, sp, #32
  ret
.name: .asciz "hw.optional.armv8_crc32"
#else
/* aarch64 Linux */
.extern getauxval
.globl crc32c_aarch64_cpuflags
crc32c_aarch64_cpuflags:
  stp x29, x30, [sp, -16]!
  mov x0, 16 /* AT_HWCAP */
  mov x29, sp
  bl getauxval
  ldp x29, x30, [sp], 16
  and w0, w0, 128
  ret
#endif

/* Uses fundamentally the same algorithm as crc32c_small_x86_64_sse42 */

#if defined(__APPLE__)
.globl _crc32c_small_aarch64_neon
_crc32c_small_aarch64_neon:
#else
.globl crc32c_small_aarch64_neon
crc32c_small_aarch64_neon:
#endif
  cmp x2, 63
  bls .fallback_1way
  sub x8, x2, #64
  mov x3, x1
  and x7, x8, -64
  add x7, x7, 64
  add x7, x1, x7
.crc32c_8way_quad:
  ldp x4, x6, [x3]
  crc32cx w0, w0, x4
  ldp x4, x5, [x3, 16]
  crc32cx w0, w0, x6
  crc32cx w0, w0, x4
  ldp x4, x6, [x3, 32]
  crc32cx w0, w0, x5
  crc32cx w0, w0, x4
  ldp x5, x4, [x3, 48]
  crc32cx w0, w0, x6
  add x3, x3, 64
  crc32cx w0, w0, x5
  crc32cx w0, w0, x4
  cmp x3, x7
  bne .crc32c_8way_quad
  and x8, x8, -64
  add x1, x1, 64
  add x1, x8, x1
  and x2, x2, 63
.fallback_1way:
  cmp x2, 7
  bls .fallback_1way_byte
  sub x6, x2, #8
  mov x3, x1
  lsr x5, x6, 3
  add w5, w5, 1
  add x5, x1, x5, lsl 3
.crc32c_1way_quad:
  ldr x4, [x3], 8
  crc32cx w0, w0, x4
  cmp x3, x5
  bne .crc32c_1way_quad
  and x6, x6, -8
  add x1, x1, 8
  add x1, x6, x1
  and x2, x2, 7
.fallback_1way_byte:
  cbz x2, .crc32c_done
  add x2, x1, x2
.crc32c_1way_byte:
  ldrb    w3, [x1], 1
  crc32cb w0, w0, w3
  cmp x1, x2
  bne .crc32c_1way_byte
.crc32c_done:
  ret
