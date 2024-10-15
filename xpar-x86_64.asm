;   Copyright (C) 2022-2024 Kamila Szewczyk
;
;   This program is free software; you can redistribute it and/or modify
;   it under the terms of the GNU General Public License as published by
;   the Free Software Foundation; either version 3 of the License, or
;   (at your option) any later version.
;
;   This program is distributed in the hope that it will be useful,
;   but WITHOUT ANY WARRANTY; without even the implied warranty of
;   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;   GNU General Public License for more details.
;
;   You should have received a copy of the GNU General Public License
;   along with this program. If not, see <http://www.gnu.org/licenses/>.

section .text

%ifdef MACHO
  global _xpar_x86_64_cpuflags
  global _crc32c_small_x86_64_sse42
  global _rse32_x86_64_generic
  global _rse32_x86_64_avx512
  extern _PROD_GEN
  
  %define xpar_x86_64_cpuflags _xpar_x86_64_cpuflags
  %define crc32c_small_x86_64_sse42 _crc32c_small_x86_64_sse42
  %define rse32_x86_64_generic _rse32_x86_64_generic
  %define rse32_x86_64_avx512 _rse32_x86_64_avx512
  %define PROD_GEN _PROD_GEN
%else
  global xpar_x86_64_cpuflags
  global crc32c_small_x86_64_sse42
  global rse32_x86_64_generic
  global rse32_x86_64_avx512
  extern PROD_GEN
%endif

; The case for this file is very simple: tabular CRC32 is a horrible
; use of the CPU. It causes a lot of cache thrashing and is generally
; somewhat slow. Profiling shows that on the author's CPU,
; AMD Ryzen 7 PRO 7840U, the runtime share of CRC32 goes down from about
; 15% to about 0.5% with this simple implementation.

; Notice that there are better ways to do CRC32 by saturating the
; CPU's execution units. On most architectures, the latency of CRC32 is 3,
; while the throughput is 1. This means that you can do 3 CRC32s in parallel.
; This is not done here (yet), but it is rather easy to do. Another improvement
; stems from the use of the PCLMULQDQ instruction, which can be used to
; calculate CRC32 in parallel with the dedicated instruction, further
; increasing throughput to about 70-80GiB/s.

; Probe the CPU features, keeping in mind that the OS can disable
; AVX512F and AVX512VL support. The return value is a bitfield:
; rax & (1 << 0) - SSE4.2 support.    rax & (1 << 1) - PCLMULQDQ support.
; rax & (1 << 2) - AVX512F support.   rax & (1 << 3) - AVX512VL support.
xpar_x86_64_cpuflags:
  push rbx
  mov eax, 1
  xor ecx, ecx
  cpuid
  mov eax, ecx
  shr eax, 20
  and eax, 0x01
  mov esi, ecx
  and esi, 0x02
  or esi, eax
  not ecx
  test ecx, 0x18000000
  jne .no_osxsave
  xor ecx, ecx
  xgetbv
  not eax
  test al, 0x06
  jne .no_osxsave
  mov eax, 0x07
  xor ecx, ecx
  cpuid
  mov eax, ebx
  shr eax, 14
  and eax, 0x04
  shr ebx, 31
  lea eax, [rax + 8 * rbx]
  or esi, eax
.no_osxsave:
  mov eax, esi
  pop rbx
  ret

; A simple SSE4.2 implementation particularly suited for small buffers.
crc32c_small_x86_64_sse42:
  mov eax, edi
  cmp rdx, 8
  jb .no_byte_padding
  lea rcx, qword [rdx - 8]
  mov edi, ecx
  shr edi, 3
  inc edi
  and edi, 7
  je .no_qword_padding
  shl edi, 3
  xor r8d, r8d
.qword_loop:
  crc32 rax, qword [rsi + r8]
  add r8, 8
  cmp rdi, r8
  jne .qword_loop
  sub rdx, r8
  add rsi, r8
.no_qword_padding:
  cmp rcx, 56
  jb .no_byte_padding
.8way_qword_loop:
  crc32 rax, qword [rsi]
  crc32 rax, qword [rsi + 8]
  crc32 rax, qword [rsi + 16]
  crc32 rax, qword [rsi + 24]
  crc32 rax, qword [rsi + 32]
  crc32 rax, qword [rsi + 40]
  crc32 rax, qword [rsi + 48]
  crc32 rax, qword [rsi + 56]
  sub rdx, 64
  add rsi, 64
  cmp rdx, 7
  ja .8way_qword_loop
.no_byte_padding:
  test rdx, rdx
  je .done
  xor ecx, ecx
.byte_loop:
  crc32 eax, byte [rsi + rcx]
  inc rcx
  cmp rdx, rcx
  jne .byte_loop
.done:
  retq

; Reed-Solomon encoder is basically a shift register. Here, we observe
; that compilers are generally very bad at optimising this kind of pattern.
; This is however excusable, as the x86_64 instruction set does not
; accommodate it very well. Below we use a somewhat cheap trick with storing
; the 32-byte circular buffers in XMM registers (128-bit wide each).
; Then we can use pslldq/psrldq to shift the data (like in a shift register).
; and make room for the new data by ANDing with the mask below.

; A version that works on all x86_64 machines.
rse32_wipe_mask:
  db 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
  db 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
K equ 223
T equ 32
rse32_x86_64_generic:
  mov rax, rsi
  mov rsi, rdi
  pxor xmm0, xmm0
  movdqu [rax + K + (T / 2)], xmm0
  movdqu [rax + K], xmm0
  movzx edx, byte [rax + K + T - 1]
  movzx ecx, byte [rax + K + T - 2]
  movzx r10d, byte [rax + K + T - 4]
  movzx r8d, byte [rax + K + T - 3]
  movd xmm3, dword [rax + K + T - 8]
  movq xmm2, qword [rax + K + T - 16]
  mov edi, K
  lea r9, [rel PROD_GEN]
  movdqa xmm1, [rel rse32_wipe_mask]
.rse32_K_loop:
  xor dl, byte [rsi + rdi - 1]
  movzx r11d, dl
  shl r11d, 5
  mov edx, ecx
  xor dl, byte [r11 + r9 + T - 1]
  mov byte [rax + K + T - 1], dl
  mov ecx, r8d
  xor cl, byte [r11 + r9 + T - 2]
  mov byte [rax + K + T - 2], cl
  mov r8d, r10d
  xor r8b, byte [r11 + r9 + T - 3]
  mov byte [rax + K + T - 3], r8b
  movd xmm4, dword [r11 + r9 + T - 7]
  pxor xmm4, xmm3
  movd dword [rax + K + T - 7], xmm4
  movq xmm3, qword [r11 + r9 + T - 15]
  pxor xmm3, xmm2
  movq qword [rax + K + T - 15], xmm3
  movdqu xmm5, [r11 + r9 + 1]
  pxor xmm5, xmm0
  movdqu [rax + K + 1], xmm5
  mov r10d, dword [r11 + r9]
  mov byte [rax + K], r10b
  movdqa xmm2, xmm5
  pslldq xmm2, 1
  movd xmm6, r10d
  movdqa xmm0, xmm1
  pandn xmm0, xmm6
  por xmm0, xmm2
  psrldq xmm5, 15
  movdqa xmm2, xmm3
  psllq xmm2, 8
  por xmm2, xmm5
  psrlq xmm3, 56
  movdqa xmm5, xmm1
  pandn xmm5, xmm3
  movdqa xmm3, xmm4
  pslld xmm3, 8
  por xmm3, xmm5
  movdqa [rsp - 24], xmm4
  movzx r10d, byte [rsp - 21]
  dec rdi
  jne .rse32_K_loop
  mov ecx, K
  mov rdi, rax
  rep movsb
  ret

rse32_x86_64_avx512:
  jmp rse32_x86_64_generic

; Need .GNU-stack to mark the stack as non-executable on ELF targets.
%ifdef ELF
  section .note.GNU-stack noalloc noexec nowrite progbits
%endif