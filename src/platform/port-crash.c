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

/*  Allocation-free crash report formatting shared by host handlers.  */

#include "common.h"

/*  Bypass the formatted I/O layer.  */
static void cw(const char * s) {
  sz n = 0;
  while (s[n]) n++;
  if (n) xpar_write(xpar_stderr, s, n);
}

static void cw_hex(u64 v, int digits) {
  static const char d[] = "0123456789abcdef";
  char buf[17];
  int i;
  if (digits < 1) digits = 1;
  if (digits > 16) digits = 16;
  buf[digits] = 0;
  for (i = digits - 1; i >= 0; i--) { buf[i] = d[v & 15];  v >>= 4; }
  cw(buf);
}

static void cw_ptr(const void * p) {
  cw("0x");
  cw_hex((u64) (uintptr_t) p, (int) (sizeof(void *) * 2));
}

static void cw_dec(u64 v) {
  char buf[21];
  int i = 20;
  buf[i] = 0;
  if (!v) buf[--i] = '0';
  while (v) { buf[--i] = (char) ('0' + (v % 10));  v /= 10; }
  cw(buf + i);
}

/*  Keep sanitizer-provided fault handlers.  */
int xpar_crash_wanted(void) {
#if defined(XPAR_SANITIZED)
  return 0;
#else
  return 1;
#endif
}

/*  A second fault inside the reporter must not loop.  */
static volatile int g_in_report;

int xpar_crash_entered(void) { return g_in_report++ != 0; }

void xpar_crash_head(const char * what, u64 code, int have_code,
                     const void * pc, const void * addr, int have_addr,
                     const void * module_base) {
  cw("\nxpar: fatal: ");
  cw(what);
  if (have_code) { cw(" (code 0x");  cw_hex(code, 8);  cw(")"); }
  cw("\n");

  if (have_addr) { cw("xpar:   faulting address ");  cw_ptr(addr);  cw("\n"); }
  if (pc) {
    cw("xpar:   program counter  ");  cw_ptr(pc);
    if (module_base && (const char *) pc >= (const char *) module_base) {
      cw("  (+0x");
      cw_hex((u64) (uintptr_t) ((const char *) pc -
                                (const char *) module_base), 8);
      cw(")");
    }
    cw("\n");
  }

  cw("xpar:   build " PACKAGE_VERSION " for " XPAR_HOST_TRIPLE "\n");
  if (module_base) { cw("xpar:   image base       ");  cw_ptr(module_base);  cw("\n"); }
  cw("xpar:   call chain, innermost first:\n");
}

/*  Print a frame and its image-relative offset.  */
void xpar_crash_frame(unsigned i, const void * pc, const void * module_base) {
  cw("xpar:     #");
  cw_dec(i);
  cw(i < 10 ? "  " : " ");
  cw_ptr(pc);
  if (module_base && (const char *) pc >= (const char *) module_base) {
    cw("  +0x");
    cw_hex((u64) (uintptr_t) ((const char *) pc -
                              (const char *) module_base), 8);
  }
  cw("\n");
}

void xpar_crash_tail(int had_frames) {
  if (!had_frames)
    cw("xpar:     (call chain unavailable)\n");
  cw("xpar:   symbolize with: addr2line -e <binary> -fpi <address-or-offset>\n");
  cw("xpar: report this bug with the output above:\n"
     "xpar: https://github.com/iczelia/xpar\n");
}

/*  Walk saved frame pointers while links remain aligned and monotonic.  */
unsigned xpar_crash_walk_fp(void * const * fp, void ** out, unsigned max) {
  unsigned n = 0;
  const void * const * prev = NULL;
  while (fp && n < max) {
    void * const * next;
    void * ret;
    if (((uintptr_t) fp & (sizeof(void *) - 1)) != 0) break;
    if (prev && (const void * const *) fp <= prev) break;
    next = (void * const *) fp[0];
    ret  = fp[1];
    if (!ret) break;
    out[n++] = ret;
    prev = (const void * const *) fp;
    fp = next;
  }
  return n;
}
