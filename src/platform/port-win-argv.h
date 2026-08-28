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

/*  Shared Microsoft backslash-and-quote parser. Include with XPAR_ARGV_CH,
    XPAR_ARGV_L, and XPAR_ARGV_FN defined; one pass counts, another fills.  */

#if !defined(XPAR_ARGV_CH) || !defined(XPAR_ARGV_L) || !defined(XPAR_ARGV_FN)
  #error "port-win-argv.h needs XPAR_ARGV_CH, XPAR_ARGV_L and XPAR_ARGV_FN"
#endif

#define XPAR_ARGV_PUT(c)                                                     \
  do {                                                                       \
    if (pass) {                                                              \
      if (blen + 1 >= bcap) {                                                \
        XPAR_ARGV_CH * nb;                                                   \
        size_t ncap = bcap * 2;                                              \
        nb = (XPAR_ARGV_CH *) HeapAlloc(GetProcessHeap(), 0,                 \
                                        ncap * sizeof(XPAR_ARGV_CH));        \
        if (!nb) goto fail;                                                  \
        memcpy(nb, buf, blen * sizeof(XPAR_ARGV_CH));                        \
        HeapFree(GetProcessHeap(), 0, buf);                                  \
        buf = nb;  bcap = ncap;                                              \
      }                                                                      \
      buf[blen++] = (c);                                                     \
    }                                                                        \
  } while (0)

static int XPAR_ARGV_FN(const XPAR_ARGV_CH * cmd, XPAR_ARGV_CH *** out) {
  int argc = 0, pass;
  XPAR_ARGV_CH ** argv = NULL, * buf = NULL;
  for (pass = 0; pass < 2; pass++) {
    const XPAR_ARGV_CH * p = cmd;
    if (pass) {
      argv = (XPAR_ARGV_CH **) HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY,
                                         (size_t) (argc + 1) * sizeof(*argv));
      if (!argv) return -1;
    }
    argc = 0;
    while (*p) {
      size_t blen = 0, bcap = 0;
      int in_quote = 0;
      while (*p == XPAR_ARGV_L(' ') || *p == XPAR_ARGV_L('\t')) p++;
      if (!*p) break;
      buf = NULL;
      if (pass) {
        bcap = 64;
        buf = (XPAR_ARGV_CH *) HeapAlloc(GetProcessHeap(), 0,
                                         bcap * sizeof(XPAR_ARGV_CH));
        if (!buf) goto fail;
      }
      while (*p) {
        if (!in_quote &&
            (*p == XPAR_ARGV_L(' ') || *p == XPAR_ARGV_L('\t'))) break;
        if (*p == XPAR_ARGV_L('\\')) {
          int nbs = 0, i;
          while (*p == XPAR_ARGV_L('\\')) { nbs++;  p++; }
          if (*p == XPAR_ARGV_L('"')) {
            for (i = 0; i < nbs / 2; i++) XPAR_ARGV_PUT(XPAR_ARGV_L('\\'));
            if (nbs & 1) { XPAR_ARGV_PUT(XPAR_ARGV_L('"'));  p++; }
            else { in_quote = !in_quote;  p++; }
          } else {
            for (i = 0; i < nbs; i++) XPAR_ARGV_PUT(XPAR_ARGV_L('\\'));
          }
        } else if (*p == XPAR_ARGV_L('"')) {
          if (in_quote && p[1] == XPAR_ARGV_L('"')) {
            XPAR_ARGV_PUT(XPAR_ARGV_L('"'));
            p += 2;
          } else { in_quote = !in_quote;  p++; }
        } else {
          XPAR_ARGV_PUT(*p);
          p++;
        }
      }
      if (pass) { buf[blen] = XPAR_ARGV_L('\0');  argv[argc] = buf;
                  buf = NULL; }
      argc++;
    }
    if (pass) { *out = argv;  return argc; }
  }
  return -1;
fail:
  if (buf) HeapFree(GetProcessHeap(), 0, buf);
  if (argv) {
    int j;
    for (j = 0; j < argc; j++)
      if (argv[j]) HeapFree(GetProcessHeap(), 0, argv[j]);
    HeapFree(GetProcessHeap(), 0, argv);
  }
  return -1;
}

#undef XPAR_ARGV_PUT
