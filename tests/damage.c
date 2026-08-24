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

/* Deterministic fault injector with exact-cell and CRC-preserving damage.
   Its CRC implementation is independent of libxpar_core. */

/* Use large-file POSIX offsets when available. Darwin needs the full BSD
   set rather than a strict _XOPEN_SOURCE. */
#if !defined(_WIN32) && !defined(__MSDOS__)
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE 1
#elif !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 700
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32) && !defined(__MSDOS__)
#include <sys/types.h>
#endif

#define CRC32C_POLY 0x82F63B78u   /*  0x1EDC6F41 reflected.  */

static unsigned int crc_tab[256];
static unsigned int crc_rev[256];   /*  Top byte of an entry to its index.  */
static int crc_ready;

static void crc_init(void) {
  unsigned int i, j, c;
  int seen[256];
  if (crc_ready) return;
  for (i = 0; i < 256; i++) {
    c = i;
    for (j = 0; j < 8; j++)
      c = (c >> 1) ^ (CRC32C_POLY & (unsigned int) -(int) (c & 1));
    crc_tab[i] = c;
  }
  memset(seen, 0, sizeof seen);
  for (i = 0; i < 256; i++) {
    unsigned int top = crc_tab[i] >> 24;
    if (seen[top]) {
      fprintf(stderr, "damage: CRC reverse table is not invertible\n");
      exit(3);
    }
    seen[top] = 1;
    crc_rev[top] = i;
  }
  crc_ready = 1;
}

static unsigned int crc_raw(unsigned int c, const unsigned char * p,
                            unsigned long long n) {
  unsigned long long i;
  for (i = 0; i < n; i++) c = (c >> 8) ^ crc_tab[(c ^ p[i]) & 0xFF];
  return c;
}

/* Backpatch four bytes to advance CRC state s to f. */

static void crc_backpatch(unsigned int s, unsigned int f,
                          unsigned char out[4]) {
  unsigned int st[5], c;
  unsigned int idx[4];
  int k;
  st[4] = f;
  for (k = 3; k >= 0; k--) {
    idx[k] = crc_rev[st[k + 1] >> 24];
    st[k]  = (st[k + 1] ^ crc_tab[idx[k]]) << 8;
  }
  c = s;
  for (k = 0; k < 4; k++) {
    out[k] = (unsigned char) ((c & 0xFF) ^ idx[k]);
    c = (c >> 8) ^ crc_tab[(c ^ out[k]) & 0xFF];
  }
  if (c != f) {
    fprintf(stderr, "damage: internal: backpatch landed on %08X, not %08X\n",
            c, f);
    exit(3);
  }
}

static unsigned long long rng_state = 0x9E3779B97F4A7C15ull;

static unsigned int rng_next(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (unsigned int) (rng_state >> 32);
}

/* Edit in place to support multi-gigabyte benchmark corpora. */

static const char * img_path;
static FILE * img_f;
static unsigned long long img_len;

#define IO_CHUNK 65536

/* Limit the only operation that buffers its entire range. */
#define FORGE_MAX (256ull * 1024ull * 1024ull)

static void io_seek(unsigned long long off) {
#if defined(_WIN32)
  if (_fseeki64(img_f, (long long) off, SEEK_SET)) {
#else
  if (fseeko(img_f, (off_t) off, SEEK_SET)) {
#endif
    perror(img_path);  exit(2);
  }
}

static void io_read(unsigned long long off, unsigned char * p, size_t n) {
  io_seek(off);
  if (n && fread(p, 1, n, img_f) != n) { perror(img_path);  exit(2); }
}

static void io_write(unsigned long long off, const unsigned char * p,
                     size_t n) {
  io_seek(off);
  if (n && fwrite(p, 1, n, img_f) != n) { perror(img_path);  exit(2); }
  if (fflush(img_f)) { perror(img_path);  exit(2); }
}

static void img_open(const char * path) {
  long long size;
  img_path = path;
  img_f = fopen(path, "r+b");
  if (!img_f) { perror(path);  exit(2); }
#if defined(_WIN32)
  if (_fseeki64(img_f, 0, SEEK_END)) { perror(path);  exit(2); }
  size = _ftelli64(img_f);
#else
  if (fseeko(img_f, 0, SEEK_END)) { perror(path);  exit(2); }
  size = (long long) ftello(img_f);
#endif
  if (size < 0) { perror(path);  exit(2); }
  img_len = (unsigned long long) size;
}

static void img_close(void) {
  if (img_f && fclose(img_f)) { perror(img_path);  exit(2); }
  img_f = NULL;
}

static void need_range(const char * what, unsigned long long off,
                       unsigned long long len) {
  if (off > img_len || len > img_len - off) {
    fprintf(stderr, "damage: %s: [%llu, %llu) is outside a %llu byte file\n",
            what, off, off + len, img_len);
    exit(2);
  }
}

static void op_flip(unsigned long long off, unsigned long long len) {
  unsigned char buf[IO_CHUNK];
  need_range("flip", off, len);
  while (len) {
    size_t take = len > sizeof buf ? sizeof buf : (size_t) len, i;
    io_read(off, buf, take);
    for (i = 0; i < take; i++) {
      unsigned char d = (unsigned char) rng_next();
      buf[i] ^= (unsigned char) (d ? d : 0xA5);
    }
    io_write(off, buf, take);
    off += take;  len -= take;
  }
}

static void op_rand(unsigned long long off, unsigned long long len) {
  unsigned char buf[IO_CHUNK];
  need_range("rand", off, len);
  while (len) {
    size_t take = len > sizeof buf ? sizeof buf : (size_t) len, i;
    io_read(off, buf, take);
    for (i = 0; i < take; i++) {
      unsigned char v = (unsigned char) rng_next();
      if (v == buf[i]) v ^= 0x5Au;
      buf[i] = v;
    }
    io_write(off, buf, take);
    off += take;  len -= take;
  }
}

static void op_zero(unsigned long long off, unsigned long long len) {
  static const unsigned char zero[IO_CHUNK];
  need_range("zero", off, len);
  while (len) {
    size_t take = len > sizeof zero ? sizeof zero : (size_t) len;
    io_write(off, zero, take);
    off += take;  len -= take;
  }
}

static unsigned int range_crc(unsigned long long off,
                              unsigned long long len) {
  unsigned char buf[IO_CHUNK];
  unsigned int c = 0xFFFFFFFFu;
  crc_init();
  while (len) {
    size_t take = len > sizeof buf ? sizeof buf : (size_t) len;
    io_read(off, buf, take);
    c = crc_raw(c, buf, take);
    off += take;  len -= take;
  }
  return ~c;
}

static void op_forge(unsigned long long off, unsigned long long len) {
  unsigned char * body;
  unsigned long long i, n;
  unsigned int want, state;
  need_range("forge", off, len);
  if (len < 8) {
    fprintf(stderr, "damage: forge needs at least 8 bytes, not %llu\n", len);
    exit(2);
  }
  if (len > FORGE_MAX) {
    fprintf(stderr, "damage: forge range too large: %llu bytes (max %llu)\n",
            len, FORGE_MAX);
    exit(2);
  }
  crc_init();
  want = range_crc(off, len);
  n = len - 4;
  body = (unsigned char *) malloc((size_t) len);
  if (!body) { fprintf(stderr, "damage: out of memory\n");  exit(2); }
  io_read(off, body, (size_t) len);
  for (i = 0; i < n; i++) {
    unsigned char v = (unsigned char) rng_next();
    if (v == body[i]) v ^= 0x5Au;
    body[i] = v;
  }
  state = crc_raw(0xFFFFFFFFu, body, (size_t) n);
  crc_backpatch(state, ~want, body + n);
  io_write(off, body, (size_t) len);
  free(body);
  if (range_crc(off, len) != want) {
    fprintf(stderr, "damage: internal: forge did not preserve the CRC\n");
    exit(3);
  }
}

static void op_crc(unsigned long long off, unsigned long long len) {
  need_range("crc", off, len);
  printf("%08X\n", range_crc(off, len));
}

/* Portable truncate via copy and rename. */
static void op_truncate(unsigned long long len) {
  unsigned char buf[IO_CHUNK];
  unsigned long long off = 0, left;
  char * tmp;
  FILE * out;
  if (len > img_len) {
    fprintf(stderr, "damage: truncate: %llu is past the end\n", len);
    exit(2);
  }
  tmp = (char *) malloc(strlen(img_path) + 8);
  if (!tmp) { fprintf(stderr, "damage: out of memory\n");  exit(2); }
  sprintf(tmp, "%s.trunc", img_path);
  out = fopen(tmp, "wb");
  if (!out) { perror(tmp);  exit(2); }
  for (left = len; left; ) {
    size_t take = left > sizeof buf ? sizeof buf : (size_t) left;
    io_read(off, buf, take);
    if (fwrite(buf, 1, take, out) != take) { perror(tmp);  exit(2); }
    off += take;  left -= take;
  }
  if (fclose(out)) { perror(tmp);  exit(2); }
  img_close();
  if (rename(tmp, img_path)) { perror(img_path);  exit(2); }
  free(tmp);
  img_open(img_path);
}

static void op_extend(unsigned long long len) {
  unsigned char buf[IO_CHUNK];
  unsigned long long at = img_len;
  while (len) {
    size_t take = len > sizeof buf ? sizeof buf : (size_t) len, i;
    for (i = 0; i < take; i++) buf[i] = (unsigned char) rng_next();
    io_write(at, buf, take);
    at += take;  len -= take;
  }
  img_len = at;
}


static int split2(const char * s, unsigned long long * a,
                  unsigned long long * b) {
  char * end;
  *a = strtoull(s, &end, 0);
  if (*end != ',') return 0;
  *b = strtoull(end + 1, &end, 0);
  return *end == 0;
}

static void usage(void) {
  fprintf(stderr,
    "usage: damage <file> <op>...\n"
    "  seed=N             reseed the generator for the operations after it\n"
    "  flip=OFF,LEN       exclusive-or pseudorandom bytes over a range\n"
    "  rand=OFF,LEN       replace a range with bytes that differ\n"
    "  zero=OFF,LEN       replace a range with zeroes\n"
    "  forge=OFF,LEN      rewrite a range, keeping its CRC-32C\n"
    "  crc=OFF,LEN        print a range's CRC-32C; changes nothing\n"
    "  truncate=LEN       cut the file down to LEN bytes\n"
    "  extend=LEN         append LEN pseudorandom bytes\n"
    "  cell=S,J           with -Z and -Y, damage one cell of one slice\n"
    "  -Z SIZE            slice size, for cell=\n"
    "  -Y SIZE            cell size, for cell=\n"
    "  -n LEN             bytes a cell= operation damages (default 64)\n"
    "  -k HOW             what cell= does: rand (default), zero, forge\n");
  exit(2);
}

int main(int argc, char ** argv) {
  const char * path;
  const char * cell_kind = "rand";
  unsigned long long z = 0, y = 0, cell_len = 64, a, b;
  int i;

  if (argc < 3) usage();
  path = argv[1];
  img_open(path);

  for (i = 2; i < argc; i++) {
    const char * s = argv[i];
    if (!strcmp(s, "-Z") || !strcmp(s, "-Y") || !strcmp(s, "-n") ||
        !strcmp(s, "-k")) {
      if (i + 1 >= argc) usage();
      if      (s[1] == 'Z') z = strtoull(argv[++i], NULL, 0);
      else if (s[1] == 'Y') y = strtoull(argv[++i], NULL, 0);
      else if (s[1] == 'n') cell_len = strtoull(argv[++i], NULL, 0);
      else                  cell_kind = argv[++i];
      continue;
    }
    if (!strncmp(s, "seed=", 5)) {
      unsigned long long v = strtoull(s + 5, NULL, 0);
      rng_state = v ? v : 0x9E3779B97F4A7C15ull;
      continue;
    }
    if (!strncmp(s, "truncate=", 9)) { op_truncate(strtoull(s + 9, NULL, 0));
                                       continue; }
    if (!strncmp(s, "extend=", 7))   { op_extend(strtoull(s + 7, NULL, 0));
                                       continue; }
    if (!strncmp(s, "cell=", 5)) {
      unsigned long long off;
      if (!split2(s + 5, &a, &b) || !z || !y) usage();
      off = a * z + b * y;
      if      (!strcmp(cell_kind, "zero"))  op_zero(off, cell_len);
      else if (!strcmp(cell_kind, "forge")) op_forge(off, y);
      else                                  op_rand(off, cell_len);
      continue;
    }
    if (!split2(strchr(s, '=') ? strchr(s, '=') + 1 : "", &a, &b)) usage();
    if      (!strncmp(s, "flip=",  5)) op_flip(a, b);
    else if (!strncmp(s, "rand=",  5)) op_rand(a, b);
    else if (!strncmp(s, "zero=",  5)) op_zero(a, b);
    else if (!strncmp(s, "forge=", 6)) op_forge(a, b);
    else if (!strncmp(s, "crc=",   4)) op_crc(a, b);
    else usage();
  }

  img_close();
  return 0;
}
