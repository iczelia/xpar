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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static unsigned int crc32c(const unsigned char * p, unsigned long long n) {
  crc_init();
  return ~crc_raw(0xFFFFFFFFu, p, n);
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

static unsigned char * img;
static unsigned long long img_len, img_cap;
static int img_dirty;

#define IMG_MAX (1024ull * 1024ull * 1024ull)

static void img_load(const char * path) {
  FILE * f = fopen(path, "rb");
  long long size;
  if (!f) { perror(path);  exit(2); }
  if (fseek(f, 0, SEEK_END)) { perror(path);  exit(2); }
  size = ftell(f);
  if (size < 0) { perror(path);  exit(2); }
  if ((unsigned long long) size > IMG_MAX) {
    fprintf(stderr, "damage: %s: larger than the %llu byte ceiling\n",
            path, IMG_MAX);
    exit(2);
  }
  rewind(f);
  img_len = img_cap = (unsigned long long) size;
  img = (unsigned char *) malloc(img_cap ? (size_t) img_cap : 1);
  if (!img) { fprintf(stderr, "damage: out of memory\n");  exit(2); }
  if (img_len && fread(img, 1, (size_t) img_len, f) != (size_t) img_len) {
    perror(path);  exit(2);
  }
  fclose(f);
}

static void img_store(const char * path) {
  FILE * f;
  if (!img_dirty) return;
  f = fopen(path, "wb");
  if (!f) { perror(path);  exit(2); }
  if (img_len && fwrite(img, 1, (size_t) img_len, f) != (size_t) img_len) {
    perror(path);  exit(2);
  }
  if (fclose(f)) { perror(path);  exit(2); }
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
  unsigned long long i;
  need_range("flip", off, len);
  for (i = 0; i < len; i++) {
    unsigned char d = (unsigned char) rng_next();
    img[off + i] ^= (unsigned char) (d ? d : 0xA5);
  }
  img_dirty = 1;
}

static void op_rand(unsigned long long off, unsigned long long len) {
  unsigned long long i;
  need_range("rand", off, len);
  for (i = 0; i < len; i++) {
    unsigned char v = (unsigned char) rng_next();
    if (v == img[off + i]) v ^= 0x5Au;
    img[off + i] = v;
  }
  img_dirty = 1;
}

static void op_zero(unsigned long long off, unsigned long long len) {
  need_range("zero", off, len);
  memset(img + off, 0, (size_t) len);
  img_dirty = 1;
}

static void op_forge(unsigned long long off, unsigned long long len) {
  unsigned long long body, i;
  unsigned int want, state;
  int changed = 0;
  need_range("forge", off, len);
  if (len < 8) {
    fprintf(stderr, "damage: forge needs at least 8 bytes, not %llu\n", len);
    exit(2);
  }
  crc_init();
  want  = crc32c(img + off, len);
  body  = len - 4;
  for (i = 0; i < body; i++) {
    unsigned char v = (unsigned char) rng_next();
    if (v == img[off + i]) v ^= 0x5Au;
    img[off + i] = v;
    changed = 1;
  }
  state = crc_raw(0xFFFFFFFFu, img + off, body);
  crc_backpatch(state, ~want, img + off + body);
  if (crc32c(img + off, len) != want) {
    fprintf(stderr, "damage: internal: forge did not preserve the CRC\n");
    exit(3);
  }
  if (!changed) {
    fprintf(stderr, "damage: internal: forge changed nothing\n");
    exit(3);
  }
  img_dirty = 1;
}

static void op_crc(unsigned long long off, unsigned long long len) {
  need_range("crc", off, len);
  printf("%08X\n", crc32c(img + off, len));
}

static void op_truncate(unsigned long long len) {
  if (len > img_len) {
    fprintf(stderr, "damage: truncate: %llu is past the end\n", len);
    exit(2);
  }
  img_len = len;
  img_dirty = 1;
}

static void op_extend(unsigned long long len) {
  unsigned long long i;
  unsigned char * p = (unsigned char *) realloc(img, (size_t) (img_len + len));
  if (!p) { fprintf(stderr, "damage: out of memory\n");  exit(2); }
  img = p;
  for (i = 0; i < len; i++) img[img_len + i] = (unsigned char) rng_next();
  img_len += len;
  img_cap  = img_len;
  img_dirty = 1;
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
  img_load(path);

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

  img_store(path);
  free(img);
  return 0;
}
