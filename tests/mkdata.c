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

/* Standalone deterministic corpus generator for tests and benchmarks. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rng_state = 0x9E3779B97F4A7C15ull;

static unsigned int rng_next(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (unsigned int) (rng_state >> 32);
}

static const char * const words[] = {
  "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
  "slice", "cell", "column", "recovery", "erasure", "manifest", "packet",
  "generation", "armour", "codec", "matrix", "additive", "transform"
};

/* Random data resists compression; text exercises deduplication and chunking. */

static void emit_random(FILE * f, unsigned long long n) {
  unsigned char buf[65536];
  while (n) {
    size_t take = n > sizeof buf ? sizeof buf : (size_t) n, i;
    Fi(take, buf[i] = (unsigned char) rng_next());
    if (fwrite(buf, 1, take, f) != take) exit(2);
    n -= take;
  }
}

static void emit_text(FILE * f, unsigned long long n) {
  char buf[65536];
  size_t fill = 0;
  unsigned column = 0;
  while (n) {
    const char * w = words[rng_next() % (sizeof words / sizeof words[0])];
    size_t len = strlen(w);
    if (fill + len + 2 > sizeof buf) {
      size_t take = fill > n ? (size_t) n : fill;
      if (fwrite(buf, 1, take, f) != take) exit(2);
      n -= take;
      fill = 0;
      continue;
    }
    memcpy(buf + fill, w, len);
    fill += len;
    if (++column >= 12) { buf[fill++] = '\n';  column = 0; }
    else                  buf[fill++] = ' ';
  }
}

static void emit_zero(FILE * f, unsigned long long n) {
  static const unsigned char zero[65536];
  while (n) {
    size_t take = n > sizeof zero ? sizeof zero : (size_t) n;
    if (fwrite(zero, 1, take, f) != take) exit(2);
    n -= take;
  }
}

static void usage(void) {
  fprintf(stderr,
          "usage: mkdata <seed> <bytes> [<file>] [--pattern=WHICH]\n"
          "       patterns: random (default), text, zero\n");
  exit(2);
}

int main(int argc, char ** argv) {
  unsigned long long seed, bytes;
  const char * path = NULL;
  const char * pattern = "random";
  FILE * f;
  int i;

  if (argc < 3) usage();
  seed  = strtoull(argv[1], NULL, 0);
  bytes = strtoull(argv[2], NULL, 0);
  for (i = 3; i < argc; i++) {
    if (!strncmp(argv[i], "--pattern=", 10)) pattern = argv[i] + 10;
    else if (argv[i][0] == '-')              usage();
    else                                     path = argv[i];
  }

  rng_state = seed ? seed : 0x9E3779B97F4A7C15ull;

  if (path) {
    f = fopen(path, "wb");
    if (!f) { perror(path);  return 2; }
  } else {
    f = stdout;
  }

  if      (!strcmp(pattern, "random")) emit_random(f, bytes);
  else if (!strcmp(pattern, "text"))   emit_text(f, bytes);
  else if (!strcmp(pattern, "zero"))   emit_zero(f, bytes);
  else usage();

  if (fflush(f)) { perror("write");  return 2; }
  if (path && fclose(f)) { perror(path);  return 2; }
  return 0;
}
