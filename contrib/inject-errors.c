/*  Deterministic error injection for self-checks; see usage() below.  */

#include "common.h"

#define K_BLOCK 223

static u64 rng_state = 1;

static void rng_seed(u64 s) { rng_state = s ? s : 1; }

static u64 rng_next(void) {
  u64 x = rng_state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  rng_state = x;
  return x;
}

static u64 rng_range(u64 n) { return rng_next() % n; }

static void die(const char * msg) {
  xpar_fprintf(xpar_stderr, "inject-errors: %s\n", msg);
  xpar_exit(2);
}

static xpar_file * xopen(const char * path, int flags) {
  xpar_file * f = xpar_open(path, flags);
  if (!f) {
    xpar_fprintf(xpar_stderr, "inject-errors: %s: %s\n",
                 path, xpar_strerror(xpar_errno()));
    xpar_exit(2);
  }
  return f;
}

static i64 parse_i64(const char * s, const char * what) {
  i64 v;
  if (xpar_parse_i64(s, &v) != 0) {
    xpar_fprintf(xpar_stderr, "inject-errors: bad %s: %s\n", what, s);
    xpar_exit(2);
  }
  return v;
}

static u64 parse_u64(const char * s, const char * what) {
  u64 v;
  if (xpar_parse_u64(s, &v) != 0) {
    xpar_fprintf(xpar_stderr, "inject-errors: bad %s: %s\n", what, s);
    xpar_exit(2);
  }
  return v;
}

static void xread_or_die(xpar_file * f, void * buf, sz n) {
  if (xpar_read(f, buf, n) != n) die("short read");
}
static void xwrite_or_die(xpar_file * f, const void * buf, sz n) {
  if (xpar_write(f, buf, n) != n) die("write failed");
}
static void xseek_or_die(xpar_file * f, i64 off) {
  if (xpar_seek(f, off, XPAR_SEEK_SET) != 0) die("seek failed");
}

static void do_scatter(const char * path, i64 count, u64 seed) {
  xpar_file * f = xopen(path, XPAR_O_READ | XPAR_O_WRITE);
  i64 size = xpar_size(f);
  if (size < 0) die("size failed");
  i64 nblocks = size / K_BLOCK;
  if (count < 0 || count > nblocks) {
    xpar_fprintf(xpar_stderr,
                 "inject-errors: file has %lld blocks, need %lld\n",
                 (long long) nblocks, (long long) count);
    xpar_exit(2);
  }
  i64 * idx = xpar_alloc_raw((sz) nblocks * sizeof(i64));
  for (i64 i = 0; i < nblocks; i++) idx[i] = i;
  rng_seed(seed);
  for (i64 i = 0; i < count; i++) {
    i64 j = i + (i64) rng_range((u64) (nblocks - i));
    i64 t = idx[i]; idx[i] = idx[j]; idx[j] = t;
  }
  for (i64 i = 0; i < count; i++) {
    i64 off = idx[i] * K_BLOCK + (i64) rng_range(K_BLOCK);
    xseek_or_die(f, off);
    u8 b;
    if (xpar_read(f, &b, 1) != 1) die("short read");
    b ^= 0xA5;
    xseek_or_die(f, off);
    xwrite_or_die(f, &b, 1);
  }
  xpar_free(idx);
  xpar_xclose(f);
}

static void do_burst(const char * path, i64 count, i64 start) {
  if (count < 0 || start < 0) die("negative count or start");
  xpar_file * f = xopen(path, XPAR_O_READ | XPAR_O_WRITE);
  u8 * buf = count ? xpar_alloc_raw((sz) count) : NULL;
  xseek_or_die(f, start);
  xread_or_die(f, buf, (sz) count);
  for (i64 i = 0; i < count; i++) buf[i] ^= 0xFF;
  xseek_or_die(f, start);
  xwrite_or_die(f, buf, (sz) count);
  xpar_free(buf);
  xpar_xclose(f);
}

static void do_truncate(const char * path, i64 delta) {
  /*  Portable truncation: read N bytes, re-create, rewrite.  */
  xpar_file * r = xopen(path, XPAR_O_READ);
  i64 size = xpar_size(r);
  if (size < 0) die("size failed");
  i64 newsize = size - delta;
  if (newsize < 0) die("truncation below zero");
  u8 * buf = newsize ? xpar_alloc_raw((sz) newsize) : NULL;
  if (newsize && xpar_read(r, buf, (sz) newsize) != (sz) newsize) die("short read");
  xpar_xclose(r);
  xpar_file * w = xopen(path, XPAR_O_WRITE | XPAR_O_CREATE | XPAR_O_TRUNCATE);
  if (newsize) xwrite_or_die(w, buf, (sz) newsize);
  xpar_xclose(w);
  xpar_free(buf);
}

static void do_swap(const char * path, i64 off_a, i64 off_b, i64 len) {
  if (len < 0 || off_a < 0 || off_b < 0) die("negative offset or length");
  xpar_file * f = xopen(path, XPAR_O_READ | XPAR_O_WRITE);
  u8 * a = len ? xpar_alloc_raw((sz) len) : NULL;
  u8 * b = len ? xpar_alloc_raw((sz) len) : NULL;
  xseek_or_die(f, off_a); xread_or_die(f, a, (sz) len);
  xseek_or_die(f, off_b); xread_or_die(f, b, (sz) len);
  xseek_or_die(f, off_a); xwrite_or_die(f, b, (sz) len);
  xseek_or_die(f, off_b); xwrite_or_die(f, a, (sz) len);
  xpar_free(a); xpar_free(b);
  xpar_xclose(f);
}

static void do_swap2(const char * pa, const char * pb, i64 off, i64 len) {
  if (len < 0 || off < 0) die("negative offset or length");
  xpar_file * fa = xopen(pa, XPAR_O_READ | XPAR_O_WRITE);
  xpar_file * fb = xopen(pb, XPAR_O_READ | XPAR_O_WRITE);
  u8 * a = len ? xpar_alloc_raw((sz) len) : NULL;
  u8 * b = len ? xpar_alloc_raw((sz) len) : NULL;
  xseek_or_die(fa, off); xread_or_die(fa, a, (sz) len);
  xseek_or_die(fb, off); xread_or_die(fb, b, (sz) len);
  xseek_or_die(fa, off); xwrite_or_die(fa, b, (sz) len);
  xseek_or_die(fb, off); xwrite_or_die(fb, a, (sz) len);
  xpar_free(a); xpar_free(b);
  xpar_xclose(fa); xpar_xclose(fb);
}

static void usage(void) {
  xpar_fputs(
    "usage:\n"
    "  inject-errors scatter   FILE COUNT SEED\n"
    "  inject-errors burst     FILE COUNT START\n"
    "  inject-errors truncate  FILE DELTA\n"
    "  inject-errors swap      FILE OFFSET_A OFFSET_B LENGTH\n"
    "  inject-errors swap2     FILE_A FILE_B OFFSET LENGTH\n",
    xpar_stderr);
  xpar_exit(2);
}

int xpar_main(int argc, char ** argv) {
  if (argc < 3) usage();
  const char * mode = argv[1];
  if (xpar_strcmp(mode, "scatter") == 0 && argc == 5) {
    do_scatter(argv[2], parse_i64(argv[3], "count"),
               parse_u64(argv[4], "seed"));
  } else if (xpar_strcmp(mode, "burst") == 0 && argc == 5) {
    do_burst(argv[2], parse_i64(argv[3], "count"),
             parse_i64(argv[4], "start"));
  } else if (xpar_strcmp(mode, "truncate") == 0 && argc == 4) {
    do_truncate(argv[2], parse_i64(argv[3], "delta"));
  } else if (xpar_strcmp(mode, "swap") == 0 && argc == 6) {
    do_swap(argv[2], parse_i64(argv[3], "offset_a"),
            parse_i64(argv[4], "offset_b"),
            parse_i64(argv[5], "length"));
  } else if (xpar_strcmp(mode, "swap2") == 0 && argc == 6) {
    do_swap2(argv[2], argv[3], parse_i64(argv[4], "offset"),
             parse_i64(argv[5], "length"));
  } else {
    usage();
  }
  return 0;
}
