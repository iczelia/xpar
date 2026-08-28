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

/*  Append a checksum-valid packet to exercise hostile-body handling.  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*  The host file owns main() and calls xpar_main.  */

#include "common.h"
#include "container.h"
#include "xpar2.h"

static void usage(void) {
  fprintf(stderr,
          "usage: forge <volume> <TYPE> <hex-body> [flags]\n"
          "       forge <volume> patch <TYPE> <body-offset> <hex>\n");
  exit(2);
}

/*  The tag covers packet bytes [0, 40) and [48, length).  */
static void resign(u8 * p, u64 len) {
  xpar_blake3_t h;
  u32 flags = xpar_rd32(p + 36);
  xpar_blake3_init(&h);
  xpar_blake3_update(&h, p, 40);
  if (!(flags & XPAR_PF_BODY_UNCHECKED) && len > XPAR_PKT_HDR)
    xpar_blake3_update(&h, p + XPAR_PKT_HDR, (sz) (len - XPAR_PKT_HDR));
  xpar_blake3_final(&h, p + 40, 8);
}

static int unhex(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/*  Rewrite one packet's body bytes in place and re-sign it.  */
static int patch_mode(const char * path, const char * type, u64 at,
                      const char * hex);

int xpar_main(int argc, char ** argv) {
  FILE * f;
  long size;
  u8 * data, * body;
  sz body_len, i;
  xpar_pkt hdr;
  xpar_buf out;

  if (argc == 6 && !strcmp(argv[2], "patch")) {
    if (strlen(argv[3]) != 4) usage();
    return patch_mode(argv[1], argv[3], strtoull(argv[4], NULL, 0), argv[5]);
  }
  if (argc != 4 && argc != 5) usage();
  if (strlen(argv[2]) != 4) usage();
  if (strlen(argv[3]) % 2) usage();
  body_len = strlen(argv[3]) / 2;

  f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]);  return 2; }
  if (fseek(f, 0, SEEK_END) != 0) { perror("seek");  return 2; }
  size = ftell(f);
  if (size <= 0) { fprintf(stderr, "forge: %s is empty\n", argv[1]); return 2; }
  rewind(f);
  data = (u8 *) malloc((size_t) size);
  if (!data || fread(data, 1, (size_t) size, f) != (size_t) size) {
    fprintf(stderr, "forge: cannot read %s\n", argv[1]);  return 2;
  }
  fclose(f);

  if (xpar_pkt_read(data, (u64) size, NULL, &hdr) != XPAR_OK) {
    fprintf(stderr, "forge: %s does not begin with a packet\n", argv[1]);
    return 2;
  }

  body = (u8 *) malloc(body_len ? body_len : 1);
  if (!body) return 2;
  for (i = 0; i < body_len; i++) {
    int hi = unhex(argv[3][2 * i]), lo = unhex(argv[3][2 * i + 1]);
    if (hi < 0 || lo < 0) usage();
    body[i] = (u8) (hi * 16 + lo);
  }

  xpar_buf_init(&out);
  xpar_pkt_write(&out, argv[2], argc == 5 ? (u32) strtoul(argv[4], NULL, 0)
                                          : 0u,
                 hdr.set_id, body, body_len, NULL);

  f = fopen(argv[1], "ab");
  if (!f) { perror(argv[1]);  return 2; }
  if (fwrite(out.data, 1, out.len, f) != out.len) {
    fprintf(stderr, "forge: short write on %s\n", argv[1]);  return 2;
  }
  if (fclose(f) != 0) { perror("close");  return 2; }
  xpar_buf_free(&out);
  free(body);  free(data);
  return 0;
}

static int patch_mode(const char * path, const char * type, u64 at,
                      const char * hex) {
  FILE * f;
  long size;
  u8 * data, * body;
  sz body_len, i;
  u64 pos = 0;
  int hits = 0;

  if (strlen(hex) % 2) usage();
  body_len = strlen(hex) / 2;
  body = (u8 *) malloc(body_len ? body_len : 1);
  if (!body) return 2;
  for (i = 0; i < body_len; i++) {
    int hi = unhex(hex[2 * i]), lo = unhex(hex[2 * i + 1]);
    if (hi < 0 || lo < 0) usage();
    body[i] = (u8) (hi * 16 + lo);
  }

  f = fopen(path, "rb");
  if (!f) { perror(path);  return 2; }
  if (fseek(f, 0, SEEK_END) != 0) { perror("seek");  return 2; }
  size = ftell(f);
  if (size <= 0) { fprintf(stderr, "forge: %s is empty\n", path);  return 2; }
  rewind(f);
  data = (u8 *) malloc((size_t) size);
  if (!data || fread(data, 1, (size_t) size, f) != (size_t) size) {
    fprintf(stderr, "forge: cannot read %s\n", path);  return 2;
  }
  fclose(f);

  while (pos + XPAR_PKT_HDR <= (u64) size) {
    u64 len;
    if (memcmp(data + pos, XPAR_PKT_MAGIC, 8)) { pos += XPAR_PKT_ALIGN;
                                                 continue; }
    len = xpar_rd64(data + pos + 8);
    if (len < XPAR_PKT_HDR || len % XPAR_PKT_ALIGN ||
        pos + len > (u64) size) { pos += XPAR_PKT_ALIGN;  continue; }
    if (!memcmp(data + pos + 32, type, 4) &&
        XPAR_PKT_HDR + at + body_len <= len) {
      memcpy(data + pos + XPAR_PKT_HDR + at, body, body_len);
      resign(data + pos, len);
      hits++;
    }
    pos += len;
  }
  if (!hits) {
    fprintf(stderr, "forge: no %s packet in %s\n", type, path);  return 2;
  }

  f = fopen(path, "wb");
  if (!f) { perror(path);  return 2; }
  if (fwrite(data, 1, (size_t) size, f) != (size_t) size) {
    fprintf(stderr, "forge: short write on %s\n", path);  return 2;
  }
  if (fclose(f) != 0) { perror("close");  return 2; }
  free(body);  free(data);
  return 0;
}
