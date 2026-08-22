/*  xpar: authenticated key-file loading and key lifetime helpers.

    Copyright (C) 2022-2026 Kamila Szewczyk.  GPLv3-only (see COPYING).  */

#include "auth.h"

#include "blake3.h"
#include "port.h"

xpar_keyfile_status xpar_keyfile_load(const char * path, xpar_key * key,
                                      u8 master[XPAR_BLAKE3_KEY_LEN]) {
  xpar_file * f = xpar_open(path, XPAR_O_RDONLY);
  xpar_blake3_t h;
  u8 buf[16384];
  u64 total = 0;
  if (!f) return XPAR_KEYFILE_OPEN;
  xpar_blake3_init_derive_key(&h, "xpar2 auth master v1");
  for (;;) {
    sz n = xpar_read(f, buf, sizeof buf);
    if (n) {
      xpar_blake3_update(&h, buf, n);
      total += n;
    }
    if (n < sizeof buf) {
      if (xpar_error(f)) {
        xpar_close(f);
        xpar_secure_zero(buf, sizeof buf);
        xpar_secure_zero(&h, sizeof h);
        return XPAR_KEYFILE_READ;
      }
      if (xpar_eof(f) || !n) break;
    }
  }
  xpar_close(f);
  xpar_secure_zero(buf, sizeof buf);
  if (!total) {
    xpar_secure_zero(&h, sizeof h);
    return XPAR_KEYFILE_EMPTY;
  }
  xpar_blake3_final(&h, master, XPAR_BLAKE3_KEY_LEN);
  xpar_secure_zero(&h, sizeof h);
  xpar_key_derive(key, master);
  return XPAR_KEYFILE_OK;
}

void xpar_key_forget(xpar_key * key, u8 master[XPAR_BLAKE3_KEY_LEN]) {
  if (key) xpar_secure_zero(key, sizeof *key);
  if (master) xpar_secure_zero(master, XPAR_BLAKE3_KEY_LEN);
}
