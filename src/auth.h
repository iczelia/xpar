/*  xpar: authenticated key-file loading interface.  */
#ifndef XPAR_AUTH_H
#define XPAR_AUTH_H

#include "container.h"

typedef enum {
  XPAR_KEYFILE_OK = 0,
  XPAR_KEYFILE_OPEN,
  XPAR_KEYFILE_EMPTY,
  XPAR_KEYFILE_READ
} xpar_keyfile_status;

xpar_keyfile_status xpar_keyfile_load(const char * path, xpar_key * key,
                                      u8 master[XPAR_BLAKE3_KEY_LEN]);

void xpar_key_forget(xpar_key * key,
                     u8 master[XPAR_BLAKE3_KEY_LEN]);

#endif
