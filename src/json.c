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

#include "json.h"

static void put(xpar_json * j, const char * s) {
  if (j->enabled) xpar_fputs(s, j->out);
}

static void sep(xpar_json * j) {
  if (!j->enabled) return;
  if (j->need_comma) put(j, ",");
  j->need_comma = true;
}

static void key(xpar_json * j, const char * k) {
  sep(j);
  if (j->enabled) xpar_fprintf(j->out, "\"%s\":", k);
}

void xpar_json_init(xpar_json * j, xpar_file * out, bool enabled) {
  j->out        = out;
  j->enabled    = enabled;
  j->start_usec = xpar_usec_now();
  j->in_object  = false;
  j->need_comma = false;
}

void xpar_json_begin(xpar_json * j, const char * type) {
  if (!j->enabled) return;
  xpar_assert(!j->in_object);
  j->in_object  = true;
  j->need_comma = false;
  put(j, "{");
  xpar_json_str(j, "type", type);
  xpar_json_u64(j, "t", xpar_usec_now() - j->start_usec);
  xpar_json_u64(j, "schema", XPAR_JSON_SCHEMA);
}

void xpar_json_end(xpar_json * j) {
  if (!j->enabled) return;
  xpar_assert(j->in_object);
  j->in_object = false;
  put(j, "}\n");
  xpar_flush(j->out);
}

/*  Emit one JSON string, quotes included, escaping per RFC 8259.  */
static void emit_string(xpar_json * j, const char * s, sz n) {
  static const char hex_digits[] = "0123456789abcdef";
  char esc[8];
  sz i = 0;

  put(j, "\"");
  while (i < n) {
    u8 c = (u8) s[i];

    if (c == '"' || c == '\\') {
      esc[0] = '\\';  esc[1] = (char) c;  esc[2] = 0;
      put(j, esc);  i++;  continue;
    }
    if (c == '\n') { put(j, "\\n");  i++;  continue; }
    if (c == '\r') { put(j, "\\r");  i++;  continue; }
    if (c == '\t') { put(j, "\\t");  i++;  continue; }
    if (c == '\b') { put(j, "\\b");  i++;  continue; }
    if (c == '\f') { put(j, "\\f");  i++;  continue; }
    if (c < 0x20) {
      esc[0] = '\\';  esc[1] = 'u';  esc[2] = '0';  esc[3] = '0';
      esc[4] = hex_digits[c >> 4];  esc[5] = hex_digits[c & 15];
      esc[6] = 0;
      put(j, esc);  i++;  continue;
    }
    if (c < 0x80) {
      esc[0] = (char) c;  esc[1] = 0;
      put(j, esc);  i++;  continue;
    }

    {
      sz len = 0, k;
      u32 cp = 0;
      bool ok = true;

      if ((c & 0xE0) == 0xC0) { len = 2;  cp = c & 0x1Fu; }
      else if ((c & 0xF0) == 0xE0) { len = 3;  cp = c & 0x0Fu; }
      else if ((c & 0xF8) == 0xF0) { len = 4;  cp = c & 0x07u; }
      else ok = false;

      if (ok && i + len > n) ok = false;
      for (k = 1; ok && k < len; k++) {
        u8 cc = (u8) s[i + k];
        if ((cc & 0xC0) != 0x80) ok = false;
        else cp = (cp << 6) | (cc & 0x3Fu);
      }
      if (ok && len == 2 && cp < 0x80)    ok = false;
      if (ok && len == 3 && cp < 0x800)   ok = false;
      if (ok && len == 4 && cp < 0x10000) ok = false;
      if (ok && cp >= 0xD800 && cp <= 0xDFFF) ok = false;
      if (ok && cp > 0x10FFFF) ok = false;

      if (!ok) { put(j, "\\ufffd");  i++;  continue; }
      for (k = 0; k < len; k++) {
        esc[0] = s[i + k];  esc[1] = 0;
        put(j, esc);
      }
      i += len;
    }
  }
  put(j, "\"");
}

void xpar_json_strn(xpar_json * j, const char * k, const char * v, sz n) {
  if (!j->enabled) return;
  key(j, k);
  emit_string(j, v, n);
}

void xpar_json_str(xpar_json * j, const char * k, const char * v) {
  if (!j->enabled) return;
  if (!v) { xpar_json_null(j, k);  return; }
  xpar_json_strn(j, k, v, xpar_strlen(v));
}

void xpar_json_u64(xpar_json * j, const char * k, u64 v) {
  if (!j->enabled) return;
  key(j, k);
  xpar_fprintf(j->out, "%" PRIu64, v);
}

void xpar_json_i64(xpar_json * j, const char * k, i64 v) {
  if (!j->enabled) return;
  key(j, k);
  xpar_fprintf(j->out, "%" PRId64, v);
}

void xpar_json_bool(xpar_json * j, const char * k, bool v) {
  if (!j->enabled) return;
  key(j, k);
  put(j, v ? "true" : "false");
}

void xpar_json_null(xpar_json * j, const char * k) {
  if (!j->enabled) return;
  key(j, k);
  put(j, "null");
}

void xpar_json_hex(xpar_json * j, const char * k, const u8 * p, sz n) {
  static const char hex_digits[] = "0123456789abcdef";
  char buf[3];
  sz i;
  if (!j->enabled) return;
  key(j, k);
  put(j, "\"");
  buf[2] = 0;
  for (i = 0; i < n; i++) {
    buf[0] = hex_digits[p[i] >> 4];
    buf[1] = hex_digits[p[i] & 15];
    put(j, buf);
  }
  put(j, "\"");
}

void xpar_json_progress(xpar_json * j, u64 done, u64 total, u64 rate_bps) {
  if (!j->enabled) return;
  xpar_json_begin(j, "progress");
  xpar_json_u64(j, "done", done);
  if (total) xpar_json_u64(j, "total", total);
  else       xpar_json_null(j, "total");
  xpar_json_u64(j, "rate_bps", rate_bps);
  xpar_json_end(j);
}

void xpar_json_progress_sink(void * user, u64 done, u64 total,
                             u64 rate_bps) {
  xpar_json_progress((xpar_json *) user, done, total, rate_bps);
}

const char * xpar_status_word(int exit_code) {
  switch (exit_code) {
    case XPAR_EXIT_OK:            return "clean";
    case XPAR_EXIT_REPAIRABLE:    return "repairable";
    case XPAR_EXIT_UNREPAIRABLE:  return "unrepairable";
    case XPAR_EXIT_NOTFOUND:      return "not-found";
    case XPAR_EXIT_USAGE:         return "usage";
    case XPAR_EXIT_IO:            return "io-error";
    case XPAR_EXIT_AUTH:          return "auth";
    default:                      return "error";
  }
}

void xpar_json_summary(xpar_json * j, const char * status, int exit_code) {
  if (!j->enabled) return;
  xpar_json_begin(j, "summary");
  xpar_json_str(j, "status", status);
  xpar_json_i64(j, "exit", exit_code);
  xpar_json_end(j);
}

/*  Mirror fatal errors to JSON when enabled.  */
static bool json_fatal_on;

void xpar_json_fatal_enable(bool on) { json_fatal_on = on; }

static void json_fatal_text(int code, const char * msg) {
  xpar_json j;
  if (!json_fatal_on) return;
  /*  Prevent recursion while emitting the record.  */
  json_fatal_on = false;
  xpar_json_init(&j, xpar_stdout, true);
  xpar_json_begin(&j, "summary");
  xpar_json_str(&j, "status", "error");
  xpar_json_i64(&j, "exit", code);
  xpar_json_str(&j, "message", msg);
  xpar_json_end(&j);
}

/*  Strip trailing diagnostic newlines.  */
static void json_trim(char * s) {
  sz n = 0;
  while (s[n]) n++;
  while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = 0;
}

void xpar_json_fatal(int code, const char * fmt, ...) {
  char msg[4096];
  va_list ap;
  if (!json_fatal_on) return;
  va_start(ap, fmt);
  xpar_vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);
  json_trim(msg);
  json_fatal_text(code, msg);
}

void xpar_fatal(int code, const char * fmt, ...) {
  char msg[4096];
  va_list ap;
  va_start(ap, fmt);
  xpar_vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);
  json_trim(msg);
  xpar_fprintf(xpar_stderr, "xpar: %s\n", msg);
  json_fatal_text(code, msg);
  xpar_exit(code);
}
