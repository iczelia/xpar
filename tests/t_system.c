/*  Copyright (C) 2022-2026 Kamila Szewczyk

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; version 3 of the License only.  */

#include "t_system.h"

#if defined(XPAR_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(XPAR_DOS)
#include <fcntl.h>
#include <process.h>
#include <unistd.h>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

static bool is_sep(char c) { return c == '/' || c == '\\'; }

static u32 process_id(void) {
#if defined(XPAR_WIN32)
  return (u32) GetCurrentProcessId();
#else
  return (u32) getpid();
#endif
}

static bool is_absolute(const char * p) {
  return is_sep(p[0]) || (p[0] && p[1] == ':' && is_sep(p[2]));
}

bool xt_path(char * out, sz cap, const char * dir, const char * leaf) {
  sz dn = xpar_strlen(dir), ln = xpar_strlen(leaf);
  bool slash = dn && !is_sep(dir[dn - 1]);
  if (dn + (slash ? 1 : 0) + ln + 1 > cap) return false;
  xpar_memcpy(out, dir, dn);
  if (slash) out[dn++] = '/';
  xpar_memcpy(out + dn, leaf, ln + 1);
  return true;
}

static char * absolute_dup(const char * cwd, const char * path) {
  char buf[XT_PATH_MAX];
  if (is_absolute(path)) return xpar_strdup(path);
  if (!xt_path(buf, sizeof buf, cwd, path)) return NULL;
  return xpar_strdup(buf);
}

bool xt_remove_tree(const char * path) {
  xpar_stat_t st;
  xpar_dir * dir;
  const xpar_dirent * ent;
  bool ok = true;

  if (xpar_lstat(path, &st) != 0)
    return xpar_errno_absent(xpar_errno());
  if (!st.is_dir || st.is_symlink) return xpar_remove(path) == 0;

  dir = xpar_opendir(path);
  if (!dir) return false;
  while ((ent = xpar_readdir(dir)) != NULL) {
    char child[XT_PATH_MAX];
    if (!xt_path(child, sizeof child, path, ent->name) ||
        !xt_remove_tree(child)) ok = false;
  }
  xpar_closedir(dir);
  return xpar_rmdir(path) == 0 && ok;
}

bool xt_mkdir(const char * path) {
  xpar_stat_t st;
  if (xpar_mkdir(path, 0700) == 0) return true;
  return xpar_lstat(path, &st) == 0 && st.is_dir;
}

bool xt_context_init(xt_context * c, int argc, char ** argv) {
  static const char * const version[] = { "--version", NULL };
  const char * xpar = xpar_getenv("XPAR");
  char def[XT_PATH_MAX], leaf[16];
  u32 pid = process_id(), attempt;
  bool made = false;
  int i;

  xpar_memset(c, 0, sizeof *c);
  c->cwd = xpar_getcwd();
  if (!c->cwd) return false;
  for (i = 1; i < argc; i++)
    if (!xpar_strncmp(argv[i], "--xpar=", 7)) xpar = argv[i] + 7;
    else if (!xpar_strcmp(argv[i], "--keep")) c->keep = 1;
  if (!xpar || !*xpar) {
#if defined(XPAR_WIN32) || defined(XPAR_DOS)
    xpar = "xpar.exe";
#else
    xpar = "xpar";
#endif
  }
  c->xpar = absolute_dup(c->cwd, xpar);
  if (!c->xpar) return false;

  for (attempt = 0; attempt < 256; attempt++) {
    xpar_snprintf(leaf, sizeof leaf, "T%07" PRIX32,
                  (pid + attempt) & 0x0FFFFFFFU);
    if (!xt_path(def, sizeof def, c->cwd, leaf)) return false;
    if (xpar_mkdir(def, 0700) == 0) { made = true;  break; }
  }
  if (!made) return false;
  c->root = xpar_strdup(def);
  if (!c->root) { (void) xpar_rmdir(def);  return false; }

  if (!xt_path(def, sizeof def, c->root, "SEED")) return false;
  c->seed = xpar_strdup(def);
  if (!c->seed || !xt_mkdir(c->seed)) return false;
  if (!xt_path(c->out, sizeof c->out, c->root, "OUT.TXT") ||
      !xt_path(c->err, sizeof c->err, c->root, "ERR.TXT")) return false;
  if (xt_run_xpar(c, c->root, version) != 0) return false;
  c->target_link_identity = !xt_file_contains(c->out, ", win95)");
  return true;
}

void xt_context_free(xt_context * c) {
  if (c->keep)
    xpar_fprintf(xpar_stderr, "t_suite kept %s\n", c->root);
  else if (c->root)
    (void) xt_remove_tree(c->root);
  xpar_free(c->cwd);
  xpar_free(c->root);
  xpar_free(c->seed);
  xpar_free(c->xpar);
  xpar_memset(c, 0, sizeof *c);
}

bool xt_case_begin(xt_context * c, char * out, sz cap, const char * name) {
  char leaf[16];
  xpar_snprintf(leaf, sizeof leaf, "C%07" PRIu32, ++c->case_no);
  xt_section_begin(name);
  if (!xt_path(out, cap, c->root, leaf)) return false;
  return xt_remove_tree(out) && xt_mkdir(out);
}

bool xt_write_pattern(const char * path, u64 bytes, u64 seed) {
  u8 buf[16384];
  xt_rng rng;
  xpar_file * f = xpar_open(path, XPAR_O_WRONLY | XPAR_O_CREAT |
                                  XPAR_O_TRUNC);
  if (!f) return false;
  xt_seed(&rng, seed);
  while (bytes) {
    sz take = bytes > sizeof buf ? sizeof buf : (sz) bytes;
    xt_fill(&rng, buf, take);
    if (xpar_write(f, buf, take) != take) { xpar_close(f);  return false; }
    bytes -= take;
  }
  return xpar_close(f) == 0;
}

bool xt_copy_file(const char * from, const char * to) {
  u8 buf[16384];
  xpar_file * in = xpar_open(from, XPAR_O_RDONLY);
  xpar_file * out;
  bool ok = true;
  if (!in) return false;
  out = xpar_open(to, XPAR_O_WRONLY | XPAR_O_CREAT | XPAR_O_TRUNC);
  if (!out) { xpar_close(in);  return false; }
  for (;;) {
    sz n = xpar_read(in, buf, sizeof buf);
    if (!n) break;
    if (xpar_write(out, buf, n) != n) { ok = false;  break; }
  }
  if (xpar_error(in)) ok = false;
  if (xpar_close(in) != 0 || xpar_close(out) != 0) ok = false;
  return ok;
}

bool xt_copy_tree(const char * from, const char * to) {
  xpar_stat_t st;
  xpar_dir * dir;
  const xpar_dirent * ent;
  bool ok = true;
  if (xpar_lstat(from, &st) != 0) return false;
  if (!st.is_dir || st.is_symlink) return xt_copy_file(from, to);
  if (!xt_mkdir(to)) return false;
  dir = xpar_opendir(from);
  if (!dir) return false;
  while ((ent = xpar_readdir(dir)) != NULL) {
    char src[XT_PATH_MAX], dst[XT_PATH_MAX];
    if (!xt_path(src, sizeof src, from, ent->name) ||
        !xt_path(dst, sizeof dst, to, ent->name) ||
        !xt_copy_tree(src, dst)) ok = false;
  }
  xpar_closedir(dir);
  return ok;
}

bool xt_files_equal(const char * a, const char * b) {
  u8 ab[16384], bb[16384];
  xpar_file * af = xpar_open(a, XPAR_O_RDONLY);
  xpar_file * bf = xpar_open(b, XPAR_O_RDONLY);
  bool equal = true;
  if (!af || !bf) { if (af) xpar_close(af);  if (bf) xpar_close(bf);  return false; }
  for (;;) {
    sz an = xpar_read(af, ab, sizeof ab);
    sz bn = xpar_read(bf, bb, sizeof bb);
    if (an != bn || (an && xpar_memcmp(ab, bb, an))) { equal = false;  break; }
    if (!an) break;
  }
  if (xpar_error(af) || xpar_error(bf)) equal = false;
  xpar_close(af);
  xpar_close(bf);
  return equal;
}

bool xt_trees_equal(const char * a, const char * b) {
  xpar_stat_t as, bs;
  xpar_dir * dir;
  const xpar_dirent * ent;
  bool equal = true;
  if (xpar_lstat(a, &as) != 0 || xpar_lstat(b, &bs) != 0) return false;
  if (as.is_dir != bs.is_dir || as.is_regular != bs.is_regular) return false;
  if (!as.is_dir) return xt_files_equal(a, b);
  dir = xpar_opendir(a);
  if (!dir) return false;
  while ((ent = xpar_readdir(dir)) != NULL) {
    char ap[XT_PATH_MAX], bp[XT_PATH_MAX];
    if (!xt_path(ap, sizeof ap, a, ent->name) ||
        !xt_path(bp, sizeof bp, b, ent->name) ||
        !xt_trees_equal(ap, bp)) equal = false;
  }
  xpar_closedir(dir);
  if (!equal) return false;
  dir = xpar_opendir(b);
  if (!dir) return false;
  while ((ent = xpar_readdir(dir)) != NULL) {
    char ap[XT_PATH_MAX];
    if (!xt_path(ap, sizeof ap, a, ent->name) || xpar_lstat(ap, &as) != 0)
      equal = false;
  }
  xpar_closedir(dir);
  return equal;
}

bool xt_file_size(const char * path, u64 * size) {
  xpar_stat_t st;
  if (xpar_lstat(path, &st) != 0 || !st.is_regular) return false;
  *size = st.size;
  return true;
}

bool xt_damage(const char * path, u64 offset, u64 bytes, u64 seed) {
  u8 buf[4096];
  xt_rng rng;
  xpar_file * f = xpar_open(path, XPAR_O_RDWR);
  if (!f) return false;
  xt_seed(&rng, seed);
  while (bytes) {
    sz i, take = bytes > sizeof buf ? sizeof buf : (sz) bytes;
    if (xpar_pread(f, buf, take, offset) != take) { xpar_close(f);  return false; }
    Fi(take,
      u8 v = (u8) xt_next(&rng);
      buf[i] ^= v ? v : 0xA5);
    if (xpar_pwrite(f, buf, take, offset) != take) { xpar_close(f);  return false; }
    offset += take;
    bytes -= take;
  }
  return xpar_close(f) == 0;
}

bool xt_truncate(const char * path, u64 bytes) {
  xpar_file * f = xpar_open(path, XPAR_O_RDWR);
  int rc;
  if (!f) return false;
  rc = xpar_ftruncate(f, bytes);
  if (xpar_close(f) != 0) rc = -1;
  return rc == 0;
}

static bool file_contains(const char * path, const char * needle, bool fold) {
  u8 buf[4096 + 256];
  sz keep = 0, nn = xpar_strlen(needle);
  xpar_file * f;
  if (!nn || nn > 256) return false;
  f = xpar_open(path, XPAR_O_RDONLY);
  if (!f) return false;
  for (;;) {
    sz n = xpar_read(f, buf + keep, 4096);
    sz have = keep + n, i;
    for (i = 0; i + nn <= have; i++) {
      sz j;
      Fj(nn,
        int a = buf[i + j], b = (u8) needle[j];
        if (fold) {
          if (a >= 'A' && a <= 'Z') a += 'a' - 'A';
          if (b >= 'A' && b <= 'Z') b += 'a' - 'A';
        }
        if (a != b) break);
      if (j == nn) { xpar_close(f);  return true; }
    }
    if (!n) break;
    keep = have < nn - 1 ? have : nn - 1;
    xpar_memmove(buf, buf + have - keep, keep);
  }
  xpar_close(f);
  return false;
}

bool xt_file_contains(const char * path, const char * needle) {
  return file_contains(path, needle, false);
}

bool xt_file_contains_ci(const char * path, const char * needle) {
  return file_contains(path, needle, true);
}

static char * read_small_file(const char * path, sz * size) {
  u64 bytes;
  char * data;
  xpar_file * f;
  if (!xt_file_size(path, &bytes) || bytes > 1024 * 1024) return NULL;
  data = xpar_malloc((sz) bytes + 1);
  if (!data) return NULL;
  f = xpar_open(path, XPAR_O_RDONLY);
  if (!f || xpar_read(f, data, (sz) bytes) != (sz) bytes) {
    if (f) xpar_close(f);
    xpar_free(data);
    return NULL;
  }
  xpar_close(f);
  data[bytes] = 0;
  *size = (sz) bytes;
  return data;
}

static const char * json_value(const char * data, sz size,
                               const char * key) {
  char pattern[128];
  sz pn, i;
  const char * found = NULL;
  if (xpar_snprintf(pattern, sizeof pattern, "\"%s\":", key) < 0)
    return NULL;
  pn = xpar_strlen(pattern);
  for (i = 0; i + pn <= size; i++) {
    if (xpar_memcmp(data + i, pattern, pn)) continue;
    found = data + i + pn;
  }
  return found;
}

bool xt_json_u64(const char * path, const char * key, u64 * value) {
  char number[32];
  sz size, n = 0;
  char * data = read_small_file(path, &size);
  const char * p;
  bool ok = false;
  if (!data) return false;
  p = json_value(data, size, key);
  while (p && *p >= '0' && *p <= '9' && n + 1 < sizeof number)
    number[n++] = *p++;
  number[n] = 0;
  if (n && xpar_parse_u64(number, value) == 0) ok = true;
  xpar_free(data);
  return ok;
}

bool xt_json_string(const char * path, const char * key, const char * value) {
  sz size, n = xpar_strlen(value);
  char * data = read_small_file(path, &size);
  const char * p;
  bool ok = false;
  if (!data) return false;
  p = json_value(data, size, key);
  if (p && *p++ == '"' && (sz) (data + size - p) > n &&
      !xpar_memcmp(p, value, n) && p[n] == '"') ok = true;
  xpar_free(data);
  return ok;
}

void xt_dump_file(const char * path, const char * label) {
  u8 buf[4096];
  xpar_file * f = xpar_open(path, XPAR_O_RDONLY);
  if (!f) return;
  xpar_fprintf(xpar_stderr, "  %s\n", label);
  if (xt_report_file) xpar_fprintf(xt_report_file, "  %s\n", label);
  for (;;) {
    sz n = xpar_read(f, buf, sizeof buf);
    if (!n) break;
    (void) xpar_write(xpar_stderr, buf, n);
    if (xt_report_file) (void) xpar_write(xt_report_file, buf, n);
  }
  (void) xpar_close(f);
}

static int fold_ascii(int c) {
  return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c;
}

static bool starts_with(const char * s, const char * p) {
  while (*p) {
    if (!*s) return false;
    if (fold_ascii((u8) *s++) != fold_ascii((u8) *p++)) return false;
  }
  return true;
}

static bool ends_with(const char * s, const char * p) {
  sz sn = xpar_strlen(s), pn = xpar_strlen(p), i;
  if (sn < pn) return false;
  Fi(pn,
    if (fold_ascii((u8) s[sn - pn + i]) != fold_ascii((u8) p[i]))
      return false);
  return true;
}

bool xt_find_file(const char * dir, const char * prefix, const char * suffix,
                  char * out, sz cap) {
  xpar_dir * d = xpar_opendir(dir);
  const xpar_dirent * ent;
  bool found = false;
  if (!d) return false;
  while ((ent = xpar_readdir(d)) != NULL) {
    if (!ent->is_regular || !starts_with(ent->name, prefix) ||
        !ends_with(ent->name, suffix)) continue;
    found = xt_path(out, cap, dir, ent->name);
    break;
  }
  xpar_closedir(d);
  return found;
}

#if defined(XPAR_WIN32)

static bool cmd_append(char * out, sz cap, sz * used, const char * arg) {
  sz i, slashes = 0;
  bool quote = !*arg;
  for (i = 0; arg[i]; i++)
    if (arg[i] == ' ' || arg[i] == '\t' || arg[i] == '"') quote = true;
  if (*used) { if (*used + 1 >= cap) return false;  out[(*used)++] = ' '; }
  if (quote) { if (*used + 1 >= cap) return false;  out[(*used)++] = '"'; }
  for (i = 0; arg[i]; i++) {
    if (arg[i] == '\\') { slashes++;  continue; }
    if (arg[i] == '"') {
      while (slashes) {
        if (*used + 2 >= cap) return false;
        out[(*used)++] = '\\';  out[(*used)++] = '\\';
        slashes--;
      }
      if (*used + 2 >= cap) return false;
      out[(*used)++] = '\\';  out[(*used)++] = '"';
    } else {
      while (slashes && *used + 1 < cap) { out[(*used)++] = '\\';  slashes--; }
      if (*used + 1 >= cap) return false;
      out[(*used)++] = arg[i];
    }
  }
  while (slashes) {
    if (quote) {
      if (*used + 2 >= cap) return false;
      out[(*used)++] = '\\';  out[(*used)++] = '\\';
    } else {
      if (*used + 1 >= cap) return false;
      out[(*used)++] = '\\';
    }
    slashes--;
  }
  if (quote) { if (*used + 1 >= cap) return false;  out[(*used)++] = '"'; }
  if (*used >= cap) return false;
  out[*used] = 0;
  return true;
}

static int run_process(const char * exe, const char * cwd,
                       const char * out_path, const char * err_path,
                       const char * const * argv) {
  SECURITY_ATTRIBUTES sa;
  STARTUPINFOA si;
  PROCESS_INFORMATION pi;
  HANDLE out, err;
  DWORD code = 127;
  char cmd[32768];
  sz used = 0, i;
  bool ok = true;

  xpar_memset(&sa, 0, sizeof sa);
  sa.nLength = sizeof sa;
  sa.bInheritHandle = TRUE;
  out = CreateFileA(out_path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                    &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  err = CreateFileA(err_path, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                    &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  if (out == INVALID_HANDLE_VALUE || err == INVALID_HANDLE_VALUE) {
    if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
    if (err != INVALID_HANDLE_VALUE) CloseHandle(err);
    return 127;
  }
  for (i = 0; argv[i]; i++)
    if (!cmd_append(cmd, sizeof cmd, &used, argv[i])) ok = false;
  xpar_memset(&si, 0, sizeof si);
  xpar_memset(&pi, 0, sizeof pi);
  si.cb = sizeof si;
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  si.hStdOutput = out;
  si.hStdError = err;
  if (ok && CreateProcessA(exe, cmd, NULL, NULL, TRUE, 0, NULL, cwd, &si, &pi)) {
    WaitForSingleObject(pi.hProcess, INFINITE);
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
  }
  CloseHandle(out);
  CloseHandle(err);
  return (int) code;
}

#elif defined(XPAR_DOS)

static int run_process(const char * exe, const char * cwd,
                       const char * out_path, const char * err_path,
                       const char * const * argv) {
  char old[XT_PATH_MAX];
  int out, err, save_out, save_err, rc = 127;
  if (!getcwd(old, sizeof old)) return 127;
  out = open(out_path, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0600);
  err = open(err_path, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0600);
  if (out < 0 || err < 0) goto done;
  save_out = dup(1);  save_err = dup(2);
  if (save_out < 0 || save_err < 0 || chdir(cwd) != 0) goto restore;
  if (dup2(out, 1) < 0 || dup2(err, 2) < 0) goto restore;
  rc = spawnv(P_WAIT, exe, (char * const *) argv);
restore:
  if (save_out >= 0) { dup2(save_out, 1);  close(save_out); }
  if (save_err >= 0) { dup2(save_err, 2);  close(save_err); }
  (void) chdir(old);
done:
  if (out >= 0) close(out);
  if (err >= 0) close(err);
  return rc;
}

#else

static int run_process(const char * exe, const char * cwd,
                       const char * out_path, const char * err_path,
                       const char * const * argv) {
  int out = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
  int err = open(err_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
  pid_t pid;
  int status;
  if (out < 0 || err < 0) { if (out >= 0) close(out);  if (err >= 0) close(err);  return 127; }
  pid = fork();
  if (pid == 0) {
    if (chdir(cwd) != 0 || dup2(out, 1) < 0 || dup2(err, 2) < 0)
      _exit(127);
    close(out);  close(err);
    execv(exe, (char * const *) argv);
    _exit(127);
  }
  close(out);  close(err);
  if (pid < 0) return 127;
  while (waitpid(pid, &status, 0) < 0)
    if (xpar_errno() != EINTR) return 127;
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return 127;
}

#endif

int xt_run_xpar(xt_context * c, const char * cwd,
                const char * const * args) {
  const char * argv[64];
  sz i;
  argv[0] = c->xpar;
  for (i = 0; args[i] && i + 2 < ARRAY_LEN(argv); i++) argv[i + 1] = args[i];
  if (args[i]) return 127;
  argv[i + 1] = NULL;
  return run_process(c->xpar, cwd, c->out, c->err, argv);
}
