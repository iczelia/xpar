/*  Copyright (C) 2022-2026 Kamila Szewczyk
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; version 3 of the License only.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <http://www.gnu.org/licenses/>.  */

/*  Run a program with arguments read from a file and preserve its status.  */

#include <fcntl.h>
#include <dpmi.h>
#include <process.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

static int save_status(const char * path, int status) {
  FILE * f;
  if (!path) return 0;
  f = fopen(path, "wb");
  if (!f) return -1;
  fprintf(f, "%d\n", status);
  return fclose(f);
}

static int redirect_to(const char * path, int fd) {
  int out;
  if (!path) return 0;
  out = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0666);
  if (out < 0) return -1;
  if (dup2(out, fd) < 0) { close(out);  return -1; }
  close(out);
  return 0;
}

static char ** read_args(const char * path, const char * program,
                         char ** storage) {
  FILE * f = fopen(path, "rb");
  char * data;
  char ** out;
  long n;
  size_t count = 0, i, at;
  if (!f || fseek(f, 0, SEEK_END) != 0 || (n = ftell(f)) < 0 ||
      fseek(f, 0, SEEK_SET) != 0) {
    if (f) fclose(f);
    return NULL;
  }
  data = (char *) malloc((size_t) n + 1);
  if (!data || fread(data, 1, (size_t) n, f) != (size_t) n) {
    free(data);  fclose(f);  return NULL;
  }
  fclose(f);
  data[n] = 0;
  for (i = 0; i < (size_t) n; i++) if (!data[i]) count++;
  if (n && data[n - 1]) count++;
  out = (char **) malloc((count + 2) * sizeof *out);
  if (!out) { free(data);  return NULL; }
  out[0] = (char *) program;
  at = 1;
  for (i = 0; i < (size_t) n;) {
    out[at++] = data + i;
    while (i < (size_t) n && data[i]) i++;
    i++;
  }
  out[at] = NULL;
  *storage = data;
  return out;
}

static int serve(const char * ready, const char * stop, const char * command) {
  const char * shell = getenv("COMSPEC");
  if (!shell || !*shell) shell = "COMMAND.COM";
  while (access(stop, F_OK) != 0) {
    if (access(ready, F_OK) == 0) {
      spawnlp(P_WAIT, shell, shell, "/C", command, NULL);
      unlink(ready);
    } else {
      __dpmi_yield();
      usleep(10000);
    }
  }
  return 0;
}

int main(int argc, char ** argv) {
  const char * status_path = NULL;
  const char * args_path = NULL;
  const char * stdout_path = NULL;
  const char * stderr_path = NULL;
  const char * program;
  char * storage = NULL;
  char ** child;
  int i = 1, status;
  if (argc == 5 && !strcmp(argv[1], "--server"))
    return serve(argv[2], argv[3], argv[4]);
  while (i < argc) {
    const char ** value;
    if      (!strcmp(argv[i], "--status")) value = &status_path;
    else if (!strcmp(argv[i], "--stdout")) value = &stdout_path;
    else if (!strcmp(argv[i], "--stderr")) value = &stderr_path;
    else if (!strcmp(argv[i], "--args"))   value = &args_path;
    else break;
    if (++i == argc) break;
    *value = argv[i++];
  }
  if (i >= argc) {
    fprintf(stderr,
            "usage: run2 --server READY STOP COMMAND\n"
            "usage: run2 [--status FILE] [--stdout FILE] [--stderr FILE] "
            "[--args FILE] PROGRAM [ARGUMENT...]\n");
    return 2;
  }
  program = argv[i];
  child = args_path ? read_args(args_path, program, &storage) : argv + i;
  if (!child) {
    perror("run2: arguments");
    return 2;
  }
  if (redirect_to(stdout_path, 1) != 0 || redirect_to(stderr_path, 2) != 0) {
    perror("run2: redirect");
    free(storage);
    if (args_path) free(child);
    return 2;
  }
  status = spawnvp(P_WAIT, program, child);
  if (status < 0) {
    perror("run2: spawn");
    status = 127;
  }
  if (save_status(status_path, status) != 0) {
    perror("run2: status");
    status = 2;
  }
  free(storage);
  if (args_path) free(child);
  return status;
}
