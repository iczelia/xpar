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

/* Standalone command timer: elapsed_us, status, maxrss_kb, block I/O. */

/* Expose XSI process and resource APIs under strict C99. */
#if !defined(_WIN32) && !defined(__MSDOS__) && !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 700
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(__MSDOS__)
#include <time.h>
#define TIMEIT_SYSTEM 1
#else
#include <errno.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

static void emit(const char * path, unsigned long long usec, int status,
                 unsigned long long maxrss_kb, unsigned long long in_blocks,
                 unsigned long long out_blocks) {
  FILE * f = path ? fopen(path, "w") : stderr;
  if (!f) { perror(path);  return; }
  fprintf(f, "elapsed_us=%llu\nstatus=%d\nmaxrss_kb=%llu\n"
             "in_blocks=%llu\nout_blocks=%llu\n",
          usec, status, maxrss_kb, in_blocks, out_blocks);
  if (path) fclose(f);
}

int main(int argc, char ** argv) {
  const char * out;

  if (argc < 3) {
    fprintf(stderr, "usage: timeit <record-file|-> <command> [args...]\n");
    return 2;
  }
  out = strcmp(argv[1], "-") ? argv[1] : NULL;

#if defined(TIMEIT_SYSTEM)
  {
    /* system() requires one string; benchmark paths contain no spaces. */
    size_t len = 1, i;
    char * line;
    clock_t begin, end;
    int status;
    for (i = 2; i < (size_t) argc; i++) len += strlen(argv[i]) + 1;
    line = (char *) malloc(len);
    if (!line) return 2;
    line[0] = 0;
    for (i = 2; i < (size_t) argc; i++) {
      strcat(line, argv[i]);
      if (i + 1 < (size_t) argc) strcat(line, " ");
    }
    begin = clock();
    status = system(line);
    end = clock();
    free(line);
    emit(out, (unsigned long long) ((double) (end - begin) * 1000000.0 /
                                    (double) CLOCKS_PER_SEC),
         status, 0, 0, 0);
    return status ? 1 : 0;
  }
#else
  {
    struct timeval a, b;
    struct rusage ru;
    pid_t pid;
    long long elapsed;
    int wstatus = 0, code;

    gettimeofday(&a, NULL);
    pid = fork();
    if (pid < 0) { perror("fork");  return 2; }
    if (pid == 0) {
      execvp(argv[2], argv + 2);
      perror(argv[2]);
      _exit(127);
    }
    while (waitpid(pid, &wstatus, 0) < 0) {
      if (errno == EINTR) continue;
      perror("waitpid");
      return 2;
    }
    gettimeofday(&b, NULL);
    getrusage(RUSAGE_CHILDREN, &ru);

    code = WIFEXITED(wstatus) ? WEXITSTATUS(wstatus)
         : WIFSIGNALED(wstatus) ? 128 + WTERMSIG(wstatus) : 255;
    /* Keep timestamp subtraction signed. */
    elapsed = (long long) (b.tv_sec - a.tv_sec) * 1000000ll +
              (long long) (b.tv_usec - a.tv_usec);
    if (elapsed < 0) elapsed = 0;
    emit(out, (unsigned long long) elapsed,
         code,
         /* ru_maxrss units are platform-specific. */
         (unsigned long long) ru.ru_maxrss,
         /* Block-layer traffic in 512-byte units; near zero when warm. */
         (unsigned long long) ru.ru_inblock,
         (unsigned long long) ru.ru_oublock);
    return code;
  }
#endif
}
