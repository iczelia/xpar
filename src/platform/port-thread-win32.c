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

/*  Win32 worker pool using Windows 95-compatible synchronization.  */

#if !(defined(_WIN32) || defined(__MINGW32__) || defined(__MINGW64__))
#error "port-thread-win32.c compiled for a non-Windows target"
#endif

#if !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0400
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "common.h"
#include "port-thread.h"

int xpar_cpu_count(void) {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwNumberOfProcessors > 0 ? (int) si.dwNumberOfProcessors : 1;
}

int xpar_core_count(void) {
  typedef BOOL (WINAPI * glpi_fn)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION,
                                  PDWORD);
  glpi_fn glpi = (glpi_fn) (void *) GetProcAddress(
    GetModuleHandleA("kernel32.dll"), "GetLogicalProcessorInformation");
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION p;
  DWORD bytes = 0, i, n;
  int cores = 0;
  if (!glpi || glpi(NULL, &bytes) ||
      GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    return xpar_cpu_count();
  p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION) xpar_alloc_raw(bytes);
  if (!glpi(p, &bytes)) { xpar_free(p);  return xpar_cpu_count(); }
  n = bytes / sizeof *p;
  Fi(n, if (p[i].Relationship == RelationProcessorCore) cores++);
  xpar_free(p);
  return cores > 0 ? cores : xpar_cpu_count();
}

struct xpar_pool {
  int              nthreads;   /*  workers plus the calling thread  */
  HANDLE *         thread;     /*  nthreads - 1 entries  */
  HANDLE *         wake;       /*  one auto-reset event per worker  */
  HANDLE           joined;     /*  set by the last worker to finish  */
  CRITICAL_SECTION cs;
  xpar_work_fn     fn;
  void *           ctx;
  sz               n, next;
  LONG             busy;
  volatile LONG    stop;
};

struct worker_arg { struct xpar_pool * p;  int k; };

static sz claim_size(sz left, int t) {
  sz c = left / ((sz) t * 4);
  if (c == 0) c = 1;
  if (c > 64) c = 64;
  return c;
}

static void drain(struct xpar_pool * p) {
  for (;;) {
    xpar_work_fn fn;
    void * ctx;
    sz start, count, i;
    EnterCriticalSection(&p->cs);
    if (p->next >= p->n) { LeaveCriticalSection(&p->cs);  return; }
    count = claim_size(p->n - p->next, p->nthreads);
    start = p->next;
    p->next += count;
    fn = p->fn;  ctx = p->ctx;
    LeaveCriticalSection(&p->cs);
    for (i = start; i < start + count; i++) fn(i, ctx);
  }
}

static DWORD WINAPI worker(LPVOID arg) {
  struct worker_arg a = *(struct worker_arg *) arg;
  xpar_free(arg);
  for (;;) {
    WaitForSingleObject(a.p->wake[a.k], INFINITE);
    if (a.p->stop) return 0;
    drain(a.p);
    if (InterlockedDecrement(&a.p->busy) == 0) SetEvent(a.p->joined);
  }
}

xpar_pool * xpar_pool_create(int threads) {
  struct xpar_pool * p = xpar_alloc_raw(sizeof *p);
  int k;
  if (threads <= 0) threads = xpar_cpu_count();
  if (threads < 1)  threads = 1;
  p->nthreads = threads;
  p->thread = NULL;  p->wake = NULL;  p->joined = NULL;
  p->fn = NULL;  p->ctx = NULL;
  p->n = p->next = 0;  p->busy = 0;  p->stop = 0;
  InitializeCriticalSection(&p->cs);
  if (threads == 1) return p;

  p->joined = CreateEvent(NULL, FALSE, FALSE, NULL);
  p->thread = xpar_alloc_raw((sz) (threads - 1) * sizeof *p->thread);
  p->wake   = xpar_alloc_raw((sz) (threads - 1) * sizeof *p->wake);
  Fk(threads - 1, p->thread[k] = NULL;  p->wake[k] = NULL);
  Fk(threads - 1,
    struct worker_arg * a;
    p->wake[k] = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (!p->wake[k] || !p->joined) { p->nthreads = k + 1;  break; }
    a = xpar_alloc_raw(sizeof *a);
    a->p = p;  a->k = k;
    p->thread[k] = CreateThread(NULL, 0, worker, a, 0, NULL);
    if (!p->thread[k]) {
      /*  Fewer threads is slower and still correct, so a pool that could
          not start its full complement runs with what it has.  */
      xpar_free(a);
      CloseHandle(p->wake[k]);
      p->wake[k] = NULL;
      p->nthreads = k + 1;
      break;
    });
  return p;
}

int xpar_pool_threads(const xpar_pool * p) { return p ? p->nthreads : 1; }

void xpar_pool_run(xpar_pool * p, sz n, xpar_work_fn fn, void * ctx) {
  int k;
  if (n == 0) return;
  if (!p || p->nthreads <= 1 || n == 1) {
    sz i;
    Fi(n, fn(i, ctx));
    return;
  }
  EnterCriticalSection(&p->cs);
  p->fn = fn;  p->ctx = ctx;
  p->n = n;    p->next = 0;
  p->busy = (LONG) (p->nthreads - 1);
  LeaveCriticalSection(&p->cs);

  /*  SetEvent carries the publication of the fields above to the woken
      thread; the kernel transition is a full barrier on every Windows
      target.  */
  Fk(p->nthreads - 1, SetEvent(p->wake[k]));

  drain(p);

  WaitForSingleObject(p->joined, INFINITE);
}

void xpar_pool_destroy(xpar_pool * p) {
  int k;
  if (!p) return;
  if (p->thread) {
    p->stop = 1;
    Fk(p->nthreads - 1, if (p->wake[k]) SetEvent(p->wake[k]));
    Fk(p->nthreads - 1,
      if (!p->thread[k]) continue;
      WaitForSingleObject(p->thread[k], INFINITE);
      CloseHandle(p->thread[k]));
    Fk(p->nthreads - 1, if (p->wake[k]) CloseHandle(p->wake[k]));
    xpar_free(p->thread);
    xpar_free(p->wake);
  }
  if (p->joined) CloseHandle(p->joined);
  DeleteCriticalSection(&p->cs);
  xpar_free(p);
}
