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

/*  pthread worker pool. The caller is one worker; a generation counter
    publishes each batch without a lost-wakeup window.  */

#include "common.h"
#include "port-thread.h"
#include <pthread.h>
#include <fcntl.h>
#include <limits.h>
#include <unistd.h>
#if defined(HAVE_SCHED_H)
#include <sched.h>
#endif

#if defined(__linux__) && defined(HAVE_SCHED_GETAFFINITY)
static bool affinity_get(cpu_set_t * set) {
  CPU_ZERO(set);
  return sched_getaffinity(0, sizeof *set, set) == 0;
}

static int affinity_count(const cpu_set_t * set) {
  int n = 0;
  sz i;
  Fi((sz) CPU_SETSIZE, if (CPU_ISSET(i, set)) n++);
  return n;
}
#endif

int xpar_cpu_count(void) {
#if defined(HAVE_SYSCONF) && defined(_SC_NPROCESSORS_ONLN)
  long online;
#endif
#if defined(__linux__) && defined(HAVE_SCHED_GETAFFINITY)
  cpu_set_t set;
  if (affinity_get(&set)) { int n = affinity_count(&set);  if (n > 0) return n; }
#endif
#if defined(HAVE_SYSCONF) && defined(_SC_NPROCESSORS_ONLN)
  online = sysconf(_SC_NPROCESSORS_ONLN);
  if (online > 0) return (int) online;
#endif
  return 1;
}

#if defined(__linux__)
static int topology_id(const char * item, int cpu) {
  char path[128], buf[32];
  int fd, value = 0;
  ssize_t n, i;
  xpar_snprintf(path, sizeof path,
                "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, item);
  fd = open(path, O_RDONLY);
  if (fd < 0) return -1;
  n = read(fd, buf, sizeof buf);
  close(fd);
  if (n <= 0) return -1;
  Fi(n,
    if (buf[i] < '0' || buf[i] > '9') break;
    if (value > (INT_MAX - (buf[i] - '0')) / 10) return -1;
    value = value * 10 + buf[i] - '0');
  return value;
}
#endif

int xpar_core_count(void) {
#if defined(__linux__) && defined(HAVE_SYSCONF) && defined(_SC_NPROCESSORS_CONF)
  long configured = sysconf(_SC_NPROCESSORS_CONF);
  int * pair, count = 0, cpu;
#if defined(HAVE_SCHED_GETAFFINITY)
  cpu_set_t set;
  bool have_set = affinity_get(&set);
#endif
  if (configured <= 0 || configured > INT_MAX / 2) return xpar_cpu_count();
  pair = xpar_alloc_raw((sz) configured * 2 * sizeof *pair);
  for (cpu = 0; cpu < configured; cpu++) {
    int package, core, i;
#if defined(HAVE_SCHED_GETAFFINITY)
    if (have_set &&
        (cpu >= CPU_SETSIZE || !CPU_ISSET((size_t) cpu, &set))) continue;
#endif
    package = topology_id("physical_package_id", cpu);
    core = topology_id("core_id", cpu);
    if (package < 0 || core < 0) continue;
    Fi(count, if (pair[2 * i] == package && pair[2 * i + 1] == core) break);
    if (i == count) { pair[2 * count] = package;  pair[2 * count + 1] = core;  count++; }
  }
  xpar_free(pair);
  if (count > 0) return MIN(count, xpar_cpu_count());
#endif
  return xpar_cpu_count();
}

struct xpar_pool {
  int             nthreads;   /*  workers plus the calling thread  */
  pthread_t *     tid;        /*  nthreads - 1 entries  */
  pthread_mutex_t m;
  pthread_cond_t  ready;      /*  a batch was published  */
  pthread_cond_t  done;       /*  a worker finished a batch  */
  xpar_work_fn    fn;
  void *          ctx;
  sz              n, next;
  u64             gen;
  int             busy;
  bool            stop;
};

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
    pthread_mutex_lock(&p->m);
    if (p->next >= p->n) { pthread_mutex_unlock(&p->m);  return; }
    count = claim_size(p->n - p->next, p->nthreads);
    start = p->next;
    p->next += count;
    fn = p->fn;  ctx = p->ctx;
    pthread_mutex_unlock(&p->m);
    for (i = start; i < start + count; i++) fn(i, ctx);
  }
}

static void * worker(void * arg) {
  struct xpar_pool * p = arg;
  u64 seen = 0;
  for (;;) {
    pthread_mutex_lock(&p->m);
    while (!p->stop && p->gen == seen) pthread_cond_wait(&p->ready, &p->m);
    if (p->stop) { pthread_mutex_unlock(&p->m);  return NULL; }
    seen = p->gen;
    pthread_mutex_unlock(&p->m);

    drain(p);

    pthread_mutex_lock(&p->m);
    if (--p->busy == 0) pthread_cond_broadcast(&p->done);
    pthread_mutex_unlock(&p->m);
  }
}

xpar_pool * xpar_pool_create(int threads) {
  struct xpar_pool * p = xpar_alloc_raw(sizeof *p);
  int k;
  if (threads <= 0) threads = xpar_cpu_count();
  if (threads < 1)  threads = 1;
  p->nthreads = threads;  p->tid = NULL;
  p->fn = NULL;  p->ctx = NULL;
  p->n = p->next = 0;  p->gen = 0;  p->busy = 0;  p->stop = false;
  if (pthread_mutex_init(&p->m, NULL) != 0) FATAL("pthread_mutex_init");
  if (pthread_cond_init(&p->ready, NULL) != 0) FATAL("pthread_cond_init");
  if (pthread_cond_init(&p->done, NULL) != 0) FATAL("pthread_cond_init");
  if (threads == 1) return p;

  p->tid = xpar_alloc_raw((sz) (threads - 1) * sizeof *p->tid);
  Fk(threads - 1,
    if (pthread_create(&p->tid[k], NULL, worker, p) != 0) {
      /*  Run with the threads that did start rather than failing the
          operation: fewer threads is slower and still correct.  */
      p->nthreads = k + 1;
      break;
    });
  return p;
}

int xpar_pool_threads(const xpar_pool * p) { return p ? p->nthreads : 1; }

void xpar_pool_run(xpar_pool * p, sz n, xpar_work_fn fn, void * ctx) {
  if (n == 0) return;
  if (!p || p->nthreads <= 1 || n == 1) {
    sz i;
    Fi(n, fn(i, ctx));
    return;
  }
  pthread_mutex_lock(&p->m);
  p->fn = fn;  p->ctx = ctx;
  p->n = n;    p->next = 0;
  p->busy = p->nthreads - 1;
  p->gen++;
  pthread_cond_broadcast(&p->ready);
  pthread_mutex_unlock(&p->m);

  drain(p);

  pthread_mutex_lock(&p->m);
  while (p->busy > 0) pthread_cond_wait(&p->done, &p->m);
  pthread_mutex_unlock(&p->m);
}

void xpar_pool_destroy(xpar_pool * p) {
  int k;
  if (!p) return;
  pthread_mutex_lock(&p->m);
  p->stop = true;
  pthread_cond_broadcast(&p->ready);
  pthread_mutex_unlock(&p->m);
  if (p->tid) { Fk(p->nthreads - 1, pthread_join(p->tid[k], NULL));  xpar_free(p->tid); }
  pthread_cond_destroy(&p->done);
  pthread_cond_destroy(&p->ready);
  pthread_mutex_destroy(&p->m);
  xpar_free(p);
}
