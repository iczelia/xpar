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

/*  DJGPP host I/O over DOS handles and DPMI.

    Signed 32-bit off_t bounds every libc offset.  Positional I/O moves the
    shared offset and relies on the single-threaded build.  Secure random
    bytes, and therefore authentication, are unavailable.  */

#if !defined(__DJGPP__)
#error "port-djgpp.c compiled for a non-DJGPP target"
#endif

#include "common.h"

#include <dir.h>
#include <dos.h>
#include <dpmi.h>
#include <errno.h>
#include <fcntl.h>
#include <go32.h>
#include <io.h>
#include <pc.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/farptr.h>
#include <sys/time.h>

const char * xpar_getenv(const char * name) { return getenv(name); }

/*  DOS keeps no standard temporary directory beyond the environment.  */
const char * xpar_tmpdir(void) {
#ifdef P_tmpdir
  return P_tmpdir;
#else
  return "/tmp";
#endif
}
#include <time.h>
#include <unistd.h>

struct xpar_file {
  int  fd;
  bool is_char;   /*  console, printer or another character device  */
  bool owned;
  bool at_eof;
  int  last_errno;
};

static struct xpar_file g_stdin  = { 0, false, false, false, 0 };
static struct xpar_file g_stdout = { 1, false, false, false, 0 };
static struct xpar_file g_stderr = { 2, false, false, false, 0 };

xpar_file * const xpar_stdin  = &g_stdin;
xpar_file * const xpar_stdout = &g_stdout;
xpar_file * const xpar_stderr = &g_stderr;

/*  int 21h AX=4400h: get device information for a handle. Bit 7 of DX
    distinguishes a character device from a file, which is the only part
    of the answer this program needs.  */
static bool fd_is_char(int fd) {
  __dpmi_regs r;
  memset(&r, 0, sizeof r);
  r.x.ax = 0x4400;
  r.x.bx = (unsigned short) fd;
  if (__dpmi_int(0x21, &r) < 0) return false;
  if (r.x.flags & 1) return false;
  return (r.x.dx & 0x80) != 0;
}

void xpar_host_init(void) {
  g_stdin.is_char  = fd_is_char(0);
  g_stdout.is_char = fd_is_char(1);
  g_stderr.is_char = fd_is_char(2);
  /*  A redirected standard stream carries container bytes, and DOS text
      mode would turn every 0x0A into 0x0D 0x0A on the way out and eat
      0x1A on the way in.  */
  if (!g_stdin.is_char)  setmode(0, O_BINARY);
  if (!g_stdout.is_char) setmode(1, O_BINARY);
  /*  Probe for a long-filename driver once, so open and findfirst take the
      AX=71xxh forms where the drive supports them.  */
  _use_lfn(".");
}

/*  off_t is a signed 32-bit long here, so anything past 2 GiB - 1 is
    rejected at the boundary instead of wrapping.  */
static bool off_fits(u64 v) { return v <= 0x7fffffffULL; }

xpar_file * xpar_open(const char * path, int flags) {
  int acc = flags & 3, oflag;
  int fd;
  struct xpar_file * f;
  if      (acc == XPAR_O_WRONLY) oflag = O_WRONLY;
  else if (acc == XPAR_O_RDWR)   oflag = O_RDWR;
  else                           oflag = O_RDONLY;
  oflag |= O_BINARY;
  if (flags & XPAR_O_CREAT)  oflag |= O_CREAT;
  if (flags & XPAR_O_TRUNC)  oflag |= O_TRUNC;
  if (flags & XPAR_O_EXCL)   oflag |= O_EXCL;
  if (flags & XPAR_O_APPEND) oflag |= O_APPEND;
  fd = open(path, oflag, 0666);
  if (fd < 0) return NULL;
  f = malloc(sizeof(*f));
  if (!f) { close(fd);  errno = ENOMEM;  return NULL; }
  f->fd = fd;  f->is_char = fd_is_char(fd);  f->owned = true;
  f->at_eof = false;  f->last_errno = 0;
  return f;
}

int xpar_close(xpar_file * f) {
  int r;
  if (!f || !f->owned) return 0;
  r = close(f->fd);
  free(f);
  return r;
}

sz xpar_read(xpar_file * f, void * buf, sz n) {
  f->last_errno = 0;
  sz total = 0;
  u8 * p = (u8 *) buf;
  while (total < n) {
    sz want = n - total;
    int got;
    if (want > 0x10000000u) want = 0x10000000u;
    got = read(f->fd, p + total, (unsigned) want);
    if (got < 0)  { f->last_errno = errno;  break; }
    if (got == 0) { f->at_eof = true;       break; }
    total += (sz) got;
  }
  return total;
}

sz xpar_write(xpar_file * f, const void * buf, sz n) {
  f->last_errno = 0;
  sz total = 0;
  const u8 * p = (const u8 *) buf;
  while (total < n) {
    sz want = n - total;
    int wrote;
    if (want > 0x10000000u) want = 0x10000000u;
    wrote = write(f->fd, p + total, (unsigned) want);
    if (wrote < 0)  { f->last_errno = errno;  break; }
    if (wrote == 0) break;
    total += (sz) wrote;
  }
  return total;
}

int xpar_seek(xpar_file * f, i64 off, int whence) {
  int w = whence == XPAR_SEEK_SET ? SEEK_SET
        : whence == XPAR_SEEK_CUR ? SEEK_CUR : SEEK_END;
  if (off > 0 && !off_fits((u64) off)) { errno = EOVERFLOW;  return -1; }
  if (off < -0x7fffffffLL)             { errno = EOVERFLOW;  return -1; }
  if (lseek(f->fd, (long) off, w) < 0) { f->last_errno = errno;  return -1; }
  f->at_eof = false;
  return 0;
}

i64 xpar_tell(xpar_file * f) {
  long p = lseek(f->fd, 0, SEEK_CUR);
  return p < 0 ? -1 : (i64) p;
}

/*  DOS writes go straight to the file system through int 21h; there is no
    user-space buffer to drain and no fsync to ask for.  */
int xpar_flush(xpar_file * f) { (void) f;  return 0; }
int xpar_fsync(xpar_file * f) { (void) f;  return 0; }

i64 xpar_size(xpar_file * f) {
  long len = filelength(f->fd);
  return len < 0 ? -1 : (i64) len;
}

bool xpar_is_seekable(xpar_file * f) {
  return lseek(f->fd, 0, SEEK_CUR) >= 0;
}
bool xpar_is_tty(xpar_file * f) { return f->is_char; }
bool xpar_eof   (xpar_file * f) { return f->at_eof; }
int  xpar_error (xpar_file * f) { return f->last_errno; }

/*  Advisory locks: not available.  */

int xpar_lock(xpar_file * f, bool exclusive) {
  (void) f;  (void) exclusive;
  errno = ENOSYS;
  return -1;
}

/*  Nothing was ever taken, so releasing it succeeds trivially; a caller
    unwinding an error path must not have to special-case this host.  */
int xpar_unlock(xpar_file * f) { (void) f;  return 0; }

bool xpar_lock_supported(void) { return false; }

/*  Emulate positional reads by seeking, then restore cursor and EOF state.  */
sz xpar_pread(xpar_file * f, void * buf, sz n, u64 off) {
  f->last_errno = 0;
  long saved;
  bool saved_eof = f->at_eof;
  sz got;
  if (!off_fits(off)) { f->last_errno = EOVERFLOW;  return 0; }
  saved = lseek(f->fd, 0, SEEK_CUR);
  if (saved < 0) { f->last_errno = errno;  return 0; }
  if (lseek(f->fd, (long) off, SEEK_SET) < 0) {
    f->last_errno = errno;
    return 0;
  }
  got = xpar_read(f, buf, n);
  f->at_eof = saved_eof;
  if (lseek(f->fd, saved, SEEK_SET) < 0) f->last_errno = errno;
  return got;
}

bool xpar_pread_batch(xpar_read_req * r, sz count) {
  return xpar_pread_serial(r, count);
}

sz xpar_pwrite(xpar_file * f, const void * buf, sz n, u64 off) {
  f->last_errno = 0;
  long saved;
  sz put;
  if (!off_fits(off)) { f->last_errno = EOVERFLOW;  return 0; }
  saved = lseek(f->fd, 0, SEEK_CUR);
  if (saved < 0) { f->last_errno = errno;  return 0; }
  if (lseek(f->fd, (long) off, SEEK_SET) < 0) {
    f->last_errno = errno;
    return 0;
  }
  put = xpar_write(f, buf, n);
  if (lseek(f->fd, saved, SEEK_SET) < 0) f->last_errno = errno;
  return put;
}

int xpar_ftruncate(xpar_file * f, u64 length) {
  if (!off_fits(length)) { errno = EOVERFLOW;  return -1; }
  if (ftruncate(f->fd, (long) length) != 0) {
    f->last_errno = errno;
    return -1;
  }
  return 0;
}

/*  FAT has no directory to sync: a directory entry is written by the same
    int 21h call that created the file.  */
int xpar_fsync_dir(const char * path) { (void) path;  return 0; }

sz xpar_xread(xpar_file * f, void * p, sz n) {
  sz got = xpar_read(f, p, n);
  if (f->last_errno) { errno = f->last_errno;  FATAL_PERROR("read"); }
  return got;
}

void xpar_xwrite(xpar_file * f, const void * p, sz n) {
  if (xpar_write(f, p, n) != n) {
    errno = f->last_errno ? f->last_errno : ENOSPC;
    FATAL_PERROR("write");
  }
}

void xpar_xwritev(xpar_file * f, const xpar_write_part * part, u32 count) {
  For(u32, i, count, xpar_xwrite(f, part[i].data, part[i].length))
}

void xpar_xclose(xpar_file * f) {
  if (!f || !f->owned) return;
  if (close(f->fd) < 0) FATAL_PERROR("close");
  free(f);
}

xpar_mmap xpar_map(const char * path) {
  xpar_mmap m;
  (void) path;
  m.map = NULL;  m.size = 0;  m.valid = false;
  return m;
}

void xpar_unmap(xpar_mmap * m) {
  m->map = NULL;  m->size = 0;  m->valid = false;
}

void xpar_advise_sequential(xpar_file * f, u64 off, u64 len) {
  (void) f;  (void) off;  (void) len;
}
void xpar_advise_random(xpar_file * f, u64 off, u64 len) {
  (void) f;  (void) off;  (void) len;
}

void * xpar_malloc(sz n) {
  void * p = calloc(n ? n : 1, 1);
  if (!p) FATAL_CODE(XPAR_EXIT_NOPLAN, "Out of memory.");
  return p;
}

void * xpar_calloc(sz n, sz size) {
  if (n && size && n > (sz) -1 / size) FATAL_CODE(XPAR_EXIT_NOPLAN,
                              "Allocation size overflow.");
  { void * p = calloc(n ? n : 1, size ? size : 1);
    if (!p) FATAL_CODE(XPAR_EXIT_NOPLAN, "Out of memory.");
    return p; }
}

void * xpar_alloc_raw(sz n) {
  void * p = malloc(n ? n : 1);
  if (!p) FATAL_CODE(XPAR_EXIT_NOPLAN, "Out of memory.");
  return p;
}

void * xpar_realloc(void * p, sz n) {
  void * q = realloc(p, n ? n : 1);
  if (!q) FATAL_CODE(XPAR_EXIT_NOPLAN, "Out of memory.");
  return q;
}

void xpar_free(void * p) { free(p); }

void * xpar_alloc_aligned(sz n, sz align) {
  u8 * raw;
  uintptr_t a;
  sz pad;
  if (align < sizeof(void *)) align = sizeof(void *);
  if (!xpar_is_pow2(align)) FATAL("Alignment is not a power of two.");
  if (n == 0) n = 1;
  pad = align + sizeof(void *);
  if (n > (sz) -1 - pad) FATAL_CODE(XPAR_EXIT_NOPLAN,
                              "Allocation size overflow.");
  raw = malloc(n + pad);
  if (!raw) FATAL_CODE(XPAR_EXIT_NOPLAN, "Out of memory.");
  a = ((uintptr_t) raw + sizeof(void *) + align - 1) &
      ~(uintptr_t) (align - 1);
  ((void **) a)[-1] = raw;
  return (void *) a;
}

void xpar_free_aligned(void * p) {
  if (p) free(((void **) p)[-1]);
}

/*  Prefer the largest allocatable block; fall back to free pages.  */
u64 xpar_physical_memory(void) {
  __dpmi_free_mem_info info;
  if (__dpmi_get_free_memory_information(&info) != 0) return 0;
  if (info.largest_available_free_block_in_bytes != 0 &&
      info.largest_available_free_block_in_bytes != (unsigned long) -1L)
    return (u64) info.largest_available_free_block_in_bytes;
  if (info.total_number_of_free_pages == 0) return 0;
  return (u64) info.total_number_of_free_pages * 4096ULL;
}

bool xpar_is_rotational(const char * path) {
  /*  Almost certainly true on this host, and deliberately not claimed: the
      planner only uses the answer as a hint and a wrong "true" would push
      every plan towards a layout the user did not ask for.  */
  (void) path;
  return false;
}


void xpar_port_write_text(xpar_file * f, const char * s, sz n) {
  xpar_write(f, s, n);
}


void xpar_exit(int code) {
  exit(code);
}

/*  The codes DJGPP's int 21h wrappers actually produce. Anything else
    falls through to a number, which is more use than a wrong name.  */
static const struct { int n; const char * s; } err_tab[] = {
  { EACCES,       "permission denied"              },
  { EEXIST,       "file exists"                    },
  { ENOENT,       "no such file or directory"      },
  { EBADF,        "bad file descriptor"            },
  { EINVAL,       "invalid argument"               },
  { EIO,          "I/O error"                      },
  { ENOMEM,       "out of memory"                  },
  { ENOSPC,       "no space left on device"        },
  { ENFILE,       "too many open files in system"  },
  { EMFILE,       "too many open files"            },
  { ENOTDIR,      "not a directory"                },
  { EISDIR,       "is a directory"                 },
  { ENAMETOOLONG, "filename too long"              },
  { EOVERFLOW,    "value too large"                },
  { EXDEV,        "cross-device link"              }
};

static char err_gen[32];

const char * xpar_strerror(int err) {
  for (sz i = 0; i < ARRAY_LEN(err_tab); i++)
    if (err_tab[i].n == err) return err_tab[i].s;
  xpar_snprintf(err_gen, sizeof err_gen, "error %d", err);
  return err_gen;
}

int xpar_errno(void) { return errno; }

/*  Combine the BIOS tick with the 8254 counter for microsecond resolution.
    The OUT bit distinguishes the counter's two mode-3 half-cycles.  */
u64 xpar_usec_now(void) {
  u32 bios0, bios1, count, out_hi, elapsed;
  u64 pit;
  do {
    u8 status;
    bios0 = _farpeekl(_dos_ds, 0x46CUL);
    outportb(0x43, 0xC2);
    status = inportb(0x40);
    count  = (u32) inportb(0x40);
    count |= (u32) inportb(0x40) << 8;
    out_hi = (status >> 7) & 1;
    bios1  = _farpeekl(_dos_ds, 0x46CUL);
  } while (bios0 != bios1);   /*  retry if IRQ0 landed inside the read  */
  elapsed = (65536u - count) >> 1;
  if (!out_hi) elapsed += 32768u;
  pit = (u64) bios1 * 65536ULL + (u64) elapsed;
  return (pit * 1000000ULL) / 1193182ULL;
}

/*  Wall-clock precision is limited to the 55 ms BIOS tick.  */
i64 xpar_wall_ns(void) {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) != 0) return (i64) time(NULL) * 1000000000LL;
  return (i64) tv.tv_sec * 1000000000LL + (i64) tv.tv_usec * 1000LL;
}

/*  DOS lacks system entropy; callers use these bytes only for staging names.  */
void xpar_random_bytes(void * buf, sz n) {
  static u64 state;
  static u64 calls;
  u8 * p = (u8 *) buf;
  sz i;
  state ^= (u64) xpar_usec_now() + ((u64) xpar_wall_ns() << 16) + ++calls;
  if (!state) state = 1;
  for (i = 0; i < n; i++) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    p[i] = (u8) (state >> 56);   /*  Use the high bits of the LCG.  */
  }
}

int main(int argc, char ** argv) {
  xpar_host_init();
  xpar_crash_install();
  return xpar_main(argc, argv);
}

/*  Print a portable summary before DJGPP's default traceback.  */

#define XPAR_CRASH_FRAMES 32

static const char * crash_name(int sig) {
  switch (sig) {
    case SIGSEGV: return "invalid memory reference (SIGSEGV)";
    case SIGILL:  return "illegal instruction (SIGILL)";
    case SIGFPE:  return "arithmetic exception (SIGFPE)";
    case SIGABRT: return "aborted (SIGABRT)";
    default:      return "fatal signal";
  }
}

static void crash_handler(int sig) {
  void * frames[XPAR_CRASH_FRAMES];
  unsigned n, i;
  if (xpar_crash_entered()) xpar_exit(XPAR_EXIT_INTERNAL);
  n = xpar_crash_walk_fp((void * const *) __builtin_frame_address(0),
                         frames, XPAR_CRASH_FRAMES);
  xpar_crash_head(crash_name(sig), (u64) sig, 1, NULL, NULL, 0, NULL);
  for (i = 0; i < n; i++) xpar_crash_frame(i, frames[i], NULL);
  xpar_crash_tail(n != 0);
  signal(sig, SIG_DFL);
  raise(sig);
  xpar_exit(XPAR_EXIT_INTERNAL);
}

void xpar_crash_install(void) {
  if (!xpar_crash_wanted()) return;
  signal(SIGSEGV, crash_handler);
  signal(SIGILL,  crash_handler);
  signal(SIGFPE,  crash_handler);
  signal(SIGABRT, crash_handler);
}
