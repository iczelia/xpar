/*
   Copyright (C) 2022-2024 Kamila Szewczyk

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "platform.h"

#if !defined(HAVE_ASPRINTF)
int asprintf(char **strp, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int len = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (len < 0) return -1;
  *strp = malloc(len + 1);
  if (!*strp) return -1;
  va_start(ap, fmt);
  len = vsnprintf(*strp, len + 1, fmt, ap);
  va_end(ap);
  return len;
}
#endif

#if !defined(HAVE_STRNDUP)
  char * strndup(const char *s, sz n) {
    char * copy = malloc(n + 1);
    if (!copy) return NULL;
    memcpy(copy, s, n);
    copy[n] = '\0';
    return copy;
  }
#endif

#if defined(HAVE_MMAP)
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

#if defined(HAVE_CREATEFILEMAPPINGA)
  #include <windows.h>
#endif

#if defined(HAVE_MMAP) && defined(HAVE_CREATEFILEMAPPINGA)
  // Bummer! Which one to choose? On Windows we will prefer the native
  // function, and its presence clearly signifies that we are on Windows.
  #undef HAVE_MMAP
#endif

#if defined(HAVE_MMAP) || defined(HAVE_CREATEFILEMAPPINGA)
mmap_t xpar_map(const char * filename) {
  mmap_t mappedFile = { NULL, 0 };
#if defined(HAVE_CREATEFILEMAPPINGA)
  HANDLE fileHandle = CreateFileA(
    filename, GENERIC_READ, FILE_SHARE_READ, NULL,
    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL
  );
  if (fileHandle == INVALID_HANDLE_VALUE) return mappedFile;
  LARGE_INTEGER fileSize;
  if (!GetFileSizeEx(fileHandle, &fileSize)) {
    CloseHandle(fileHandle); return mappedFile;
  }
  HANDLE fileMapping = CreateFileMappingA(
    fileHandle, NULL, PAGE_READONLY,
    fileSize.HighPart, fileSize.LowPart, NULL
  );
  if (!fileMapping) {
    CloseHandle(fileHandle); return mappedFile;
  }
  mappedFile.map = MapViewOfFile(fileMapping, FILE_MAP_READ, 0, 0, 0);
  if (!mappedFile.map) {
    CloseHandle(fileMapping); CloseHandle(fileHandle); return mappedFile;
  }
  mappedFile.size = (size_t) fileSize.QuadPart;
  CloseHandle(fileMapping);
  CloseHandle(fileHandle);
#else
  int fd = open(filename, O_RDONLY);
  if (fd == -1) return mappedFile;
  struct stat st;
  if (fstat(fd, &st) == -1) {
    close(fd); return mappedFile;
  }
  mappedFile.size = (size_t) st.st_size;
  mappedFile.map = mmap(NULL, mappedFile.size, PROT_READ, MAP_SHARED, fd, 0);
  if (mappedFile.map == MAP_FAILED) {
    close(fd); return mappedFile;
  }
  close(fd);
  // Reserve a single page at the end of the allocation

#endif
  return mappedFile;
}

void xpar_unmap(mmap_t * file) {
  #if defined(HAVE_MMAP)
    if (file->map) munmap(file->map, file->size);
  #else
    if (file->map) UnmapViewOfFile(file->map);
  #endif
  file->map = NULL; file->size = 0;
}
#endif

#if defined(HAVE_IO_H)
  #include <io.h>
  #include <fcntl.h>
#endif

void platform_init(void) {
  #if defined(HAVE__SETMODE)
    _setmode(STDIN_FILENO, O_BINARY);
    _setmode(STDOUT_FILENO, O_BINARY);
  #endif
}

#include <errno.h>
void xfclose(FILE * des) {
  if(fflush(des)) FATAL_PERROR("fflush");
#if defined(HAVE__COMMIT)
  if (_commit(fileno(des))) FATAL_PERROR("commit");
#elif defined(HAVE_FSYNC)
  // EINVAL means that we tried to fsync something that can't be synced.
  if (fsync(fileno(des))) {
    if (errno != EINVAL) FATAL_PERROR("fsync");
    errno = 0;
  }
#endif
  if (fclose(des) == EOF) FATAL_PERROR("fclose");
}
void xfwrite(const void * ptr, sz size, FILE * stream) {
  if (fwrite(ptr, 1, size, stream) != size) FATAL_PERROR("fwrite");
  if (ferror(stream)) FATAL_PERROR("fwrite");
}
sz xfread(void * ptr, sz size, FILE * stream) {
  sz n = fread(ptr, 1, size, stream);
  if (ferror(stream)) FATAL_PERROR("fread");
  return n;
}
void * xmalloc(sz size) {
  void * ptr = calloc(size, 1);
  if (!ptr) FATAL_PERROR("malloc");
  return ptr;
}
void notty(FILE * des) {
  #if defined(HAVE_ISATTY)
  if (isatty(fileno(des)))
    FATAL("Refusing to read/write binary data from/to a terminal.");
  errno = 0; // isatty sets errno to weird values on some platforms.
  #endif
}
bool is_seekable(FILE * des) {
  return fseek(des, 0, SEEK_CUR) != -1;
}