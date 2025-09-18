EXTRA_DIST = README.md
bin_PROGRAMS = xpar
noinst_HEADERS = platform.h crc32c.h jmode.h vmode.h lmode.h common.h yarg.h \
								 sharding.h sse2neon.h
xpar_SOURCES = platform.c xpar.c crc32c.c jmode.c vmode.c lmode.c

if XPAR_X86_64
xpar_SOURCES += xpar-x86_64.asm
SUFFIXES = .asm
.asm.o:
	$(NASM) $(NAFLAGS) -g -o $@ $<
endif

if XPAR_AARCH64
xpar_SOURCES += xpar-aarch64.S
xpar_CCASFLAGS = -march=armv8-a+crypto+crc
endif

dist_man_MANS = xpar.1

# Developer convenience targets
.PHONY: update-ChangeLog
update-ChangeLog:
	contrib/gitlog-to-changelog > ChangeLog

.PHONY: self-check
self-check:
	./xpar -Jef xpar && ./xpar -Jdf xpar.xpa xpar.org \
		&& cmp xpar xpar.org && rm xpar.org xpar.xpa
	./xpar -Jef -i 2 xpar && ./xpar -Jdf xpar.xpa xpar.org \
		&& cmp xpar xpar.org && rm xpar.org xpar.xpa
	./xpar -Wef --dshards=4 --pshards=2 xpar \
	  && ./xpar -Wdf xpar.org xpar.xpa.* \
		&& cmp xpar xpar.org
	rm xpar.org xpar.xpa*
	./xpar -Wef --dshards=4 --pshards=2 xpar && rm -f xpar.xpa.000 \
	  && ./xpar -Wdf xpar.org xpar.xpa.* \
		&& cmp xpar xpar.org
	rm xpar.org xpar.xpa*
	./xpar -Lef --dshards=4 --pshards=2 xpar \
	  && ./xpar -Ldf xpar.org xpar.xpa.* \
		&& cmp xpar xpar.org
	rm xpar.org xpar.xpa*
	./xpar -Lef --dshards=4 --pshards=2 xpar && rm -f xpar.xpa.000 \
	  && ./xpar -Ldf xpar.org xpar.xpa.* \
		&& cmp xpar xpar.org
	rm xpar.org xpar.xpa*
	./xpar -Lef --dshards=12 --pshards=8 xpar && rm -f xpar.xpa.000 \
	  && ./xpar -Ldf xpar.org xpar.xpa.* \
		&& cmp xpar xpar.org
	rm xpar.org xpar.xpa*
	
.PHONY: cloc
cloc:
	cloc $(xpar_SOURCES) Makefile.am configure.ac xpar-aarch64.S \
		xpar-x86_64.asm