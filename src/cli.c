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

/*  Argument parsing, validation, and help.  */

#include "cli.h"
#include "manifest.h"
#include "pathname.h"
#include "slice.h"
#include "v1detect.h"
#include "volname.h"
#include "port-cpu.h"
#include "port-fs.h"

#if defined(__GNUC__) && !defined(__clang__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpragmas"
  #pragma GCC diagnostic ignored "-Wcalloc-transposed-args"
#endif
#include "yarg.h"
#if defined(__GNUC__) && !defined(__clang__)
  #pragma GCC diagnostic pop
#endif

static char * dup_str(const char * s) {
  char * p = xpar_strdup(s);
  FATAL_UNLESS_CODE(XPAR_EXIT_NOPLAN, "Out of memory.", p != NULL);
  return p;
}

static void push_str(char *** v, u32 * n, const char * s) {
  char ** nv = (char **) xpar_realloc(*v, (*n + 1) * sizeof(char *));
  FATAL_UNLESS_CODE(XPAR_EXIT_NOPLAN, "Out of memory.", nv != NULL);
  nv[*n] = dup_str(s);  *v = nv;  (*n)++;
}

static bool is_dir(const char * p);

static void need_dir(const char * nm, const char * p) {
  FATAL_UNLESS("Option %s needs an existing directory, and '%s' is not "
               "one.", is_dir(p), nm, p);
}

static void free_strv(char ** v, u32 n) {
  for (u32 i = 0; i < n; i++) xpar_free(v[i]);
  xpar_free(v);
}

static bool all_digits(const char * s) {
  if (!s || !*s) return false;
  while (*s) { if (*s < '0' || *s > '9') return false;  s++; }
  return true;
}

static int lower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

static bool ci_equal(const char * a, const char * b) {
  while (*a && lower((u8) *a) == lower((u8) *b)) { a++;  b++; }
  return !*a && !*b;
}

static int scan_decimal(const char * s, u64 * ip, u64 * fnum, u64 * fden,
                        const char ** end) {
  u64 v = 0, num = 0, den = 1;  int digits = 0;
  while (*s >= '0' && *s <= '9') {
    if (v > (UINT64_MAX - (u64) (*s - '0')) / 10) return -1;
    v = v * 10 + (u64) (*s - '0');  s++;  digits++;
  }
  if (!digits) return -1;
  if (*s == '.') {
    int fd = 0;
    s++;
    while (*s >= '0' && *s <= '9') {
      /*  Stop accumulating well short of overflow; the digits past the
          eighteenth cannot change a byte count.  */
      if (den <= 100000000000000000ull) {
        num = num * 10 + (u64) (*s - '0');  den *= 10;
      }
      s++;  fd++;
    }
    if (!fd) return -1;
  }
  *ip = v;  *fnum = num;  *fden = den;  *end = s;
  return 0;
}

/*  K, M, G, T are powers of 1024 and KB, MB, GB, TB powers of 1000
    Case is not significant, and the suffix must be the whole
    remainder: `1K%` is a malformed size, not a kilobyte.  */
static int size_mult(const char * s, u64 * mult) {
  static const struct { const char * suf;  u64 mult; } tab[] = {
    { "",   1                            },
    { "k",  1024ull                      }, { "kb", 1000ull          },
    { "m",  1024ull * 1024               }, { "mb", 1000000ull       },
    { "g",  1024ull * 1024 * 1024        }, { "gb", 1000000000ull    },
    { "t",  1024ull * 1024 * 1024 * 1024 }, { "tb", 1000000000000ull }
  };
  for (sz i = 0; i < ARRAY_LEN(tab); i++)
    if (ci_equal(s, tab[i].suf)) { *mult = tab[i].mult;  return 0; }
  return -1;
}

int xpar_cli_parse_size(const char * s, u64 * out) {
  u64 ip = 0, num = 0, den = 1, mult = 1, base, frac;
  const char * end = "";
  if (!s || scan_decimal(s, &ip, &num, &den, &end)) return -1;
  if (size_mult(end, &mult)) return -1;
  /*  A fraction with no unit is a typo rather than a request for a byte
      and a half: `-m 1.5` is somebody who meant 1.5G.  */
  if (den > 1 && mult == 1) return -1;
  if (ip > UINT64_MAX / mult) return -1;
  base = ip * mult;
  frac = (u64) ((f64) num * (f64) mult / (f64) den);
  if (base > UINT64_MAX - frac) return -1;
  *out = base + frac;
  return 0;
}

int xpar_cli_parse_recovery(const char * s, xpar_rspec * out) {
  u64 ip = 0, num = 0, den = 1, mult = 1;
  const char * end = "";
  xpar_rspec r;
  r.kind = XPAR_R_NONE;  r.count = 0;  r.factor = 0.0;
  if (!s || scan_decimal(s, &ip, &num, &den, &end)) return -1;
  if (end[0] == '%' && !end[1]) {
    r.kind = XPAR_R_PERCENT;  r.factor = (f64) ip + (f64) num / (f64) den;
  } else if ((end[0] == 'x' || end[0] == 'X') && !end[1]) {
    r.kind = XPAR_R_TIMES;    r.factor = (f64) ip + (f64) num / (f64) den;
  } else if (!end[0]) {
    /*  A bare number is a slice count, so a fraction of one is not a
        smaller request but a mistake: `-r 1.5` names no whole slice.  */
    if (den > 1) return -1;
    r.kind = XPAR_R_COUNT;    r.count = ip;
  } else {
    if (size_mult(end, &mult) || xpar_cli_parse_size(s, &r.count)) return -1;
    r.kind = XPAR_R_BYTES;
  }
  /* Zero requests no recovery; positive fractions round up later. */
  if ((r.kind == XPAR_R_PERCENT || r.kind == XPAR_R_TIMES) && r.factor < 0.0)
    return -1;
  if ((r.kind == XPAR_R_PERCENT && r.factor > 6553600.0) ||
      (r.kind == XPAR_R_TIMES && r.factor > 65536.0)) return -1;
  *out = r;
  return 0;
}

static u64 need_u64(const char * nm, const char * v) {
  u64 ip = 0, num = 0, den = 1;  const char * end = "";
  FATAL_UNLESS("Option %s expects a whole number.",
               v && !scan_decimal(v, &ip, &num, &den, &end) && !*end &&
               den == 1, nm);
  return ip;
}

static u64 need_range(const char * nm, const char * v, u64 lo, u64 hi) {
  u64 n = need_u64(nm, v);
  FATAL_UNLESS("Option %s expects a number between %" PRIu64 " and %" PRIu64
               ".",
               n >= lo && n <= hi, nm, lo,
               hi);
  return n;
}

static u64 need_size(const char * nm, const char * v) {
  u64 out = 0;
  FATAL_UNLESS("Option %s expects a size such as 4096, 64K or 2.5MB.",
               v && !xpar_cli_parse_size(v, &out), nm);
  return out;
}

static int need_word(const char * nm, const char * v,
                     const char * const * words) {
  int i;
  if (v) for (i = 0; words[i]; i++) if (!xpar_strcmp(v, words[i])) return i;
  xpar_fprintf(xpar_stderr, "xpar: Option %s does not accept '%s'.\n",
               nm, v ? v : "");
  xpar_fputs("xpar: it takes one of:", xpar_stderr);
  for (i = 0; words[i]; i++) xpar_fprintf(xpar_stderr, " %s", words[i]);
  xpar_fputs("\n", xpar_stderr);
  xpar_exit(XPAR_EXIT_USAGE);
  return 0;
}

static const char * const w_layout[] = { "sidecar", "split", "armoured", NULL };
static const char * const w_codec[]  = { "auto", "fft", "matrix", NULL };
static const char * const w_field[]  = { "auto", "8", "16", NULL };
static const char * const w_align[]  = { "none", "slice", "1k", NULL };
static const char * const w_tag[]    = { "none", "8", "16", NULL };
static const char * const w_armour[] = { "none", "metadata", "all", NULL };
static const char * const w_dedup[]  = { "none", "file", "chunk", NULL };
static const char * const w_color[]  = { "auto", "always", "never", NULL };
static const char * const w_resync[] = { "off", "auto", "always", NULL };
static const char * const w_rescan[] = { "stat", "hash", "none", NULL };
static const char * const w_owner[]  = { "name", "numeric", NULL };
static const char * const w_scope[]  = { "generation", "chain", NULL };

static int word_field(const char * nm, const char * v) {
  int i = need_word(nm, v, w_field);
  return i == 0 ? XPAR_CLI_AUTO : (i == 1 ? 8 : 16);
}

/*  --preserve and --require  */
static const struct { const char * name;  u32 bits, lit; } pres_tokens[] = {
  { "mtime",     XPAR_PRES_MTIME,     0                   },
  { "atime",     XPAR_PRES_ATIME,     0                   },
  { "ctime",     XPAR_PRES_CTIME,     XPAR_PRES_CTIME     },
  { "btime",     XPAR_PRES_BTIME,     0                   },
  { "times",     XPAR_PRES_TIMES,     0                   },
  { "mode",      XPAR_PRES_MODE,      0                   },
  { "setid",     XPAR_PRES_SETID,     XPAR_PRES_SETID     },
  { "attrs",     XPAR_PRES_ATTRS,     0                   },
  { "owner",     XPAR_PRES_OWNER,     0                   },
  { "xattr",     XPAR_PRES_XATTR,     0                   },
  { "xattr-all", XPAR_PRES_XATTR | XPAR_PRES_XATTR_ALL,
                                      XPAR_PRES_XATTR_ALL },
  { "links",     XPAR_PRES_LINKS,     0                   },
  { "all",       XPAR_PRES_ALL,       0                   },
  { "none",      0,                   0                   }
};

static void bad_token(const char * nm, const char * tok) {
  xpar_fprintf(xpar_stderr, "xpar: Option %s does not accept the token "
               "'%s'.\n", nm, tok);
  xpar_fputs("xpar: tokens: mtime atime ctime btime times mode setid "
             "attrs owner\n"
             "xpar:         xattr xattr-all links all none\n", xpar_stderr);
  xpar_exit(XPAR_EXIT_USAGE);
}

static u32 parse_pres(const char * nm, const char * v, u32 dflt, u32 * lit,
                      u32 * named) {
  u32 set = dflt;
  bool replaced = false;
  const char * p = v;
  FATAL_UNLESS("Option %s expects a comma-separated token list.", v && *v, nm);
  while (*p) {
    char tok[24];
    int sign = 0, n = 0;
    sz i;
    if (*p == '+' || *p == '-') { sign = *p;  p++; }
    while (*p && *p != ',') {
      if (n < (int) sizeof tok - 1) tok[n++] = *p;
      p++;
    }
    tok[n] = '\0';
    if (*p == ',') p++;
    for (i = 0; i < ARRAY_LEN(pres_tokens); i++)
      if (!xpar_strcmp(tok, pres_tokens[i].name)) break;
    if (i == ARRAY_LEN(pres_tokens)) bad_token(nm, tok);
    if (sign == '-') { set &= ~pres_tokens[i].bits;  continue; }
    if (!sign && !replaced) { set = 0;  replaced = true; }
    set |= pres_tokens[i].bits;
    if (lit) *lit |= pres_tokens[i].lit;
    if (named) *named |= pres_tokens[i].bits;
  }
  return set;
}

static void parse_genref(const char * nm, const char * v, xpar_genref * g) {
  FATAL_UNLESS("Option %s expects a generation number or a set id.",
               v && *v, nm);
  g->by_id = false;  g->number = 0;  g->id_prefix = NULL;
  /*  Reject generation selectors that would narrow to u32.  */
  if (all_digits(v)) {
    g->number = need_range(nm, v, 0, 0xFFFFFFFFu);
    return;
  }
  for (const char * p = v; *p; p++) {
    int c = lower((u8) *p);
    FATAL_UNLESS("Option %s expects a generation number or hexadecimal "
                 "set-id prefix, not '%s'.",
                 (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'), nm, v);
  }
  FATAL_UNLESS("Option %s takes at most 32 hexadecimal digits: a set id "
               "is 16 bytes.", xpar_strlen(v) <= 32, nm);
  g->by_id = true;
  g->id_prefix = dup_str(v);
  for (char * q = g->id_prefix; *q; q++) *q = (char) lower((u8) *q);
}

enum {
  O_JSON = 256, O_PROGRESS, O_NO_PROGRESS, O_COLOR, O_REPRODUCIBLE,
  O_SIMD,
  O_MIN_RECOVERY, O_MAX_RECOVERY, O_LAYOUT, O_CODEC, O_FIELD, O_ALIGN,
  O_CELL,
  O_SLICE_TAG, O_ARMOUR, O_ARMOUR_FIELD, O_ARMOUR_T, O_ARMOUR_PCT,
  O_BURST, O_DEPTH, O_VOLUMES, O_DEDUP, O_DEDUP_CHUNK, O_DEDUP_MEMORY,
  O_DEDUP_MAX_REFS, O_PRESERVE, O_EXCLUDE, O_INCLUDE, O_FOLLOW, O_BASE,
  O_LABELS, O_AUTH_KEY, O_AUTH_ONLY, O_NO_VERIFY_AFTER, O_SPOOL,
  O_SPOOL_DIR, O_STDIN_NAME,
  O_FAST, O_STRONG, O_RESYNC, O_RESYNC_STEP, O_RESYNC_WINDOW, O_RESYNC_EXH,
  O_SCAN, O_GENERATION, O_CHAIN, O_IN_PLACE, O_TO, O_BACKUP, O_PARANOID,
  O_KEEP_JOURNAL, O_NO_JOURNAL, O_REPLACE_JOURNAL, O_DRY_RUN,
  O_EXIT_ON_CHANGE, O_DEEP,
  O_REWRITE, O_REBUILD_CELLS, O_STDOUT, O_OWNER_MAP, O_REQUIRE, O_STRICT_NAMES,
  O_VOLUME, O_RESCAN, O_VERIFY_UNCHANGED, O_ALLOW_MISSING, O_DEDUP_SCOPE,
  O_REPLACE, O_BEFORE, O_DEPS, O_LINKS, O_LIST_DEDUP, O_TIERS
};

static const yarg_options t_global[] = {
  { 'v',            no_argument,       "verbose"      },
  { 'q',            no_argument,       "quiet"        },
  { 'f',            no_argument,       "force"        },
  { 'j',            required_argument, "jobs"         },
  { 'm',            required_argument, "memory"       },
  { O_JSON,         no_argument,       "json"         },
  { O_PROGRESS,     no_argument,       "progress"     },
  { O_NO_PROGRESS,  no_argument,       "no-progress"  },
  { O_COLOR,        required_argument, "color"        },
  { O_REPRODUCIBLE, no_argument,       "reproducible" },
  { O_SIMD,         required_argument, "simd"         },
  { O_AUTH_KEY,     required_argument, "auth-key"     },
  { 'h',            no_argument,       "help"         },
  { 'V',            no_argument,       "version"      },
  { 0,              no_argument,       NULL           }
};

static const yarg_options t_create[] = {
  { 'o',               required_argument, "output"          },
  { 'r',               required_argument, "recovery"        },
  { O_MIN_RECOVERY,    required_argument, "min-recovery"    },
  { O_MAX_RECOVERY,    required_argument, "max-recovery"    },
  { 's',               required_argument, "slice-size"      },
  { 'b',               required_argument, "slices"          },
  { O_CELL,            required_argument, "cell"            },
  { O_LAYOUT,          required_argument, "layout"          },
  { O_CODEC,           required_argument, "codec"           },
  { O_FIELD,           required_argument, "field"           },
  { O_ALIGN,           required_argument, "align"           },
  { O_SLICE_TAG,       required_argument, "slice-tag"       },
  { O_ARMOUR,          required_argument, "armour"          },
  { O_ARMOUR_FIELD,    required_argument, "armour-field"    },
  { O_ARMOUR_T,        required_argument, "armour-t"        },
  { O_ARMOUR_PCT,      required_argument, "armour-pct"      },
  { O_BURST,           required_argument, "burst"           },
  { O_DEPTH,           required_argument, "depth"           },
  { O_VOLUMES,         required_argument, "volumes"         },
  { O_DEDUP,           required_argument, "dedup"           },
  { O_DEDUP_CHUNK,     required_argument, "dedup-chunk"     },
  { O_DEDUP_MEMORY,    required_argument, "dedup-memory"    },
  { O_DEDUP_MAX_REFS,  required_argument, "dedup-max-refs"  },
  { O_PRESERVE,        required_argument, "preserve"        },
  { 'R',               no_argument,       "recurse"         },
  { O_EXCLUDE,         required_argument, "exclude"         },
  { O_INCLUDE,         required_argument, "include"         },
  { O_FOLLOW,          no_argument,       "follow-symlinks" },
  { O_BASE,            required_argument, "base"            },
  { O_LABELS,          no_argument,       "labels"          },
  { O_AUTH_ONLY,       no_argument,       "auth-only"       },
  { O_NO_VERIFY_AFTER, no_argument,       "no-verify-after" },
  { O_SPOOL,           no_argument,       "spool"           },
  { O_SPOOL_DIR,       required_argument, "spool-dir"       },
  { O_STDIN_NAME,      required_argument, "stdin-name"      },
  { 0,                 no_argument,       NULL              }
};

static const yarg_options t_verify[] = {
  { O_FAST,          no_argument,       "fast"              },
  { O_STRONG,        no_argument,       "strong"            },
  { O_RESYNC,        required_argument, "resync"            },
  { O_RESYNC_STEP,   required_argument, "resync-step"       },
  { O_RESYNC_WINDOW, required_argument, "resync-window"     },
  { O_RESYNC_EXH,    no_argument,       "resync-exhaustive" },
  { O_SCAN,          required_argument, "scan"              },
  { O_GENERATION,    required_argument, "generation"        },
  { O_CHAIN,         no_argument,       "chain"             },
  { 0,               no_argument,       NULL                }
};

static const yarg_options t_repair[] = {
  { O_IN_PLACE,       no_argument,       "in-place"          },
  { O_TO,             required_argument, "to"                },
  { O_BACKUP,         no_argument,       "backup"            },
  { O_PARANOID,       no_argument,       "paranoid"          },
  { O_KEEP_JOURNAL,   no_argument,       "keep-journal"      },
  { O_NO_JOURNAL,     no_argument,       "no-journal"        },
  { O_REPLACE_JOURNAL, no_argument,      "replace-journal"   },
  { O_DRY_RUN,        no_argument,       "dry-run"           },
  { O_EXIT_ON_CHANGE, no_argument,       "exit-on-change"    },
  /*  Repair runs the same resynchronising pass and prints the same advice
      to widen it, so it has to take the options that advice names.  */
  { O_RESYNC,         required_argument, "resync"            },
  { O_RESYNC_STEP,    required_argument, "resync-step"       },
  { O_RESYNC_WINDOW,  required_argument, "resync-window"     },
  { O_RESYNC_EXH,     no_argument,       "resync-exhaustive" },
  { O_GENERATION,     required_argument, "generation"        },
  { O_CHAIN,          no_argument,       "chain"             },
  { 0,                no_argument,       NULL                }
};

static const yarg_options t_scrub[] = {
  { O_DEEP,          no_argument,       "deep"          },
  { O_REWRITE,       no_argument,       "rewrite"       },
  { O_REBUILD_CELLS, no_argument,       "rebuild-cells" },
  { O_GENERATION,    required_argument, "generation"    },
  { O_CHAIN,         no_argument,       "chain"         },
  { 0,               no_argument,       NULL            }
};

static const yarg_options t_extract[] = {
  { O_TO,         required_argument, "to"         },
  { O_STDOUT,     no_argument,       "stdout"     },
  { O_PRESERVE,   required_argument, "preserve"   },
  { O_OWNER_MAP,  required_argument, "owner-map"  },
  { O_REQUIRE,    required_argument, "require"    },
  { O_STRICT_NAMES, no_argument,     "strict-names" },
  { O_GENERATION, required_argument, "generation" },
  { 0,            no_argument,       NULL         }
};

static const yarg_options t_recover[] = {
  { O_VOLUME,     required_argument, "volume"     },
  { O_TO,         required_argument, "to"         },
  { O_GENERATION, required_argument, "generation" },
  { 0,            no_argument,       NULL         }
};

static const yarg_options t_addrecovery[] = {
  { 'r',          required_argument, "recovery"   },
  { O_GENERATION, required_argument, "generation" },
  { 0,            no_argument,       NULL         }
};

static const yarg_options t_add[] = {
  { O_RESCAN,           required_argument, "rescan"           },
  { O_VERIFY_UNCHANGED, no_argument,       "verify-unchanged" },
  { O_ALLOW_MISSING,    no_argument,       "allow-missing"    },
  { O_DEDUP_SCOPE,      required_argument, "dedup-scope"      },
  { 0,                  no_argument,       NULL               }
};

static const yarg_options t_consolidate[] = {
  { O_REPLACE, no_argument, "replace" },
  { O_DRY_RUN, no_argument, "dry-run" },
  { 0,         no_argument, NULL      }
};

static const yarg_options t_prune[] = {
  { O_BEFORE,     required_argument, "before"     },
  { O_GENERATION, required_argument, "generation" },
  { O_DRY_RUN,    no_argument,       "dry-run"    },
  { 0,            no_argument,       NULL         }
};

static const yarg_options t_list[] = {
  { O_GENERATION, required_argument, "generation" },
  { O_CHAIN,      no_argument,       "chain"      },
  { O_LINKS,      no_argument,       "links"      },
  { O_LIST_DEDUP, no_argument,       "dedup"      },
  { 0,            no_argument,       NULL         }
};

static const yarg_options t_info[] = {
  { O_DEPS,       no_argument,       "deps"       },
  { O_GENERATION, required_argument, "generation" },
  { O_CHAIN,      no_argument,       "chain"      },
  { 0,            no_argument,       NULL         }
};

static const yarg_options t_explain[] = {
  { O_GENERATION, required_argument, "generation" },
  { 0,            no_argument,       NULL         }
};

static const yarg_options t_none[] = {
  { 0, no_argument, NULL }
};

static const yarg_options t_benchmark[] = {
  { O_TIERS, no_argument, "tiers" },
  { 0,       no_argument, NULL    }
};

/*  `add` and `consolidate` inherit the whole of create's table  */
static const yarg_verb verbs[] = {
  { XPAR_VERB_CREATE,      "create",      t_create,      NULL     },
  { XPAR_VERB_VERIFY,      "verify",      t_verify,      NULL     },
  { XPAR_VERB_REPAIR,      "repair",      t_repair,      NULL     },
  { XPAR_VERB_SCRUB,       "scrub",       t_scrub,       NULL     },
  { XPAR_VERB_EXTRACT,     "extract",     t_extract,     NULL     },
  { XPAR_VERB_RECOVER,     "recover",     t_recover,     NULL     },
  { XPAR_VERB_ADDRECOVERY, "addrecovery", t_addrecovery, NULL     },
  { XPAR_VERB_ADD,         "add",         t_add,         t_create },
  { XPAR_VERB_CONSOLIDATE, "consolidate", t_consolidate, t_create },
  { XPAR_VERB_PRUNE,       "prune",       t_prune,       NULL     },
  { XPAR_VERB_LIST,        "list",        t_list,        NULL     },
  { XPAR_VERB_INFO,        "info",        t_info,        NULL     },
  { XPAR_VERB_EXPLAIN,     "explain",     t_explain,     NULL     },
  { XPAR_VERB_UNDO,        "undo",        t_none,        NULL     },
  { XPAR_VERB_RECOVER_PROLOGUE, "recover-prologue", t_none, NULL  },
  { XPAR_VERB_BENCHMARK,   "benchmark",   t_benchmark,   NULL     },
  { 0,                     NULL,          NULL,          NULL     }
};

static const char * const verb_desc[] = {
  NULL,
  "Build a protection set from the given input paths",
  "Check a set; writes nothing",
  "Check a set and repair what the recovery data covers",
  "Full algebraic pass over a set",
  "Reconstruct entries from a split or armoured set",
  "Regenerate one volume of a set",
  "Add recovery slices to an existing generation",
  "Append a generation protecting new or changed files",
  "Collapse a chain of generations into one",
  "Remove older generations from a chain",
  "Print the manifest",
  "Print geometry, codec, plan and redundancy",
  "Print the hand-recovery recipe",
  "Replay an in-place repair journal",
  "Brute-force a destroyed armoured prologue",
  "Measure the low-level SIMD kernels"
};

const char * xpar_verb_name(xpar_verb v) {
  for (int i = 0; verbs[i].name; i++)
    if (verbs[i].verb == (int) v) return verbs[i].name;
  return "xpar";
}

/*  Help.  */

void xpar_cli_version(void) {
  const xpar_cpu_tier * best = xpar_cpu_tier_at(xpar_cpu_tier_best());
  xpar_fprintf(xpar_stdout, "xpar %s (%s, %s)\n", PACKAGE_VERSION,
               XPAR_HOST_TRIPLE, best ? best->name : "scalar");
  xpar_fputs(
    "Copyright (C) 2022-2026 Kamila Szewczyk.\n"
    "License GPLv3: GNU GPL version 3 only"
    " <https://gnu.org/licenses/gpl-3.0.html>.\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.\n", xpar_stdout);
}

static const char help_global[] =
  "Global options (after the verb):\n"
  "  -v, --verbose        Repeatable; -vv reports per-slice detail\n"
  "  -q, --quiet          Diagnostics only\n"
  "  -f, --force          Overwrite, and permit the guarded operations\n"
  "  -j, --jobs=N         Worker threads (default: one per core)\n"
  "  -m, --memory=SIZE    Working-set ceiling for the planner\n"
  "      --json           JSON Lines on stdout, human output on stderr\n"
  "      --progress       Progress on stderr, at most one line a second\n"
  "      --no-progress    Never report progress\n"
  "      --color=WHEN     auto (default), always, never\n"
  "      --reproducible   Leave the clock and the host out of the set\n"
  "      --simd=TIER      Force a SIMD tier; 'auto' is the default\n"
  "      --auth-key=FILE  Authenticate a set with the key in FILE\n"
  "  -h, --help           This message, or a verb's own options\n"
  "  -V, --version        Print the version and exit\n";

static const char help_preserve[] =
  "\n"
  "  --preserve tokens: mtime atime ctime btime times mode setid attrs\n"
  "  owner xattr xattr-all links all none.  A bare list replaces the\n"
  "  default (mtime,mode,attrs,links), +token adds and -token removes.\n";

/*  Indexed by xpar_verb. The usage line is separate so that every entry
    can be a plain option block.  */
static const char * const verb_usage[] = {
  NULL,
  "xpar create [options] <path>...",
  "xpar verify [options] <set>",
  "xpar repair [options] <set>",
  "xpar scrub [options] <set>",
  "xpar extract [options] <set>",
  "xpar recover [options] <set>",
  "xpar addrecovery [options] <set>",
  "xpar add [options] <set> <path>...",
  "xpar consolidate [options] <set>",
  "xpar prune [options] <set>",
  "xpar list [options] <set>",
  "xpar info [options] <set>",
  "xpar explain [options] <set>",
  "xpar undo [options] <set>",
  "xpar recover-prologue [options] <file>",
  "xpar benchmark [options]"
};

static const char * const verb_opts[] = {
  NULL,
  /*  create  */
  "  -o, --output=BASE          Base name for the output volumes\n"
  "  -r, --recovery=SPEC        count | N% | SIZE | Nx\n"
  "      --min-recovery=N       Floor for the derived slice count\n"
  "      --max-recovery=SPEC    Recovery-axis space held for top-ups\n"
  "  -s, --slice-size=SIZE      Slice size Z; excludes -b\n"
  "  -b, --slices=N             Slice count S; excludes -s\n"
  "      --cell=SIZE            Cell size Y, the erasure unit\n"
  "      --layout=WHICH         sidecar (default), split, armoured\n"
  "      --codec=WHICH          auto (default), fft, matrix\n"
  "      --field=W              auto (default), 8, 16\n"
  "      --align=WHICH          none (default), slice, 1k\n"
  "      --slice-tag=W          none, 8 (default), 16\n"
  "      --armour=WHICH         none, metadata, all (default)\n"
  "      --armour-field=W       auto (default), 8, 16\n"
  "      --armour-t=N           Symbols corrected per inner codeword\n"
  "      --armour-pct=P         Inner-code overhead, 0 < P <= 50\n"
  "      --burst=SIZE           Burst to tolerate; excludes --depth\n"
  "      --depth=D              Interleave depth (default 1)\n"
  "      --volumes=WHICH        ladder (default), equal, or a count\n"
  "      --dedup=WHICH          none, file (default), chunk\n"
  "      --dedup-chunk=SIZE     Chunk size for --dedup=chunk\n"
  "      --dedup-memory=SIZE    Ceiling for the dedup index\n"
  "      --dedup-max-refs=N     Cap on aliases of one blob\n"
  "      --preserve=LIST        Metadata to record (see below)\n"
  "  -R, --recurse              Descend into directories\n"
  "      --exclude=GLOB         Repeatable\n"
  "      --include=GLOB         Repeatable; restricts to matches\n"
  "      --follow-symlinks      Store the target, not the link\n"
  "      --base=DIR             Store names relative to DIR\n"
  "      --labels               Write a label file per volume\n"
  "      --auth-only            Omit public CRC and whole-file hashes\n"
  "      --no-verify-after      Skip the read-back pass\n"
  "      --spool                Buffer a pipe to a file first\n"
  "      --spool-dir=DIR        Buffer it under DIR; implies --spool\n"
  "      --stdin-name=PATH      Manifest path for a lone '-' input\n",
  /*  verify  */
  "      --fast                 Tags only; skip the algebraic pass\n"
  "      --strong               Check slice tags, not just slice CRCs\n"
  "      --resync=WHICH         off, auto (default), always\n"
  "      --resync-step=N        Sample every Nth offset\n"
  "      --resync-window=SIZE   Displacement searched either way\n"
  "      --resync-exhaustive    Confirm every candidate; expensive\n"
  "      --scan=DIR             Also look for volumes in DIR\n"
  "      --generation=G         Number, or a set-id prefix\n"
  "      --chain                Every generation, oldest first\n",
  /*  repair  */
  "      --in-place             Rewrite the originals\n"
  "      --to=DIR               Write the repaired tree into DIR\n"
  "      --backup               Rewrite, keeping the damaged original\n"
  "      --paranoid             Re-read and re-verify every write\n"
  "      --keep-journal         Keep the undo journal on success\n"
  "      --no-journal           Do not write one; excludes the above\n"
  "      --replace-journal      Overwrite a journal an earlier repair left\n"
  "      --dry-run              Report what would change\n"
  "      --exit-on-change       Exit 1 when anything was repaired\n"
  "      --resync=WHICH         off, auto (default), always\n"
  "      --resync-step=N        Sample every Nth offset\n"
  "      --resync-window=SIZE   Displacement searched either way\n"
  "      --resync-exhaustive    Confirm every candidate; expensive\n"
  "      --generation=G         Number, or a set-id prefix\n"
  "      --chain                Every generation, oldest first\n",
  /*  scrub  */
  "      --deep                 Re-encode recovery and compare it\n"
  "      --rewrite              Write back what the pass corrected\n"
  "      --rebuild-cells        Rebuild the cell CRC table\n"
  "      --generation=G         Number, or a set-id prefix\n"
  "      --chain                Every generation, oldest first\n",
  /*  extract  */
  "      --to=DIR               Extract into DIR\n"
  "      --stdout               Write the stream to stdout\n"
  "      --preserve=LIST        Metadata to apply (see below)\n"
  "      --owner-map=WHICH      name (default), numeric\n"
  "      --require=LIST         Turn a degradation into an error\n"
  "      --strict-names         Apply Windows and DOS naming rules\n"
  "      --generation=G         Number, or a set-id prefix\n",
  /*  recover  */
  "      --volume=WHICH         Volume number, or its name\n"
  "      --to=DIR               Write the volume into DIR\n"
  "      --generation=G         Number, or a set-id prefix\n",
  /*  addrecovery  */
  "  -r, --recovery=SPEC        count | N% | SIZE | Nx\n"
  "      --generation=G         Number, or a set-id prefix\n",
  /*  add  */
  "      --rescan=WHICH         stat (default), hash, none\n"
  "      --verify-unchanged     Alias for --rescan=hash\n"
  "      --allow-missing        Do not stop on a vanished entry\n"
  "      --dedup-scope=WHICH    generation (default), chain\n"
  "  ... and everything create accepts.\n",
  /*  consolidate  */
  "      --replace              Remove the collapsed generations\n"
  "      --dry-run              Report what would change\n"
  "  ... and everything create accepts.\n",
  /*  prune  */
  "      --before=G             Remove every generation before G\n"
  "      --generation=G         Remove G; repeatable\n"
  "      --dry-run              Report what would change\n",
  /*  list  */
  "      --generation=G         Number, or a set-id prefix\n"
  "      --chain                Every generation, oldest first\n"
  "      --links                Group hard-link aliases by target\n"
  "      --dedup                Show extents and reference counts\n",
  /*  info  */
  "      --deps                 Per-generation dependency table\n"
  "      --generation=G         Number, or a set-id prefix\n"
  "      --chain                Every generation, oldest first\n",
  /*  explain  */
  "      --generation=G         Number, or a set-id prefix\n",
  /*  undo  */
  "",
  /*  recover-prologue  */
  "",
  /*  benchmark  */
  "      --tiers                Time every runnable kernel tier\n"
};

void xpar_cli_help(xpar_verb v) {
  if (v == XPAR_VERB_NONE) {
    xpar_fputs(
      "Usage: xpar <verb> [options] [arguments]\n"
      "       xpar <set>                    same as: xpar verify <set>\n"
      "\n"
      "Verbs, of which any unambiguous prefix will do:\n", xpar_stdout);
    for (int i = 0; verbs[i].name; i++)
      xpar_fprintf(xpar_stdout, "  %-18s%s\n", verbs[i].name,
                   verb_desc[verbs[i].verb]);
    xpar_fprintf(xpar_stdout, "\n%s", help_global);
    xpar_fputs(
      "\n"
      "Sizes take K, M, G, T (1024-based) or KB, MB, GB, TB (1000-based);\n"
      "a bare number is bytes.  Run 'xpar <verb> --help' for a verb.\n",
      xpar_stdout);
    return;
  }
  xpar_fprintf(xpar_stdout, "Usage: %s\n\n", verb_usage[v]);
  if (*verb_opts[v]) xpar_fprintf(xpar_stdout, "%s\n", verb_opts[v]);
  xpar_fputs(help_global, xpar_stdout);
  if (v == XPAR_VERB_CREATE || v == XPAR_VERB_EXTRACT ||
      v == XPAR_VERB_ADD    || v == XPAR_VERB_CONSOLIDATE)
    xpar_fputs(help_preserve, xpar_stdout);
}

static void v1_flag_refuse(int argc, char ** argv) {
  static const struct { const char * flag;  const char * v2; } tab[] = {
    { "-Jse", "create --layout=sidecar"              },
    { "-Jsd", "repair"                               },
    { "-Jst", "verify"                               },
    { "-Je",  "create --layout=armoured"             },
    { "-Jd",  "extract"                              },
    { "-Jt",  "verify"                               },
    { "-We",  "create --layout=split --codec=matrix" },
    { "-Wd",  "repair"                               },
    { "-Wt",  "verify"                               },
    { "-Le",  "create --layout=split --codec=fft"    },
    { "-Ld",  "repair"                               },
    { "-Lt",  "verify"                               }
  };
  for (sz i = 0; i < ARRAY_LEN(tab); i++) {
    sz n = xpar_strlen(tab[i].flag);
    if (xpar_strncmp(argv[1], tab[i].flag, n)) continue;
    xpar_fprintf(xpar_stderr, "xpar: '%s' is an xpar 1.x mode flag, and "
                 "version 2.0 takes a verb.\n", tab[i].flag);
    xpar_fprintf(xpar_stderr, "xpar: use: xpar %s", tab[i].v2);
    for (int k = 2; k < argc; k++)
      xpar_fprintf(xpar_stderr, " %s", argv[k]);
    xpar_fputs("\n", xpar_stderr);
    xpar_exit(XPAR_EXIT_USAGE);
  }
  /*  Recognize bare v1 mode flags without suggesting one verb.  */
  if (!argv[1][2]) {
    xpar_fprintf(xpar_stderr, "xpar: '%s' is an xpar 1.x mode flag; "
                 "version 2.0 requires a verb.\n", argv[1]);
    xpar_fputs("xpar: use create, verify, repair, or extract; 'xpar --help' "
               "lists all verbs.\n", xpar_stderr);
    xpar_exit(XPAR_EXIT_USAGE);
  }
}

static bool is_dir(const char * p) {
  xpar_dir * d = xpar_opendir(p);
  if (!d) return false;
  xpar_closedir(d);
  return true;
}

static bool is_file(const char * p) {
  xpar_stat_t st;
  return xpar_lstat(p, &st) == 0 && !st.is_dir;
}

static char * cat_str(const char * a, const char * b) {
  sz na = xpar_strlen(a), nb = xpar_strlen(b);
  char * p = (char *) xpar_malloc(na + nb + 1);
  xpar_memcpy(p, a, na);  xpar_memcpy(p + na, b, nb + 1);
  return p;
}

static char * join_path(const char * dir, const char * name) {
  sz n = xpar_strlen(dir);
  if (!n) return dup_str(name);
  if (xpar_path_sep(dir[n - 1])) return cat_str(dir, name);
  { char * mid = cat_str(dir, "/");
    char * out = cat_str(mid, name);
    xpar_free(mid);
    return out; }
}

static void push_vol(xpar_setref * s, char * path) {
  char ** nv = (char **) xpar_realloc(s->vol, (s->count + 1) * sizeof(char *));
  FATAL_UNLESS_CODE(XPAR_EXIT_NOPLAN, "Out of memory.", nv != NULL);
  nv[s->count] = path;  s->vol = nv;  s->count++;
}

static void push_vol_once(xpar_setref * s, char * path) {
  u32 i;
  for (i = 0; i < s->count; i++)
    if (!xpar_strcmp(s->vol[i], path)) {
      xpar_free(path);
      return;
    }
  push_vol(s, path);
}

static void sort_vols(xpar_setref * s) {
  for (u32 i = 1; i < s->count; i++) {
    char * v = s->vol[i];
    u32 j = i;
    while (j && xpar_strcmp(s->vol[j - 1], v) > 0) {
      s->vol[j] = s->vol[j - 1];  j--;
    }
    s->vol[j] = v;
  }
}

static char * swap_ext(const char * p) {
  sz n = xpar_strlen(p), i = n;
  while (i) {
    i--;
    if (xpar_path_sep(p[i])) return NULL;
    if (p[i] == '.') {
      char * head = xpar_strndup(p, i);
      char * out;
      FATAL_UNLESS_CODE(XPAR_EXIT_NOPLAN, "Out of memory.", head != NULL);
      out = cat_str(head, XPAR_EXT);
      xpar_free(head);
      return out;
    }
  }
  return NULL;
}

static void strip_gen_suffix(char * base) {
  sz n = xpar_strlen(base), i = n;
  while (i > 0 && base[i - 1] >= '0' && base[i - 1] <= '9') i--;
  if (i == n || i < 2) return;
  if (base[i - 1] == 'g' && base[i - 2] == '.') base[i - 2] = '\0';
}

static void strip_volume_suffix(char * base) {
  sz n = xpar_strlen(base), dot = n, i;
  while (dot && base[dot - 1] != '/' && base[dot - 1] != '\\' &&
         base[dot - 1] != '.') dot--;
  if (!dot || base[dot - 1] != '.' || dot >= n || base[dot] != 'v') return;
  i = dot + 1;
  if (i == n || base[i] < '0' || base[i] > '9') return;
  while (i < n && base[i] >= '0' && base[i] <= '9') i++;
  if (i == n || base[i++] != '+') return;
  if (i == n || base[i] < '0' || base[i] > '9') return;
  while (i < n && base[i] >= '0' && base[i] <= '9') i++;
  if (i == n) base[dot - 1] = '\0';
}

static void gather_chain_siblings(xpar_setref * s) {
  const char * leaf = s->base;
  sz dlen = 0;
  char * dir;
  xpar_dir * d;
  const xpar_dirent * e;
  sz i;
  for (i = 0; s->base[i]; i++)
    if (xpar_path_sep(s->base[i])) {
      dlen = i + 1;
      leaf = s->base + dlen;
    }
  dir = dlen ? xpar_strndup(s->base, dlen) : dup_str("");
  FATAL_UNLESS_CODE(XPAR_EXIT_NOPLAN, "Out of memory.", dir != NULL);
  d = xpar_opendir(*dir ? dir : ".");
  if (d) {
    while ((e = xpar_readdir(d)) != NULL)
      if (!e->is_dir && xpar_vname_is_index(e->name, leaf))
        push_vol_once(s, join_path(dir, e->name));
    xpar_closedir(d);
  }
  xpar_free(dir);
}

void xpar_cli_resolve_set(const char * arg, xpar_setref * out) {
  out->vol = NULL;  out->count = 0;  out->base = NULL;  out->dir = NULL;
  FATAL_UNLESS("A set argument is required.", arg && *arg);

  if (is_dir(arg)) {
    xpar_dir * d = xpar_opendir(arg);
    const xpar_dirent * e;
    if (!d) FATAL_PERROR(arg);
    out->dir = dup_str(arg);
    while ((e = xpar_readdir(d)) != NULL)
      if (!e->is_dir && xpar_vname_has_ext(e->name))
        push_vol(out, join_path(arg, e->name));
    xpar_closedir(d);
    sort_vols(out);
    FATAL_UNLESS("Directory '%s' holds no " XPAR_EXT " file.",
                 out->count > 0, arg);
  } else if (is_file(arg)) {
    if (xpar_vname_has_ext(arg)) {
      char * b = xpar_strndup(arg, xpar_strlen(arg) - XPAR_EXT_LEN);
      FATAL_UNLESS_CODE(XPAR_EXIT_NOPLAN, "Out of memory.", b != NULL);
      strip_volume_suffix(b);
      strip_gen_suffix(b);
      out->base = b;
      push_vol(out, dup_str(arg));
    } else {
      char * a = cat_str(arg, XPAR_EXT);
      char * b = swap_ext(arg);
      if (is_file(a)) { push_vol(out, a);  out->base = dup_str(arg); }
      else xpar_free(a);
      if (b && is_file(b)) {
        push_vol(out, b);
        if (!out->base) {
          out->base = xpar_strndup(b, xpar_strlen(b) - XPAR_EXT_LEN);
          FATAL_UNLESS_CODE(XPAR_EXIT_NOPLAN, "Out of memory.",
                            out->base != NULL);
        }
      } else xpar_free(b);
      if (!out->count)
        FATAL_FORMAT("No xpar set guards '%s': neither '%s" XPAR_EXT
                     "' nor the same name with its extension replaced by '"
                     XPAR_EXT "' is here.", arg, arg);
    }
  } else {
    /*  A base name: `base` means `base.xpa`.  */
    char * a = cat_str(arg, XPAR_EXT);
    if (is_file(a)) { push_vol(out, a);  out->base = dup_str(arg); }
    else {
      xpar_free(a);
      FATAL_FORMAT("No xpar set found for '%s'; use 'xpar --help' to list "
                   "verbs.", arg);
    }
  }
  if (out->base) gather_chain_siblings(out);
  sort_vols(out);
  for (u32 i = 0; i < out->count; i++) xpar_v1_refuse_if_v1(out->vol[i]);
}

void xpar_setref_free(xpar_setref * s) {
  free_strv(s->vol, s->count);
  xpar_free(s->base);  xpar_free(s->dir);
  s->vol = NULL;  s->count = 0;  s->base = NULL;  s->dir = NULL;
}

/*  Defaults.  */

static void defaults(xpar_options * o) {
  xpar_memset(o, 0, sizeof *o);
  o->jobs         = 0;
  o->progress     = XPAR_PROGRESS_AUTO;
  o->color        = XPAR_COLOR_AUTO;
  o->layout       = XPAR_LAYOUT_SIDECAR;
  o->codec        = XPAR_CLI_AUTO;
  o->field        = XPAR_CLI_AUTO;
  o->align        = XPAR_ALIGN_PACKED;
  o->slice_tag    = 8;
  o->armour       = XPAR_ARMOUR_ALL;
  o->armour_field = XPAR_CLI_AUTO;
  o->volumes      = XPAR_VOLS_LADDER;
  o->dedup        = XPAR_DEDUP_FILE;
  o->preserve     = XPAR_PRES_DEFAULT;
  o->reproducible = xpar_getenv("SOURCE_DATE_EPOCH") != NULL;
  o->resync       = XPAR_RESYNC_AUTO;
  o->resync_step  = 1;
  o->rescan       = XPAR_RESCAN_STAT;
  o->dedup_scope  = XPAR_SCOPE_GENERATION;
  o->owner_map    = XPAR_OWNERMAP_NAME;
  o->dest         = XPAR_DEST_DEFAULT;
}

/*  One option.  */

static void apply(xpar_options * o, const yarg_option * a, u32 * pres_lit,
                  u32 * pres_named) {
  const char * v = a->arg;
  char nm[48];
  xpar_snprintf(nm, sizeof nm, "--%s", a->long_opt ? a->long_opt : "");
  switch (a->opt) {
    case 'v': o->verbose++;  o->quiet = false;  break;
    case 'q': o->quiet = true;  o->verbose = 0;  break;
    case 'f': o->force = true;  break;
    case 'j': o->jobs = (int) need_range(nm, v, 1, 1024);  break;
    case 'm': o->memory = need_size(nm, v);
              FATAL_UNLESS("Option -m expects a positive size.", o->memory);
              break;
    case O_JSON:         o->json = true;  break;
    case O_PROGRESS:     o->progress = XPAR_PROGRESS_ON;   break;
    case O_NO_PROGRESS:  o->progress = XPAR_PROGRESS_OFF;  break;
    case O_COLOR:        o->color = need_word(nm, v, w_color);  break;
    case O_REPRODUCIBLE: o->reproducible = true;  break;
    case O_SIMD:
      xpar_free(o->simd);  o->simd = dup_str(v ? v : "");
      break;

    case 'o': xpar_free(o->output);  o->output = dup_str(v ? v : "");  break;
    case 'r':
      FATAL_UNLESS("Option %s expects a count, a percentage, a size or a "
                   "multiple, as in 100, 15%%, 2.5M or 1x.",
                   v && !xpar_cli_parse_recovery(v, &o->recovery), nm);
      break;
    case O_MIN_RECOVERY: o->min_recovery = need_u64(nm, v);  break;
    case O_MAX_RECOVERY:
      FATAL_UNLESS("Option %s expects a count, a percentage, a size or a "
                   "multiple, as in 100, 15%%, 2.5M or 1x.",
                   v && !xpar_cli_parse_recovery(v, &o->max_recovery), nm);
      break;
    case 's': o->slice_size = need_size(nm, v);
              FATAL_UNLESS("Option -s expects a positive size.",
                           o->slice_size);
              break;
    case 'b': o->slices = need_u64(nm, v);
              FATAL_UNLESS("Option -b expects a positive slice count.",
                           o->slices);
              break;
    case O_CELL:
      o->cell_bytes = need_size(nm, v);
      /* Do not round an explicit cell size. */
      FATAL_UNLESS("--cell must be a multiple of 64 and at least %d bytes.",
                   o->cell_bytes >= XPAR_CELL_MIN &&
                          o->cell_bytes % 64 == 0, XPAR_CELL_MIN);
      FATAL_UNLESS("--cell cannot exceed %" PRIu64 " bytes.",
                   o->cell_bytes <= XPAR_SLICE_REFUSE, XPAR_SLICE_REFUSE);
      break;
    case O_LAYOUT:    o->layout = need_word(nm, v, w_layout);
                      o->layout_given = true;  break;
    case O_CODEC: {
      int i = need_word(nm, v, w_codec);
      o->codec = i ? i - 1 : XPAR_CLI_AUTO;
      break;
    }
    case O_FIELD:        o->field = word_field(nm, v);  break;
    case O_ALIGN:        o->align = need_word(nm, v, w_align);  break;
    case O_SLICE_TAG: {
      int i = need_word(nm, v, w_tag);
      o->slice_tag = i == 0 ? 0 : (i == 1 ? 8 : 16);
      o->slice_tag_given = true;
      break;
    }
    case O_ARMOUR:       o->armour = need_word(nm, v, w_armour);  break;
    case O_ARMOUR_FIELD: o->armour_field = word_field(nm, v);  break;
    case O_ARMOUR_T:  o->armour_t = (u32) need_range(nm, v, 1, 32767);  break;
    case O_ARMOUR_PCT: {
      u64 ip = 0, num = 0, den = 1;  const char * end = "";
      FATAL_UNLESS("Option %s expects a percentage such as 5 or 2.5.",
                   v && !scan_decimal(v, &ip, &num, &den, &end) &&
                   (!end[0] || (end[0] == '%' && !end[1])), nm);
      o->armour_pct = (f64) ip + (f64) num / (f64) den;
      FATAL_UNLESS("Option %s must be above 0 and at most 50.",
                   o->armour_pct > 0.0 && o->armour_pct <= 50.0, nm);
      break;
    }
    case O_BURST: o->burst = need_size(nm, v);
                  FATAL_UNLESS("Option --burst expects a positive size.",
                               o->burst);
                  break;
    case O_DEPTH: o->depth = (u32) need_range(nm, v, 1, 65535);  break;
    case O_VOLUMES:
      if (v && !xpar_strcmp(v, "ladder")) o->volumes = XPAR_VOLS_LADDER;
      else if (v && !xpar_strcmp(v, "equal")) o->volumes = XPAR_VOLS_EQUAL;
      else if (all_digits(v)) {
        o->volumes = XPAR_VOLS_FIXED;
        o->volume_count = (u32) need_range(nm, v, 1, 65535);
      } else FATAL("Option %s takes ladder, equal or a volume count.", nm);
      break;
    case O_DEDUP:           o->dedup = need_word(nm, v, w_dedup);  break;
    case O_DEDUP_CHUNK:     o->dedup_chunk = need_size(nm, v);  break;
    case O_DEDUP_MEMORY:    o->dedup_memory = need_size(nm, v);  break;
    case O_DEDUP_MAX_REFS:  o->dedup_max_refs = need_u64(nm, v);  break;
    case O_PRESERVE:
      o->preserve = parse_pres(nm, v, XPAR_PRES_DEFAULT, pres_lit,
                               pres_named);
      break;
    /*  --require does not imply metadata preservation.  */
    case O_REQUIRE: o->require = parse_pres(nm, v, 0, NULL, NULL);  break;
    case 'R':      o->recurse = true;  break;
    case O_EXCLUDE: push_str(&o->exclude, &o->exclude_count, v ? v : "");
                    break;
    case O_INCLUDE: push_str(&o->include, &o->include_count, v ? v : "");
                    break;
    case O_FOLLOW:  o->follow_symlinks = true;  break;
    case O_BASE:    need_dir(nm, v);
                    xpar_free(o->base_dir);  o->base_dir = dup_str(v);  break;
    case O_LABELS:  o->labels = true;  break;
    case O_AUTH_KEY: xpar_free(o->auth_key);  o->auth_key = dup_str(v);
                     break;
    case O_AUTH_ONLY:       o->auth_only = true;  break;
    case O_NO_VERIFY_AFTER: o->no_verify_after = true;  break;
    case O_SPOOL: o->spool = true;  break;
    case O_SPOOL_DIR:
      o->spool = true;
      need_dir(nm, v);
      xpar_free(o->spool_dir);  o->spool_dir = dup_str(v);
      break;
    case O_STDIN_NAME:
      xpar_free(o->stdin_name);  o->stdin_name = dup_str(v ? v : "");
      break;

    case O_FAST:   o->fast = true;    break;
    case O_STRONG: o->strong = true;  break;
    case O_RESYNC: o->resync = need_word(nm, v, w_resync);  break;
    case O_RESYNC_STEP:
      o->resync_step = (u32) need_range(nm, v, 1, 1u << 30);  break;
    case O_RESYNC_WINDOW: o->resync_window = need_size(nm, v);  break;
    case O_RESYNC_EXH:    o->resync_exhaustive = true;  break;
    case O_SCAN: need_dir(nm, v);
                 xpar_free(o->scan_dir);  o->scan_dir = dup_str(v);  break;
    case O_GENERATION: {
      xpar_genref g;
      xpar_genref * nv;
      parse_genref(nm, v, &g);
      nv = (xpar_genref *) xpar_realloc(o->gens,
                                        (o->gen_count + 1) * sizeof g);
      FATAL_UNLESS_CODE(XPAR_EXIT_NOPLAN, "Out of memory.", nv != NULL);
      nv[o->gen_count] = g;  o->gens = nv;  o->gen_count++;
      break;
    }
    case O_CHAIN:  o->chain = true;  break;
    case O_LINKS:      o->list_links = true;  break;
    case O_LIST_DEDUP: o->list_dedup = true;  break;
    case O_TIERS:      o->benchmark_tiers = true;  break;
    case O_BEFORE: parse_genref(nm, v, &o->before);  o->have_before = true;
                   break;

    case O_IN_PLACE:
    case O_BACKUP:
    case O_TO: {
      int want = a->opt == O_IN_PLACE ? XPAR_DEST_IN_PLACE
               : a->opt == O_BACKUP   ? XPAR_DEST_BACKUP : XPAR_DEST_TO;
      FATAL_UNLESS("Options --in-place, --to, and --backup are exclusive.",
                   o->dest == XPAR_DEST_DEFAULT || o->dest == want);
      o->dest = want;
      if (want == XPAR_DEST_TO) {
        xpar_free(o->to_dir);  o->to_dir = dup_str(v);
      }
      break;
    }
    case O_PARANOID:       o->paranoid = true;  break;
    case O_KEEP_JOURNAL:   o->keep_journal = true;  break;
    case O_NO_JOURNAL:     o->no_journal = true;  break;
    case O_REPLACE_JOURNAL: o->replace_journal = true;  break;
    case O_DRY_RUN:        o->dry_run = true;  break;
    case O_EXIT_ON_CHANGE: o->exit_on_change = true;  break;

    case O_DEEP:          o->deep = true;  break;
    case O_REWRITE:       o->rewrite = true;  break;
    case O_REBUILD_CELLS: o->rebuild_cells = true;  break;

    case O_STDOUT:    o->to_stdout = true;  break;
    case O_OWNER_MAP: o->owner_map = need_word(nm, v, w_owner);  break;
    case O_STRICT_NAMES: o->strict_names = true;  break;

    case O_VOLUME:
      FATAL_UNLESS("Option %s expects a volume number or a volume name.",
                   v && *v, nm);
      o->volume_given = true;
      if (all_digits(v)) o->volume_index = need_u64(nm, v);
      else { xpar_free(o->volume_name);  o->volume_name = dup_str(v); }
      break;

    case O_RESCAN:           o->rescan = need_word(nm, v, w_rescan);  break;
    case O_VERIFY_UNCHANGED: o->rescan = XPAR_RESCAN_HASH;  break;
    case O_ALLOW_MISSING:    o->allow_missing = true;  break;
    case O_DEDUP_SCOPE:      o->dedup_scope = need_word(nm, v, w_scope);
                             break;
    case O_REPLACE:          o->replace = true;  break;
    case O_DEPS:             o->deps = true;  break;
    default: FATAL("internal: option %d has no handler.", a->opt);
  }
}

/*  Cross-option rules.  */

static bool takes_set(xpar_verb v) {
  return v != XPAR_VERB_CREATE && v != XPAR_VERB_BENCHMARK &&
         v != XPAR_VERB_RECOVER_PROLOGUE;
}

static void positionals(xpar_options * o, char ** pos, int n) {
  int first = 0;
  if (o->verb == XPAR_VERB_BENCHMARK) {
    FATAL_UNLESS("Verb benchmark takes no arguments.", n == 0);
    return;
  }
  if (takes_set(o->verb) || o->verb == XPAR_VERB_RECOVER_PROLOGUE) {
    FATAL_UNLESS("Verb %s needs one %s argument.", n >= 1,
                 xpar_verb_name(o->verb),
                 o->verb == XPAR_VERB_RECOVER_PROLOGUE ? "file" : "set");
    o->set = dup_str(pos[0]);
    first = 1;
  }
  if (o->verb == XPAR_VERB_CREATE || o->verb == XPAR_VERB_ADD) {
    for (int i = first; i < n; i++) {
      push_str(&o->paths, &o->path_count, pos[i]);
      if (!xpar_strcmp(pos[i], "-")) o->from_stdin = true;
    }
    FATAL_UNLESS("Verb %s needs at least one input path.", o->path_count > 0,
                 xpar_verb_name(o->verb));
  } else
    FATAL_UNLESS("Verb %s takes one argument, and %d were given.",
                 n == first, xpar_verb_name(o->verb), n);
}

static void validate(xpar_options * o, u32 pres_lit) {
  FATAL_UNLESS("Options -s and -b are mutually exclusive.",
               !(o->slice_size && o->slices));
  FATAL_UNLESS("Options --armour-t and --armour-pct are mutually exclusive.",
               !(o->armour_t && o->armour_pct > 0.0));
  FATAL_UNLESS("Options --burst and --depth are mutually exclusive.",
               !(o->burst && o->depth));
  FATAL_UNLESS("Options --keep-journal and --no-journal contradict each "
               "other.", !(o->keep_journal && o->no_journal));
  FATAL_UNLESS("Options --before and --generation are mutually exclusive.",
               !(o->have_before && o->gen_count));
  /*  Tiny chunk averages exceed extent and index limits.  */
  FATAL_UNLESS("--dedup-chunk must be between 4 KiB and 1 GiB.",
               !o->dedup_chunk ||
               (o->dedup_chunk >= 4096 && o->dedup_chunk <= ((u64) 1 << 30)));
  FATAL_UNLESS("--auth-only needs --auth-key=FILE.",
               !o->auth_only || o->auth_key);
  /*  Sidecar verification uses lstat, so followed links cannot match.  */
  FATAL_UNLESS("--follow-symlinks cannot write a sidecar set; use "
               "--layout=split or --layout=armoured.",
               !o->follow_symlinks || o->layout != XPAR_LAYOUT_SIDECAR ||
               (o->verb != XPAR_VERB_CREATE && o->verb != XPAR_VERB_ADD &&
                o->verb != XPAR_VERB_CONSOLIDATE));

  if (o->from_stdin) {
    xpar_path_status ns;
    FATAL_UNLESS("Pipe input requires --stdin-name=PATH.",
                 o->stdin_name && o->stdin_name[0]);
    ns = xpar_path_check(o->stdin_name,
                         (u32) xpar_strlen(o->stdin_name), 0);
    FATAL_UNLESS("--stdin-name must be a safe relative manifest path: %s.",
                 ns == XPAR_PATH_OK, xpar_path_reason(ns));
    FATAL_UNLESS("Creating from a pipe requires -o/--output.",
                 o->verb != XPAR_VERB_CREATE ||
                 (o->output && o->output[0]));
    FATAL_UNLESS("A pipe must be the only input.",
                 o->path_count == 1);
    if (o->verb == XPAR_VERB_CREATE && !o->spool) {
      FATAL_UNLESS("Direct pipe input requires the matrix codec.",
                   o->codec != XPAR_CODEC_FFT);
      FATAL_UNLESS("Armoured pipe input requires --spool.",
                   o->layout != XPAR_LAYOUT_ARMOURED);
      FATAL_UNLESS("Direct pipe input requires -s, not -b.",
                   !o->slices);
      o->codec = XPAR_CODEC_MATRIX;
    }
    FATAL_UNLESS("Direct pipe input requires --spool or -r as a count or size.",
                 o->verb == XPAR_VERB_ADD || o->spool ||
                 o->recovery.kind == XPAR_R_COUNT ||
                 o->recovery.kind == XPAR_R_BYTES);
  } else {
    FATAL_UNLESS("--stdin-name is meaningful only when the create input is "
                 "'-'.", !o->stdin_name);
  }

  if (o->verb == XPAR_VERB_RECOVER)
    FATAL_UNLESS("Verb recover needs --volume to say which volume to "
                 "regenerate.", o->volume_given);

  if (o->verb == XPAR_VERB_EXTRACT) {
    FATAL_UNLESS("ctime cannot be restored because metadata writes reset it.",
                 !(pres_lit & XPAR_PRES_CTIME));
    FATAL_UNLESS("Restoring privileged mode bits requires -f.",
                 !(pres_lit & XPAR_PRES_SETID) || o->force);
    FATAL_UNLESS("Restoring privileged xattr namespaces requires -f.",
                 !(pres_lit & XPAR_PRES_XATTR_ALL) || o->force);
    o->preserve &= ~(u32) XPAR_PRES_CTIME;
    if (!(pres_lit & XPAR_PRES_SETID)) o->preserve &= ~(u32) XPAR_PRES_SETID;
    if (!(pres_lit & XPAR_PRES_XATTR_ALL))
      o->preserve &= ~(u32) XPAR_PRES_XATTR_ALL;

    FATAL_UNLESS("Options --stdout and --json both claim stdout; send "
                 "the extracted stream to a file, or drop --json.",
                 !(o->to_stdout && o->json));
    FATAL_UNLESS("Refusing to write binary data to a terminal; -f "
                 "overrides.", !o->to_stdout || o->force ||
                 !xpar_is_tty(xpar_stdout));
  }

  if (o->set && !xpar_strcmp(o->set, "-"))
    FATAL_UNLESS("Verb %s needs random access and cannot read a set from "
                 "a pipe; spool it to a file first.",
                 o->verb != XPAR_VERB_VERIFY && o->verb != XPAR_VERB_REPAIR &&
                 o->verb != XPAR_VERB_SCRUB && o->verb != XPAR_VERB_RECOVER,
                 xpar_verb_name(o->verb));
}

/* Diagnose verbs placed after options. */
static const char * misplaced_verb(int argc, char ** argv) {
  int i;
  sz v;
  for (i = 1; i < argc; i++) {
    if (!xpar_strcmp(argv[i], "--")) break;
    for (v = 0; v < sizeof verbs / sizeof *verbs; v++)
      if (verbs[v].name && !xpar_strcmp(argv[i], verbs[v].name))
        return i > 1 ? verbs[v].name : NULL;
  }
  return NULL;
}

void xpar_cli_parse(int argc, char ** argv, xpar_options * o) {
  yarg_settings st;
  yarg_verb_result * r;
  u32 pres_lit = 0;
  u32 pres_named = 0;
  int k;

  defaults(o);
  st.dash_dash = true;
  st.style     = YARG_STYLE_UNIX;

  if (argc > 1 && argv[1][0] == '-' && argv[1][1] &&
      (argv[1][1] == 'J' || argv[1][1] == 'W' || argv[1][1] == 'L'))
    v1_flag_refuse(argc, argv);

  r = yarg_parse_verb(argc, argv, verbs, t_global, st);
  FATAL_UNLESS_CODE(XPAR_EXIT_NOPLAN, "Out of memory.", r != NULL);
  if (r->status == YARG_VERB_AMBIGUOUS) {
    xpar_fprintf(xpar_stderr, "xpar: '%s' is an ambiguous verb prefix.\n",
                 argv[1]);
    xpar_fprintf(xpar_stderr, "xpar: candidates: %s\n",
                 r->cands ? r->cands : "");
    xpar_exit(XPAR_EXIT_USAGE);
  }
  FATAL_UNLESS_CODE(XPAR_EXIT_NOPLAN, "Out of memory.", r->res != NULL);
  if (!r->verb) {
    const char * v = misplaced_verb(argc, argv);
    if (v) FATAL("'%s' must come first.", v);
  }
  if (r->res->error) {
    xpar_fprintf(xpar_stderr, "xpar: %s", r->res->error);
    xpar_fputs("xpar: 'xpar <verb> --help' lists what a verb takes.\n",
               xpar_stderr);
    xpar_exit(XPAR_EXIT_USAGE);
  }
  o->verb = r->verb ? (xpar_verb) r->verb->verb : XPAR_VERB_NONE;

  for (k = 0; k < r->res->argc; k++) {
    if (r->res->args[k].opt == 'h') {
      xpar_cli_help(o->verb);  yarg_verb_destroy(r);  xpar_exit(XPAR_EXIT_OK);
    }
    if (r->res->args[k].opt == 'V') {
      xpar_cli_version();  yarg_verb_destroy(r);  xpar_exit(XPAR_EXIT_OK);
    }
  }

  if (o->verb == XPAR_VERB_NONE) {
    FATAL_UNLESS("A verb is required; 'xpar --help' lists them.",
                 r->res->pos_argc > 0);
    FATAL_UNLESS("Only a single set argument may stand in for a verb; "
                 "'xpar --help' lists them.", r->res->pos_argc == 1);
    o->verb = XPAR_VERB_VERIFY;
  }

  for (k = 0; k < r->res->argc; k++)
    apply(o, &r->res->args[k], &pres_lit, &pres_named);

  o->preserve_explicit = pres_named;

  positionals(o, r->res->pos_args, r->res->pos_argc);
  validate(o, pres_lit);
  yarg_verb_destroy(r);

  if (takes_set(o->verb)) xpar_cli_resolve_set(o->set, &o->set_ref);
  if (o->verb == XPAR_VERB_RECOVER_PROLOGUE) xpar_v1_refuse_if_v1(o->set);
}

void xpar_cli_free(xpar_options * o) {
  free_strv(o->paths, o->path_count);
  free_strv(o->exclude, o->exclude_count);
  free_strv(o->include, o->include_count);
  for (u32 i = 0; i < o->gen_count; i++) xpar_free(o->gens[i].id_prefix);
  xpar_free(o->gens);
  xpar_free(o->before.id_prefix);
  xpar_free(o->set);         xpar_free(o->output);
  xpar_free(o->simd);
  xpar_free(o->base_dir);    xpar_free(o->auth_key);
  xpar_free(o->spool_dir);   xpar_free(o->scan_dir);
  xpar_free(o->stdin_name);
  xpar_free(o->to_dir);      xpar_free(o->volume_name);
  xpar_setref_free(&o->set_ref);
  defaults(o);
}
