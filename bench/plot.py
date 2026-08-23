#!/usr/bin/env python3
#  Copyright (C) 2022-2026 Kamila Szewczyk
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; version 3 of the License only.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Aggregate benchmark CSV into medians, spreads and optional plots.

    python3 bench/plot.py <result-dir> [--out <dir>] [--format png|svg|pdf]
"""

import argparse
import csv
import json
import os
import statistics
import sys

MIB = 1048576.0


def load_environment(path):
    try:
        with open(os.path.join(path, "environment.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load_rows(path):
    with open(os.path.join(path, "results.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("elapsed_us", "input_bytes", "maxrss_kb", "status",
                    "rep", "field"):
            try:
                r[key] = int(r[key])
            except (KeyError, TypeError, ValueError):
                r[key] = 0
    return rows


def load_kernels(path):
    out = []
    try:
        with open(os.path.join(path, "kernels.json")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("type") == "kernel":
                    out.append(rec)
    except OSError:
        pass
    return out


def key_of(row):
    return (row["op"], row["codec"], row["field"], row["slice_size"],
            row["recovery"], row["layout"])


def aggregate(rows):
    """Median throughput per configuration, with the spread kept."""
    groups = {}
    for r in rows:
        groups.setdefault(key_of(r), []).append(r)

    out = []
    for key, rs in sorted(groups.items()):
        good = [r for r in rs if r["status"] == 0 and r["elapsed_us"] > 0]
        op, codec, field, slice_size, recovery, layout = key
        if not good:
            out.append(dict(op=op, codec=codec, field=field,
                            slice_size=slice_size, recovery=recovery,
                            layout=layout, reps=len(rs),
                            status=rs[0]["status"],
                            median_us=0, min_us=0, max_us=0, mib_s=0.0,
                            maxrss_kb=0))
            continue
        times = sorted(r["elapsed_us"] for r in good)
        nbytes = good[0]["input_bytes"]
        median = statistics.median(times)
        out.append(dict(
            op=op, codec=codec, field=field, slice_size=slice_size,
            recovery=recovery, layout=layout, reps=len(good), status=0,
            median_us=int(median), min_us=times[0], max_us=times[-1],
            mib_s=(nbytes / MIB) / (median / 1e6) if median else 0.0,
            maxrss_kb=max(r["maxrss_kb"] for r in good)))
    return out


def write_summary(path, agg):
    fields = ["op", "codec", "field", "slice_size", "recovery", "layout",
              "reps", "status", "median_us", "min_us", "max_us", "mib_s",
              "maxrss_kb"]
    dest = os.path.join(path, "summary.csv")
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in agg:
            row = dict(row)
            row["mib_s"] = "%.2f" % row["mib_s"]
            w.writerow(row)
    return dest


def print_table(agg):
    head = ("%-15s %-7s %-3s %-9s %-6s %10s %10s %8s" %
            ("op", "codec", "gf", "slice", "rec", "median us", "MiB/s",
             "spread"))
    print(head)
    print("-" * len(head))
    for r in agg:
        if r["status"]:
            spread = "-"
            rate = "refused %d" % r["status"]
        else:
            span = r["max_us"] - r["min_us"]
            spread = ("%.1f%%" % (100.0 * span / r["median_us"])
                      if r["median_us"] else "-")
            rate = "%.1f" % r["mib_s"]
        print("%-15s %-7s %-3s %-9s %-6s %10s %10s %8s" %
              (r["op"], r["codec"], r["field"], r["slice_size"],
               r["recovery"], r["median_us"] or "-", rate, spread))


# Figures.

def plot_all(path, agg, kernels, env, fmt):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed, so no figures were drawn.\n"
              "summary.csv holds the same numbers.", file=sys.stderr)
        return []

    written = []
    subtitle = "%s, %s" % (env.get("cpu_model") or "unknown CPU",
                           env.get("xpar_version") or "unknown build")
    if env.get("cache_mode") == "drop":
        subtitle += ", cold cache"

    def save(fig, name):
        dest = os.path.join(path, name + "." + fmt)
        fig.savefig(dest, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(dest)

    ok = [r for r in agg if not r["status"] and r["mib_s"] > 0]

    #  Throughput per operation, one bar group per codec and field.
    if ok:
        ops = sorted({r["op"] for r in ok})
        configs = sorted({(r["codec"], r["field"]) for r in ok})
        fig, ax = plt.subplots(figsize=(1.6 * len(ops) + 4, 4.5))
        width = 0.8 / max(len(configs), 1)
        for i, cfg in enumerate(configs):
            xs, ys = [], []
            for j, op in enumerate(ops):
                sel = [r["mib_s"] for r in ok
                       if r["op"] == op and (r["codec"], r["field"]) == cfg]
                xs.append(j + i * width - 0.4 + width / 2)
                ys.append(statistics.median(sel) if sel else 0.0)
            ax.bar(xs, ys, width=width,
                   label="%s, GF(2^%d)" % (cfg[0], cfg[1]))
        ax.set_xticks(range(len(ops)))
        ax.set_xticklabels(ops, rotation=15, ha="right")
        ax.set_ylabel("MiB/s (median of repetitions)")
        ax.set_title("xpar throughput by operation\n" + subtitle, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        save(fig, "throughput-by-op")

    #  Throughput against the redundancy the set carries.
    creates = [r for r in ok if r["op"] == "create"]
    if creates and len({r["recovery"] for r in creates}) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        def rec_key(text):
            try:
                return float(text.rstrip("%"))
            except ValueError:
                return 0.0

        for cfg in sorted({(r["codec"], r["field"], r["slice_size"])
                           for r in creates}):
            sel = sorted((r for r in creates
                          if (r["codec"], r["field"], r["slice_size"]) == cfg),
                         key=lambda r: rec_key(r["recovery"]))
            if len(sel) < 2:
                continue
            ax.plot([rec_key(r["recovery"]) for r in sel],
                    [r["mib_s"] for r in sel], marker="o",
                    label="%s, GF(2^%d), Z=%s" % (cfg[0], cfg[1], cfg[2]))
        ax.set_xlabel("recovery, per cent of the data")
        ax.set_ylabel("create MiB/s")
        ax.set_title("Encoding cost against redundancy\n" + subtitle,
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        save(fig, "throughput-vs-recovery")

    #  The region kernels, which is where the SIMD tiers show up.
    if kernels:
        by_op = {}
        for k in kernels:
            usec = k.get("usec") or 1
            rate = (k.get("bytes", 0) / MIB) / (usec / 1e6)
            by_op.setdefault(k.get("operation", "?"), []).append(
                (k.get("tier", "?"), rate))
        fig, axes = plt.subplots(1, len(by_op),
                                 figsize=(4.2 * len(by_op), 4.2),
                                 squeeze=False)
        for ax, (op, pairs) in zip(axes[0], sorted(by_op.items())):
            pairs.sort(key=lambda p: p[1])
            ax.barh([p[0] for p in pairs], [p[1] for p in pairs])
            ax.set_xlabel("MiB/s")
            ax.set_title(op, fontsize=10)
            ax.grid(axis="x", alpha=0.3)
        fig.suptitle("Kernel tiers\n" + subtitle, fontsize=10)
        save(fig, "kernels")

    return written


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="a directory written by bench/run.sh")
    ap.add_argument("--format", default="png", choices=("png", "svg", "pdf"))
    args = ap.parse_args()

    path = args.results
    if not os.path.isfile(os.path.join(path, "results.csv")):
        sys.exit("%s holds no results.csv" % path)

    env = load_environment(path)
    rows = load_rows(path)
    agg = aggregate(rows)
    dest = write_summary(path, agg)

    if env:
        print("%s, %s" % (env.get("cpu_model") or "unknown CPU",
                          env.get("xpar_version") or "unknown build"))
        print("corpus %s bytes from seed %s, %s repetitions, %s cache" %
              (env.get("corpus_bytes"), env.get("corpus_seed"),
               env.get("repetitions"), env.get("cache_mode")))
        print()
    print_table(agg)
    print("\nwrote %s" % dest)
    for f in plot_all(path, agg, load_kernels(path), env, args.format):
        print("wrote %s" % f)


if __name__ == "__main__":
    main()
