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

"""Aggregate benchmark results and plot scan and repair throughput.

Rows that fail work checks are counted but excluded from figures.

    python3 bench/plot.py <result-dir> [--format png|svg|pdf]
"""

import argparse
import csv
import json
import os
import statistics
import sys

MIB = 1048576.0

NUMERIC = ("rep", "seed", "corpus_bytes", "field", "slice_size",
           "cell_bytes", "slices", "recovery_slices", "damaged_cells",
           "damaged_slices", "column_depth", "column_groups",
           "repaired_bytes", "archive_bytes", "nominal_payload_bytes",
           "format_overhead_bytes", "scan_bytes", "elapsed_us",
           "maxrss_kb", "in_blocks", "out_blocks", "status", "expect",
           "work_ok")


def load_environment(path):
    try:
        with open(os.path.join(path, "environment.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load_competitors(path):
    try:
        with open(os.path.join(path, "competitors.json")) as f:
            return json.load(f).get("tools", [])
    except (OSError, ValueError):
        return []


def load_rows(path):
    with open(os.path.join(path, "results.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in NUMERIC:
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
                try:
                    rec = json.loads(line.strip() or "{}")
                except ValueError:
                    continue
                if rec.get("type") == "kernel":
                    out.append(rec)
    except OSError:
        pass
    return out


def key_of(row):
    """What makes two rows repetitions of the same measurement.

    The corpus size belongs here: a scaling sweep varies nothing else, and
    without it four sizes aggregate into one row that reports whichever
    the median happened to land on."""
    return (row["experiment"], row["tool"], row["op"], row["codec"],
            row["field"], row["slice_size"], row["cell_bytes"],
            row["corpus_bytes"], row["recovery_spec"], row["layout"],
            row["damage"], row["note"])


def rate(num, usec):
    return (num / MIB) / (usec / 1e6) if usec and num else 0.0


def aggregate(rows):
    """Aggregate usable repetitions."""
    groups = {}
    for r in rows:
        groups.setdefault(key_of(r), []).append(r)

    out = []
    for key, rs in sorted(groups.items()):
        (exp, tool, op, codec, field, slice_size, cell, corpus_bytes,
         rec, layout, dmg, note) = key
        usable = [r for r in rs if r["work_ok"] == 1 and r["elapsed_us"] > 0]
        # Infer foreign-tool recovery from post-run byte checks.
        note = rs[0]["note"] or ""
        if tool.startswith("xpar"):
            recovered = 1 if rs[0]["status"] == 0 else 0
        elif "recovered" in note:
            recovered = 1
        elif "lost" in note:
            recovered = 0
        else:
            recovered = 1 if rs[0]["status"] == 0 else 0
        base = dict(experiment=exp, tool=tool, op=op, codec=codec,
                    recovered=recovered,
                    field=field, slice_size=slice_size, cell_bytes=cell,
                    recovery_spec=rec, layout=layout, damage=dmg, note=note,
                    reps=len(rs), usable=len(usable),
                    status=rs[0]["status"], expect=rs[0]["expect"],
                    corpus_bytes=rs[0]["corpus_bytes"],
                    corpus=rs[0]["corpus"],
                    slices=rs[0]["slices"],
                    recovery_slices=rs[0]["recovery_slices"],
                    damaged_cells=rs[0]["damaged_cells"],
                    damaged_slices=rs[0]["damaged_slices"],
                    column_depth=rs[0]["column_depth"],
                    column_groups=rs[0]["column_groups"],
                    archive_bytes=rs[0]["archive_bytes"],
                    nominal_payload_bytes=rs[0]["nominal_payload_bytes"],
                    format_overhead_bytes=(
                        rs[0]["format_overhead_bytes"]),
                    expected_unsupported=(
                        rs[0].get("expected_unsupported") or
                        rs[0].get("unsupported", "")))
        if not usable:
            base.update(median_us=0, min_us=0, max_us=0, spread_pct=0.0,
                        create_mib_s=0.0, scan_mib_s=0.0,
                        repaired_mib_s=0.0, read_amplification=0.0,
                        phys_read_mib=0.0, phys_write_mib=0.0,
                        repaired_bytes=0, scan_bytes=0, maxrss_kb=0)
            out.append(base)
            continue

        times = sorted(r["elapsed_us"] for r in usable)
        median = int(statistics.median(times))
        repaired = usable[0]["repaired_bytes"]
        scan = usable[0]["scan_bytes"] or usable[0]["corpus_bytes"]
        base.update(
            median_us=median, min_us=times[0], max_us=times[-1],
            spread_pct=100.0 * (times[-1] - times[0]) / median if median
            else 0.0,
            create_mib_s=rate(usable[0]["corpus_bytes"], median),
            scan_mib_s=rate(scan, median),
            repaired_mib_s=rate(repaired, median),
            read_amplification=(scan / repaired) if repaired else 0.0,
            phys_read_mib=statistics.median(
                r["in_blocks"] for r in usable) * 512 / MIB,
            phys_write_mib=statistics.median(
                r["out_blocks"] for r in usable) * 512 / MIB,
            repaired_bytes=repaired, scan_bytes=scan,
            maxrss_kb=max(r["maxrss_kb"] for r in usable))
        out.append(base)
    return out


SUMMARY_FIELDS = [
    "experiment", "tool", "op", "codec", "field", "slice_size", "cell_bytes",
    "recovery_spec", "layout", "damage", "note", "reps", "usable", "status",
    "expect", "recovered", "expected_unsupported", "corpus_bytes",
    "slices", "recovery_slices", "damaged_cells", "damaged_slices",
    "column_depth", "column_groups", "archive_bytes",
    "nominal_payload_bytes", "format_overhead_bytes",
    "scan_bytes", "repaired_bytes", "median_us", "min_us", "max_us",
    "spread_pct", "create_mib_s", "scan_mib_s", "repaired_mib_s",
    "read_amplification", "phys_read_mib", "phys_write_mib", "maxrss_kb"]


def write_summary(path, agg):
    dest = os.path.join(path, "summary.csv")
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in agg:
            row = dict(row)
            for k in ("spread_pct", "create_mib_s", "scan_mib_s",
                      "repaired_mib_s", "read_amplification", "phys_read_mib",
                      "phys_write_mib"):
                row[k] = "%.3f" % row[k]
            w.writerow(row)
    return dest


def config_label(r):
    """Format a row's distinguishing settings."""
    bits = []
    if r["codec"]:
        bits.append("%s/gf%d" % (r["codec"], r["field"]))
    if r["slice_size"]:
        bits.append("Z=%dk" % (r["slice_size"] // 1024))
    if r["cell_bytes"]:
        bits.append("Y=%dk" % (r["cell_bytes"] // 1024))
    if r["recovery_spec"]:
        bits.append("r=%s" % r["recovery_spec"])
    if r["note"]:
        bits.append(r["note"])
    elif r["damage"]:
        bits.append(r["damage"])
    return " ".join(bits) or "-"


def print_tables(agg, rows):
    bad = [r for r in rows if r["work_ok"] != 1]
    if bad:
        print("WARNING: %d of %d repetitions failed their work check and are "
              "excluded" % (len(bad), len(rows)))
        seen = set()
        for r in bad:
            k = (r["experiment"], r["op"], r["tool"], r["note"])
            if k in seen:
                continue
            seen.add(k)
            print("  %s %s %s %s" % k)
        print()

    for exp in sorted({r["experiment"] for r in agg}):
        sel = [r for r in agg if r["experiment"] == exp]
        print("== %s ==" % exp)
        if exp in ("envelope", "scatter"):
            head = "%-20s %-13s %-10s %8s %8s %6s %6s %9s %9s" % (
                "tool", "damage", "note", "cells", "slices", "depth",
                "groups", "outcome", "median us")
            print(head)
            print("-" * len(head))
            for r in sorted(sel, key=lambda r: (r["tool"], r["note"],
                                                r["damaged_cells"],
                                                r["column_depth"])):
                if r["tool"].startswith("xpar"):
                    outcome = "repaired" if r["recovered"] else "refused"
                else:
                    outcome = "recovered" if r["recovered"] else "lost"
                print("%-20s %-13s %-10s %8d %8d %6d %6d %9s %9d" % (
                    r["tool"], (r["damage"] or "-")[:13],
                    (r["note"] or "-")[:10], r["damaged_cells"],
                    r["damaged_slices"], r["column_depth"],
                    r["column_groups"], outcome, r["median_us"]))
        else:
            head = "%-20s %-14s %-22s %10s %10s %10s %8s" % (
                "tool", "op", "configuration", "median us", "scan MiB/s",
                "rep MiB/s", "spread")
            print(head)
            print("-" * len(head))
            for r in sel:
                print("%-20s %-14s %-22s %10d %10.1f %10.1f %7.1f%%" % (
                    r["tool"], r["op"][:14], config_label(r)[:22],
                    r["median_us"], r["scan_mib_s"], r["repaired_mib_s"],
                    r["spread_pct"]))
        print()


def plot_all(path, agg, kernels, env, fmt):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; wrote summary.csv without figures.",
              file=sys.stderr)
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

    def having(exp):
        return [r for r in agg if r["experiment"] == exp and r["usable"]]

    # Throughput by operation, one group per tool.
    ok = [r for r in agg if r["usable"] and r["experiment"] in
          ("throughput", "baseline")]
    if ok:
        ops = sorted({r["op"] for r in ok})
        tools = sorted({r["tool"] for r in ok})
        fig, ax = plt.subplots(figsize=(1.9 * len(ops) + 4, 4.6))
        width = 0.8 / max(len(tools), 1)
        for i, tool in enumerate(tools):
            xs, ys = [], []
            for j, op in enumerate(ops):
                sel = [r["scan_mib_s"] for r in ok
                       if r["op"] == op and r["tool"] == tool
                       and r["scan_mib_s"] > 0]
                xs.append(j + i * width - 0.4 + width / 2)
                ys.append(statistics.median(sel) if sel else 0.0)
            ax.bar(xs, ys, width=width, label=tool)
        ax.set_xticks(range(len(ops)))
        ax.set_xticklabels(ops, rotation=15, ha="right")
        ax.set_ylabel("MiB/s scanned (median)")
        ax.set_title("Throughput by operation\n" + subtitle, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        save(fig, "throughput-by-op")

    # The envelope: outcome against the depth of the deepest column.
    env_rows = having("envelope")
    if env_rows:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        widths = sorted({r["note"] for r in env_rows},
                        key=lambda w: int(w.split("=")[-1]) if "=" in w
                        else 0)
        for w in widths:
            sel = sorted((r for r in env_rows if r["note"] == w),
                         key=lambda r: r["column_depth"])
            ax.plot([r["column_depth"] for r in sel],
                    [r["recovered"] for r in sel],
                    marker="o", label=w or "width")
        r0 = env_rows[0]
        ax.axvline(r0["recovery_slices"] + 0.5, color="crimson", ls="--",
                   label="R = %d" % r0["recovery_slices"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["refused", "repaired"])
        ax.set_xlabel("erasures in the deepest column")
        ax.set_title("Recovery envelope: the deepest column decides\n" +
                     subtitle, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        save(fig, "envelope")

        # Contrast total cells with deepest-column outcome.
        fig, ax = plt.subplots(figsize=(7, 4.4))
        good = [r for r in env_rows if r["recovered"]]
        bad = [r for r in env_rows if not r["recovered"]]
        ax.scatter([r["damaged_cells"] for r in good],
                   [r["column_depth"] for r in good], marker="o",
                   label="repaired")
        ax.scatter([r["damaged_cells"] for r in bad],
                   [r["column_depth"] for r in bad], marker="x",
                   label="refused")
        ax.axhline(env_rows[0]["recovery_slices"] + 0.5, color="crimson",
                   ls="--", label="R")
        ax.set_xlabel("cells lost in total")
        ax.set_ylabel("erasures in the deepest column")
        ax.set_title("Outcome follows the depth, not the total\n" + subtitle,
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        save(fig, "envelope-total")

    # Scattered against concentrated, xpar and PAR2 side by side.
    sc = [r for r in having("scatter") if r["op"] == "repair"]
    if sc:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        tools = sorted({r["tool"] for r in sc})
        for ti, tool in enumerate(tools):
            for shape in sorted({r["damage"] for r in sc}):
                sel = sorted((r for r in sc if r["tool"] == tool and
                              r["damage"] == shape),
                             key=lambda r: r["damaged_cells"])
                if not sel:
                    continue
                # Separate otherwise overlapping block-code curves.
                off = (ti - (len(tools) - 1) / 2.0) * 0.035
                off += 0.012 if shape == "spread" else -0.012
                ax.plot([r["damaged_cells"] for r in sel],
                        [r["recovered"] + off for r in sel],
                        marker="o" if shape == "spread" else "s",
                        ls="-" if tool.startswith("xpar") else "--",
                        alpha=0.85, label="%s, %s" % (tool, shape))
        ax.set_xscale("log", base=2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["lost", "recovered"])
        ax.set_ylim(-0.25, 1.25)
        ax.set_xlabel("faults injected")
        ax.set_title("Scattered faults against concentrated ones, "
                     "matched redundancy\n" + subtitle, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        save(fig, "scatter")

    # Cell size against metadata and repair cost.
    cs = having("cellsize")
    if cs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        creates = sorted((r for r in cs if r["op"] == "create"),
                         key=lambda r: r["cell_bytes"])
        if creates:
            base = min(r["archive_bytes"] for r in creates) or 1
            axes[0].plot([r["cell_bytes"] for r in creates],
                         [100.0 * (r["archive_bytes"] - base) / base
                          for r in creates], marker="o")
            axes[0].set_xscale("log", base=2)
            axes[0].set_xlabel("cell size Y, bytes")
            axes[0].set_ylabel("archive growth over the largest cell, %")
            axes[0].set_title("Metadata cost of a finer erasure unit",
                              fontsize=10)
            axes[0].grid(alpha=0.3)
        reps = sorted((r for r in cs if r["op"] == "repair"),
                      key=lambda r: r["cell_bytes"])
        if reps:
            axes[1].plot([r["cell_bytes"] for r in reps],
                         [r["repaired_bytes"] / MIB for r in reps],
                         marker="o", label="user bytes put back")
            axes[1].set_xscale("log", base=2)
            axes[1].set_xlabel("cell size Y, bytes")
            axes[1].set_ylabel("MiB rewritten for 64 faults")
            axes[1].set_title("Write amplification of a coarser unit",
                              fontsize=10)
            axes[1].grid(alpha=0.3)
        fig.suptitle(subtitle, fontsize=9)
        save(fig, "cellsize")

    # Repair amplification.
    am = having("amplify")
    if am:
        sel = sorted(am, key=lambda r: r["damaged_cells"])
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot([r["damaged_cells"] for r in sel],
                [r["read_amplification"] for r in sel], marker="o",
                label="bytes read per byte put back")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("cells lost")
        ax.set_ylabel("read amplification")
        ax.set_title("A repair reads the archive whatever it puts back\n" +
                     subtitle, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        save(fig, "amplify")

    # Scaling.
    sg = having("scaling")
    if sg:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for op in sorted({r["op"] for r in sg}):
            sel = sorted((r for r in sg if r["op"] == op),
                         key=lambda r: r["corpus_bytes"])
            metric = "create_mib_s" if op == "create" else "scan_mib_s"
            ax.plot([r["corpus_bytes"] / MIB for r in sel],
                    [r[metric] for r in sel], marker="o", label=op)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("corpus, MiB")
        ax.set_ylabel("MiB/s")
        ax.set_title("Throughput against corpus size\n" + subtitle,
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        save(fig, "scaling")

    # Kernel tiers.
    if kernels:
        by_op = {}
        for k in kernels:
            usec = k.get("usec") or 1
            by_op.setdefault(k.get("operation", "?"), []).append(
                (k.get("tier", "?"), (k.get("bytes", 0) / MIB) / (usec / 1e6)))
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
        print("seed %s, %s repetitions, %s cache, %s cores" %
              (env.get("corpus_seed"), env.get("repetitions"),
               env.get("cache_mode"), env.get("cores")))
        print()
    for t in load_competitors(path):
        print("baseline %-24s %-12s %s" %
              (t.get("name"), t.get("version"), t.get("status")))
    print()
    print_tables(agg, rows)
    print("wrote %s" % dest)
    for f in plot_all(path, agg, load_kernels(path), env, args.format):
        print("wrote %s" % f)


if __name__ == "__main__":
    main()
