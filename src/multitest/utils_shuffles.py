#!/usr/bin/env python3
"""
Parse actual_sample_complexity_N*_n*_alpha*_beta*.txt result files.

Usage
-----
    python read_sample_complexity.py results/actual_sample_complexity_N500_n0_alpha0.05_beta0.txt
    python read_sample_complexity.py "results/actual_sample_complexity_*.txt" --csv out.csv
    python read_sample_complexity.py <file> --method Bonferroni

As a library
------------
    from read_sample_complexity import parse_file
    res = parse_file("actual_sample_complexity_N500_n0_alpha0.05_beta0.txt")
    res.params["N"]                              # 500.0
    res["Bonferroni"].sample_complexity          # 1122.0
    res.total()                                  # sum over methods that actually ran
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

NUM = r"nan|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

FNAME_RE = re.compile(
    rf"actual_sample_complexity"
    rf"_N(?P<N>{NUM})"
    rf"_n(?P<n>{NUM})"
    rf"_alpha(?P<alpha>{NUM})"
    rf"_beta(?P<beta>{NUM})"
    rf"\.txt$"
)

# e.g. "Our Approach (Graphical):   0.0  (rejected nan/10)"
ROW_RE = re.compile(
    rf"^(?P<method>\S.*?):\s+"
    rf"(?P<complexity>{NUM})"
    rf"(?:\s*\(rejected\s+(?P<rejected>{NUM})\s*/\s*(?P<total>\d+)\))?"
    rf"\s*$"
)

HEADER_KV_RE = re.compile(rf"(\w+)\s*=\s*({NUM})")
TOTAL_HYP_RE = re.compile(r"^Total hypotheses:\s*(\d+)\s*$")


def _to_float(s: str | None) -> float:
    if s is None:
        return math.nan
    return math.nan if s.strip().lower() == "nan" else float(s)


@dataclass
class MethodResult:
    method: str
    sample_complexity: float
    rejected: float = math.nan
    n_hypotheses: int | None = None

    @property
    def did_run(self) -> bool:
        """A 0.0 complexity with `rejected nan` is a sentinel for 'no data',
        not a genuinely free method. Treat those rows as missing."""
        return not (math.isnan(self.rejected) and self.sample_complexity == 0.0)


@dataclass
class ComplexityFile:
    path: Path
    params: dict[str, float] = field(default_factory=dict)
    total_hypotheses: int | None = None
    methods: dict[str, MethodResult] = field(default_factory=dict)

    def __getitem__(self, method: str) -> MethodResult:
        return self.methods[method]

    def get(self, method: str, default=None):
        return self.methods.get(method, default)

    def total(self, include_missing: bool = False) -> float:
        """Sum of sample complexity across methods.

        By default skips rows that never ran (0.0 / rejected nan)."""
        return sum(
            m.sample_complexity
            for m in self.methods.values()
            if include_missing or m.did_run
        )


def parse_file(path: str | Path, strict: bool = False) -> ComplexityFile:
    path = Path(path)
    res = ComplexityFile(path=path)

    m = FNAME_RE.search(path.name)
    if m:
        res.params = {k: _to_float(v) for k, v in m.groupdict().items()}

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or set(line) <= {"=", "-"}:
            continue

        if line.lower().startswith("actual sample complexit"):
            # Header params override / fill in the filename-derived ones.
            for k, v in HEADER_KV_RE.findall(line):
                res.params[k] = _to_float(v)
            continue

        th = TOTAL_HYP_RE.match(line)
        if th:
            res.total_hypotheses = int(th.group(1))
            continue

        row = ROW_RE.match(line)
        if row:
            name = row.group("method").strip()
            res.methods[name] = MethodResult(
                method=name,
                sample_complexity=_to_float(row.group("complexity")),
                rejected=_to_float(row.group("rejected")),
                n_hypotheses=int(row.group("total")) if row.group("total") else res.total_hypotheses,
            )
            continue

        msg = f"{path.name}:{lineno}: unparsed line: {line!r}"
        if strict:
            raise ValueError(msg)
        print(f"warning: {msg}", file=sys.stderr)

    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="file paths or glob patterns")
    ap.add_argument("--method", help="print only this method's sample complexity")
    ap.add_argument("--include-missing", action="store_true",
                    help="count 0.0/rejected-nan rows in the total instead of skipping them")
    ap.add_argument("--csv", help="write one tidy row per (file, method) to this CSV")
    ap.add_argument("--strict", action="store_true", help="error on unparsed lines")
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.paths:
        hits = sorted(glob.glob(p))
        files.extend(Path(h) for h in hits) if hits else files.append(Path(p))

    results = [parse_file(f, strict=args.strict) for f in files]

    if args.method:
        for r in results:
            mr = r.get(args.method)
            val = "MISSING" if mr is None else f"{mr.sample_complexity:g}"
            print(f"{r.path.name}\t{val}")
    else:
        for r in results:
            pstr = ", ".join(f"{k}={v:g}" for k, v in r.params.items())
            print(f"\n{r.path.name}  [{pstr}]  ({r.total_hypotheses} hypotheses)")
            width = max((len(m) for m in r.methods), default=0)
            for mr in r.methods.values():
                flag = "" if mr.did_run else "   <- no data"
                rej = "nan" if math.isnan(mr.rejected) else f"{mr.rejected:g}"
                print(f"  {mr.method:<{width}}  {mr.sample_complexity:>10.1f}"
                      f"  (rejected {rej}/{mr.n_hypotheses}){flag}")
            print(f"  {'TOTAL':<{width}}  {r.total(args.include_missing):>10.1f}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "N", "n", "alpha", "beta", "method",
                        "sample_complexity", "rejected", "n_hypotheses", "did_run"])
            for r in results:
                for mr in r.methods.values():
                    w.writerow([
                        r.path.name,
                        *(r.params.get(k, "") for k in ("N", "n", "alpha", "beta")),
                        mr.method, mr.sample_complexity, mr.rejected,
                        mr.n_hypotheses, mr.did_run,
                    ])
        print(f"\nwrote {args.csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())