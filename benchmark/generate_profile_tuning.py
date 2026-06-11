#!/usr/bin/env python3
"""Emit the checked-in profile tuning table.

The semantic rows (Knuth, Motzkin, leading-zero demotion, integral/custom
fallbacks) live in poly.hpp. This generator owns the profile-calibrated scalar
and batch rows that are expected to drift with target CPUs. CI runs this script
and diffs it against include/polygnition/detail/profile_tuning.inc.

Passing --batch-probe runs a batch_throughput_probe executable and derives
profile/width-specific throughput rows from its CSV output; otherwise the
checked-in canonical rows are emitted. Passing --algorithm-probe runs
algorithm_selection_probe and preserves its CSV on stderr so a calibration job
can review the measured scalar choices alongside the generated diff.
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from pathlib import Path

ANY = "any_degree"

LATENCY_REAL = {
    "generic": [(0, 16, "horner", 0), (16, 48, "dorn", 4), (48, ANY, "dorn", 5)],
    "x86_intel_coffee_lake": [
        (0, 15, "horner", 0),
        (15, 47, "dorn", 4),
        (47, 80, "dorn", 5),
        (80, ANY, "dorn", 6),
    ],
    "x86_intel_icelake_avx512": [
        (0, 16, "horner", 0),
        (16, 18, "dorn", 2),
        (18, 64, "dorn", 4),
        (64, ANY, "dorn", 6),
    ],
    "x86_amd_zen3": [
        (0, 13, "horner", 0),
        (13, 17, "dorn", 2),
        (17, 47, "dorn", 4),
        (47, 64, "dorn", 5),
        (64, ANY, "dorn", 6),
    ],
}

LATENCY_COMPLEX = {
    "generic": [(0, 8, "horner", 0), (8, 11, "dorn", 2), (11, ANY, "dorn", 4)],
    "x86_intel_coffee_lake": [(0, 6, "horner", 0), (6, ANY, "dorn", 2)],
    "x86_intel_icelake_avx512": [
        (0, 8, "horner", 0),
        (8, 11, "dorn", 2),
        (11, ANY, "dorn", 4),
    ],
    "x86_amd_zen3": [(0, 6, "horner", 0), (6, 8, "dorn", 2), (8, ANY, "dorn", 4)],
}

PROFILE_NAMES = [
    "generic",
    "x86_intel_coffee_lake",
    "x86_intel_icelake_avx512",
    "x86_amd_zen3",
]


def canonical_throughput_rows(double_lanes: int) -> list[tuple[int, int, int | str, str, int, int, int]]:
    # Small runtime polynomials stay on Horner; static safe quartics are handled
    # earlier by the Motzkin semantic row. Float rows use the widest lane count
    # swept by the probe. Ice Lake can select an 8-lane double row; if the
    # translation unit lacks AVX-512, lanes<T, 8> falls back to array storage and
    # remains correct.
    return [
        (32, 0, 8, "horner", 0, 8, 4),
        (32, 8, 26, "dorn", 4, 8, 4),
        (32, 26, ANY, "dorn", 6, 8, 4),
        (64, 0, 8, "horner", 0, double_lanes, 4),
        (64, 8, 26, "dorn", 4, double_lanes, 4),
        (64, 26, ANY, "dorn", 6, double_lanes, 4),
    ]


CANONICAL_THROUGHPUT = {
    "generic": canonical_throughput_rows(4),
    "x86_intel_coffee_lake": canonical_throughput_rows(4),
    "x86_intel_icelake_avx512": canonical_throughput_rows(8),
    "x86_amd_zen3": canonical_throughput_rows(4),
}


def run_probe(path: str | None) -> str:
    if path is None:
        return ""
    completed = subprocess.run([path], check=True, text=True, capture_output=True)
    sys.stderr.write(completed.stdout)
    return completed.stdout


def strategy_from_mode(mode: str) -> tuple[str, int] | None:
    if mode == "batch_explicit_horner":
        return ("horner", 0)
    if mode.startswith("batch_dorn"):
        return ("dorn", int(mode.removeprefix("batch_dorn")))
    return None


def default_lanes(profile: str, element_bits: int) -> int:
    if element_bits == 32:
        return 8
    if profile == "x86_intel_icelake_avx512":
        return 8
    return 4


def coalesce_rows(
    rows: list[tuple[int, int, int | str, str, int, int, int]],
) -> list[tuple[int, int, int | str, str, int, int, int]]:
    coalesced: list[tuple[int, int, int | str, str, int, int, int]] = []
    for bits, first, last, strategy, stride, lanes, unroll in rows:
        if first == last:
            continue
        if coalesced and coalesced[-1][0] == bits and coalesced[-1][3:] == (
            strategy,
            stride,
            lanes,
            unroll,
        ):
            prev = coalesced[-1]
            coalesced[-1] = (bits, prev[1], last, strategy, stride, lanes, unroll)
        else:
            coalesced.append((bits, first, last, strategy, stride, lanes, unroll))
    return coalesced


def derive_width_rows(
    profile: str,
    element_bits: int,
    by_degree: dict[int, tuple[float, str, int, int, int]],
    best_horner: tuple[float, int, int] | None,
) -> list[tuple[int, int, int | str, str, int, int, int]]:
    if best_horner is None:
        small_lanes = default_lanes(profile, element_bits)
        small_unroll = 4
    else:
        _, small_lanes, small_unroll = best_horner
    rows: list[tuple[int, int, int | str, str, int, int, int]] = [
        (element_bits, 0, 8, "horner", 0, small_lanes, small_unroll)
    ]
    calibrated_points = [
        (degree, strategy, stride, lanes, unroll)
        for degree, (_, strategy, stride, lanes, unroll) in sorted(by_degree.items())
        if degree >= 8
    ]
    if not calibrated_points:
        return rows
    active_strategy = calibrated_points[0][1:]
    active_start = 8
    prev_degree = calibrated_points[0][0]
    for degree, strategy, stride, lanes, unroll in calibrated_points[1:]:
        candidate = (strategy, stride, lanes, unroll)
        if candidate != active_strategy:
            boundary = max(8, (prev_degree + degree) // 2)
            rows.append((element_bits, active_start, boundary, *active_strategy))
            active_start = boundary
            active_strategy = candidate
        prev_degree = degree
    rows.append((element_bits, active_start, ANY, *active_strategy))
    return coalesce_rows(rows)


def derive_throughput_rows(csv_text: str) -> dict[str, list[tuple[int, int, int | str, str, int, int, int]]]:
    by_width: dict[tuple[str, int], dict[int, tuple[float, str, int, int, int]]] = {}
    best_horner: dict[tuple[str, int], tuple[float, int, int]] = {}
    reader = csv.DictReader(line for line in io.StringIO(csv_text) if not line.startswith("#"))
    for row in reader:
        mode = row.get("mode", "")
        strategy = strategy_from_mode(mode)
        if strategy is None:
            continue
        profile = row.get("profile") or "generic"
        if profile not in PROFILE_NAMES:
            continue
        element_bits = int(row.get("element_bits") or 64)
        degree = int(row["degree"])
        lanes = int(row.get("lanes") or 4)
        unroll = int(row.get("unroll") or 4)
        ns = float(row["ns_per_eval"])
        key = (profile, element_bits)
        current = by_width.setdefault(key, {}).get(degree)
        if current is None or ns < current[0]:
            by_width[key][degree] = (ns, strategy[0], strategy[1], lanes, unroll)
        if mode == "batch_explicit_horner":
            horner_current = best_horner.get(key)
            if horner_current is None or ns < horner_current[0]:
                best_horner[key] = (ns, lanes, unroll)
    if not by_width:
        return CANONICAL_THROUGHPUT

    rows_by_profile = {profile: list(rows) for profile, rows in CANONICAL_THROUGHPUT.items()}
    for (profile, element_bits), degree_choices in sorted(by_width.items()):
        measured = derive_width_rows(
            profile, element_bits, degree_choices, best_horner.get((profile, element_bits))
        )
        existing = [row for row in rows_by_profile[profile] if row[0] != element_bits]
        rows_by_profile[profile] = sorted(existing + measured, key=lambda row: (row[0], row[1]))
    return rows_by_profile


def fmt_bound(value: int | str) -> str:
    return str(value)


def fmt_row(
    coeff: str,
    var: str,
    intent: str,
    first: int,
    last: int | str,
    strategy: str,
    stride: int,
    lanes: int,
    element_bits: int = 64,
    unroll: int | None = None,
) -> str:
    suffix = ", " + str(element_bits) + ", " + str(element_bits) + ", " + str(lanes)
    if unroll is not None:
        suffix += ", coefficient_shape::any, accuracy_class::any, " + str(unroll)
    return (
        "      tuning_row{value_category::" + coeff + ", value_category::" + var + ",\n"
        "                 storage_category::any, evaluation_intent::" + intent + ", "
        + str(first) + ", " + fmt_bound(last) + ",\n"
        "                 strategy_category::" + strategy + ", " + str(stride)
        + suffix + "}"
    )


def profile_rows(profile: str, throughput_rows: list[tuple[int, int, int | str, str, int, int, int]]) -> list[str]:
    rows: list[str] = []
    for element_bits, first, last, strategy, stride, lanes, unroll in throughput_rows:
        rows.append(fmt_row("floating_real", "floating_real", "throughput", first, last, strategy, stride, lanes, element_bits, unroll))
    for first, last, strategy, stride in LATENCY_REAL[profile]:
        rows.append(fmt_row("floating_real", "floating_real", "latency", first, last, strategy, stride, 1))
    for first, last, strategy, stride in LATENCY_COMPLEX[profile]:
        rows.append(fmt_row("complex_floating", "complex_floating", "latency", first, last, strategy, stride, 1))
    return rows


def render(
    throughput_rows: dict[str, list[tuple[int, int, int | str, str, int, int, int]]] = CANONICAL_THROUGHPUT,
) -> str:
    chunks = [
        "// Generated by benchmark/generate_profile_tuning.py.",
        "// Regenerate after running algorithm_selection_probe and batch_throughput_probe.",
        "",
    ]
    for profile_index, profile in enumerate(PROFILE_NAMES):
        rows = profile_rows(profile, throughput_rows[profile])
        chunks.append(f"template <> struct profile_tuning<target::profile::{profile}> {{")
        chunks.append("  using table = profile_table<")
        for index, row in enumerate(rows):
            chunks.append(row + (">;" if index == len(rows) - 1 else ","))
        chunks.append("};")
        if profile_index + 1 != len(PROFILE_NAMES):
            chunks.append("")
    chunks.append("")
    return "\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm-probe")
    parser.add_argument("--batch-probe")
    parser.add_argument("--batch-csv")
    parser.add_argument("--output")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    run_probe(args.algorithm_probe)
    batch_text = Path(args.batch_csv).read_text() if args.batch_csv else run_probe(args.batch_probe)
    throughput_rows = derive_throughput_rows(batch_text) if batch_text else CANONICAL_THROUGHPUT
    generated = render(throughput_rows)

    if args.check:
        inc = Path(__file__).resolve().parents[1] / "include" / "polygnition" / "detail" / "profile_tuning.inc"
        if inc.read_text() != generated:
            sys.stderr.write(
                "profile tuning include is stale; rerun benchmark/generate_profile_tuning.py > "
                "include/polygnition/detail/profile_tuning.inc\n"
            )
            raise SystemExit(1)
        return
    if args.output:
        Path(args.output).write_text(generated)
    else:
        sys.stdout.write(generated)


if __name__ == "__main__":
    main()
