#!/usr/bin/env python3
"""Aggregate BNN UQ metrics across seeds and backbones."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

BNN_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BNN_DIR / "results"

SUMMARY_METRICS = ("picp", "pinaw", "cwc")


def collect_per_seed_files(split: str, backbone: str | None, include_quicktest: bool):
    pattern = re.compile(r"^seed_(\d+)(_quicktest)?$")
    files = []

    backbone_dirs = (
        [RESULTS_DIR / backbone]
        if backbone
        else sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())
    )

    for backbone_dir in backbone_dirs:
        if not backbone_dir.exists():
            continue
        for seed_dir in sorted(backbone_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            match = pattern.match(seed_dir.name)
            if not match:
                continue
            if match.group(2) and not include_quicktest:
                continue

            metrics_path = seed_dir / f"bnn_uq_metrics_{split}.csv"
            if metrics_path.exists():
                files.append(metrics_path)

    return files


def load_per_seed_metrics(files: list[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()

    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def build_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    if per_seed_df.empty:
        return pd.DataFrame()

    metric_cols = [
        col
        for col in per_seed_df.columns
        if any(col.startswith(f"{metric}_") for metric in SUMMARY_METRICS)
    ]
    if not metric_cols:
        return pd.DataFrame()

    long_df = per_seed_df.melt(
        id_vars=["backbone", "seed"],
        value_vars=metric_cols,
        var_name="metric_coverage",
        value_name="value",
    )
    long_df[["metric", "coverage_pct"]] = long_df["metric_coverage"].str.extract(
        r"^(picp|pinaw|cwc|mpiw)_(\d+)$"
    )
    long_df = long_df.dropna(subset=["metric", "coverage_pct"])
    long_df["coverage"] = long_df["coverage_pct"].astype(int) / 100.0
    long_df = long_df[long_df["metric"].isin(SUMMARY_METRICS)]

    grouped = (
        long_df.groupby(["backbone", "coverage", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary = grouped.pivot_table(
        index=["backbone", "coverage"],
        columns="metric",
        values=["mean", "std", "count"],
    )
    summary.columns = [f"{stat}_{metric}" for stat, metric in summary.columns]
    summary = summary.reset_index()

    count_cols = [col for col in summary.columns if col.startswith("count_")]
    if count_cols:
        summary["n_seeds"] = summary[count_cols].max(axis=1).astype(int)
        summary = summary.drop(columns=count_cols)

    ordered_cols = ["backbone", "coverage", "n_seeds"]
    for metric in SUMMARY_METRICS:
        ordered_cols.extend([f"{metric}_mean", f"{metric}_std"])
    ordered_cols = [col for col in ordered_cols if col in summary.columns]
    return summary[ordered_cols].sort_values(["backbone", "coverage"]).reset_index(drop=True)


def merge_with_existing(df: pd.DataFrame, output_path: Path, backbone: str | None) -> pd.DataFrame:
    if not backbone or not output_path.exists() or df.empty:
        return df

    existing = pd.read_csv(output_path)
    if existing.empty:
        return df

    if "backbone" not in existing.columns:
        return df

    remaining = existing[existing["backbone"] != backbone]
    return pd.concat([remaining, df], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Aggregate BNN UQ metrics across seeds")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--backbone", default=None, help="Filter to a single backbone")
    parser.add_argument(
        "--include-quicktest",
        action="store_true",
        help="Include seed_*_quicktest directories",
    )
    args = parser.parse_args()

    files = collect_per_seed_files(args.split, args.backbone, args.include_quicktest)
    if not files:
        print(f"No per-seed metrics found for split={args.split}")
        return

    per_seed_df = load_per_seed_metrics(files)
    summary_df = build_summary(per_seed_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_path = RESULTS_DIR / f"bnn_per_seed_{args.split}.csv"
    summary_path = RESULTS_DIR / f"bnn_summary_{args.split}.csv"

    per_seed_df = merge_with_existing(per_seed_df, per_seed_path, args.backbone)
    summary_df = merge_with_existing(summary_df, summary_path, args.backbone)

    per_seed_df.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Loaded {len(files)} per-seed metric file(s)")
    print(f"Wrote per-seed table: {per_seed_path}")
    print(f"Wrote summary table:  {summary_path}")


if __name__ == "__main__":
    main()
