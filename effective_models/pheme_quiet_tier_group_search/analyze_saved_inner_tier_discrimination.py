"""Analyze saved inner-LOEO tier-discrimination scores without re-fitting.

The analysis intentionally uses only ``inner_tier_macro_f1``.  Unlike the
previous top-K overlap metric, this class-prediction score is unaffected by
row-order resolution among equal high-tier probabilities.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd

from strict30_protected_feature_registry import FEATURE_GROUPS


BASE = Path(__file__).resolve().parent
SOURCE_RUN = BASE / "experiments" / "20260902_031535_048090_strict30_clean_quiet_tier_full"
OUT_ROOT = BASE / "experiments"
MODEL_GROUPS = FEATURE_GROUPS + ("source_reply_embedding",)


def _has_group(groups: str, group: str) -> bool:
    return group in groups.split("+")


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_saved_inner_tier_discrimination_analysis")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "post-hoc feature-group discrimination analysis from saved inner LOEO tier macro-F1 scores; no fitting",
        "source_run": str(SOURCE_RUN.resolve()),
        "metric": "inner_tier_macro_f1",
        "research_note": "Does not use the row-order-sensitive top-K overlap metric and does not change the source experiment.",
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting saved inner tier-discrimination analysis.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading seven saved inner-LOEO selection tables...", flush=True)
        files = sorted(SOURCE_RUN.glob("inner_selection_*.csv"))
        if len(files) != 7:
            raise ValueError(f"Expected seven inner selection files, found {len(files)}")
        parts = []
        for path in files:
            outer_event = path.stem.removeprefix("inner_selection_")
            frame = pd.read_csv(path)
            if len(frame) != 255 or frame.config.duplicated().any():
                raise ValueError(f"{path.name}: expected 255 unique configurations")
            frame["outer_event"] = outer_event
            parts.append(frame)
        scores = pd.concat(parts, ignore_index=True)
        print("[2/4] Computing cross-outer configuration stability...", flush=True)
        config_summary = (scores.groupby(["config", "groups", "feature_count"], as_index=False)
                          .agg(mean_inner_tier_macro_f1=("inner_tier_macro_f1", "mean"),
                               std_inner_tier_macro_f1=("inner_tier_macro_f1", "std"),
                               min_inner_tier_macro_f1=("inner_tier_macro_f1", "min"),
                               max_inner_tier_macro_f1=("inner_tier_macro_f1", "max"))
                          .sort_values(["mean_inner_tier_macro_f1", "std_inner_tier_macro_f1", "feature_count", "config"], ascending=[False, True, True, True], kind="stable"))
        print("[3/4] Measuring per-event best tier classifier and feature-group contribution...", flush=True)
        best_rows = []
        contribution_rows = []
        for outer_event, frame in scores.groupby("outer_event", sort=True):
            ranked = frame.sort_values(["inner_tier_macro_f1", "feature_count", "config"], ascending=[False, True, True], kind="stable")
            best_rows.append(ranked.iloc[0].to_dict())
            for group in MODEL_GROUPS:
                with_group = frame.loc[frame.groups.map(lambda value: _has_group(value, group))]
                without_group = frame.loc[~frame.groups.map(lambda value: _has_group(value, group))]
                contribution_rows.append({
                    "outer_event": outer_event, "feature_group": group,
                    "best_with_group_f1": float(with_group.inner_tier_macro_f1.max()),
                    "best_without_group_f1": float(without_group.inner_tier_macro_f1.max()),
                })
        best = pd.DataFrame(best_rows).sort_values("outer_event")
        contribution = pd.DataFrame(contribution_rows)
        contribution["best_f1_delta_when_available"] = contribution.best_with_group_f1 - contribution.best_without_group_f1
        group_summary = (contribution.groupby("feature_group", as_index=False)
                         .agg(mean_best_f1_delta=("best_f1_delta_when_available", "mean"),
                              median_best_f1_delta=("best_f1_delta_when_available", "median"),
                              events_positive_delta=("best_f1_delta_when_available", lambda values: int((values > 1e-12).sum())),
                              events_negative_delta=("best_f1_delta_when_available", lambda values: int((values < -1e-12).sum())),
                              max_best_with_group_f1=("best_with_group_f1", "max"))
                         .sort_values(["mean_best_f1_delta", "events_positive_delta", "feature_group"], ascending=[False, False, True], kind="stable"))
        print("[4/4] Writing analysis tables and run record...", flush=True)
        config_summary.to_csv(output / "configuration_tier_f1_stability.csv", index=False)
        best.to_csv(output / "per_outer_best_tier_f1_configuration.csv", index=False)
        contribution.to_csv(output / "per_outer_group_tier_f1_contribution.csv", index=False)
        group_summary.to_csv(output / "group_tier_f1_contribution_summary.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "metrics": {"outer_events": len(best), "configurations": len(config_summary), "mean_per_outer_best_inner_tier_f1": float(best.inner_tier_macro_f1.mean())}, "output_files": ["configuration_tier_f1_stability.csv", "per_outer_best_tier_f1_configuration.csv", "per_outer_group_tier_f1_contribution.csv", "group_tier_f1_contribution_summary.csv"]})
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: saved inner tier-discrimination analysis saved to: {output.resolve()}", flush=True)
    except Exception as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: tier-discrimination analysis preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
