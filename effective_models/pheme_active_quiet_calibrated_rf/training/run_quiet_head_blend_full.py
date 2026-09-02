"""Full LOEO evaluation of a bounded quiet-RF plus exhaustive quiet-head blend.

This is intentionally a foreground, long-running research script.  Head-group
and blend selection are made only from outer-train inner-LOEO predictions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import run_quiet_head_blend_smoke as common
from pheme_quiet_expert_calibration_common import fit_calibrators
from quiet_head_blend_score import bounded_head_blend
from quiet_head_combination_common import combinations_for, fit_score
from run_quiet_tier_hybrid_smoke import apply_calibration, calibrator_frame, curve_rows, graph_inputs


HEAD_TREES = 300
BLEND_CONFIGS = ((0.0, 0.0), (0.25, 0.10), (0.50, 0.10), (0.50, 0.20), (0.75, 0.20))
BUDGETS = (1, 3, 5, 10, 20, 50)
MAX_ALLOWED_DECLINE = 0.01


def reductions(frame: pd.DataFrame, score: str, quiet_only: bool) -> dict[int, float]:
    values = {budget: [] for budget in BUDGETS}
    for _, event in frame.groupby("validation_event", sort=True):
        if quiet_only:
            event = event.loc[event.expert.eq("quiet")]
        if event.empty:
            continue
        ranked = event.sort_values([score, "thread_id"], ascending=[False, True], kind="stable")
        total = float(event.impact.sum())
        for budget in BUDGETS:
            values[budget].append(float(ranked.head(min(budget, len(event))).impact.sum()) / total if total else 0.0)
    return {budget: float(np.mean(value)) if value else 0.0 for budget, value in values.items()}


def select_head(baseline: pd.DataFrame, heads: dict[tuple[str, ...], pd.DataFrame], outer_event: str) -> tuple[tuple[str, ...] | None, pd.DataFrame]:
    base = reductions(baseline, "baseline_raw", quiet_only=True)
    rows = []
    for groups, head in heads.items():
        joined = baseline.merge(head, on=["validation_event", "thread_id"], how="inner", validate="one_to_one")
        metric = reductions(joined, "head_raw", quiet_only=True)
        feasible = metric[1] >= base[1] and metric[3] >= base[3] and all(metric[k] >= base[k] - MAX_ALLOWED_DECLINE for k in (5, 10, 20, 50))
        rows.append({"outer_event": outer_event, "groups": "+".join(groups), "feasible": feasible, **{f"quiet_head_k{k}": metric[k] for k in BUDGETS}, **{f"quiet_delta_k{k}": metric[k] - base[k] for k in BUDGETS}})
    table = pd.DataFrame(rows)
    feasible = table.loc[table.feasible]
    if feasible.empty:
        return None, table
    chosen = feasible.sort_values(["quiet_delta_k1", "quiet_delta_k3", "quiet_delta_k50", "groups"], ascending=[False, False, False, True], kind="stable").iloc[0]
    return tuple(chosen.groups.split("+")), table


def main() -> None:
    output = common.OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_head_blend_full")
    output.mkdir(parents=True, exist_ok=False)
    combos = combinations_for("full")
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "mode": "full", "model": "active_quiet_rf_bounded_quiet_head_blend", "dataset": str(common.DATASET), "split": "eligible outer LOEO; all remaining events provide inner-LOEO OOF selection", "cutoff_seconds": 1800, "seed": common.SEED, "head_combinations": len(combos), "head_trees": HEAD_TREES, "blend_configurations": [list(x) for x in BLEND_CONFIGS], "head_selection": "quiet K1 then K3 maximize; K5/K10/K20/K50 may decline by at most 0.01", "blend_selection": "full-policy K1 then K3 maximize, subject to no K5/K10/K20/K50 decline", "research_safety": "No held-out outer outcomes select a head group, blend, gate, PCA, RF, or calibration. All features are protected strict-30 materialized inputs."}
    common.write_record(output, record)
    try:
        print("Starting full bounded quiet-head blend policy (foreground job).", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading and validating protected strict-30 inputs...", flush=True)
        graphs, _, eligible = common.load_graphs(); depth, age = common.load_safe_nodes(); offsets = common.load_node_offsets()
        frame, _, embeddings = graph_inputs(graphs); columns = common.columns_by_group(frame)
        indexed = frame.set_index("thread_id")
        if not indexed.index.is_unique:
            raise ValueError("Duplicate protected feature identity")
        all_scores: list[pd.DataFrame] = []; all_curves: list[dict] = []; head_tables: list[pd.DataFrame] = []; blend_tables: list[pd.DataFrame] = []; details = []
        print(f"[2/6] Running {len(eligible)} outer LOEO folds; each has {len(combos)} head groups...", flush=True)
        for outer_number, outer_event in enumerate(eligible, 1):
            print(f"  outer {outer_number}/{len(eligible)}: {outer_event}", flush=True)
            outer_train = [g for g in graphs if str(g.event_id) != outer_event]; held = [g for g in graphs if str(g.event_id) == outer_event]
            base_rows: list[dict] = []; head_rows = {groups: [] for groups in combos}
            inner_events = sorted({str(g.event_id) for g in outer_train})
            for inner_number, event in enumerate(inner_events, 1):
                print(f"    inner {inner_number}/{len(inner_events)}: validation={event}; heads 1/{len(combos)}", flush=True)
                train = [g for g in outer_train if str(g.event_id) != event]; validation = [g for g in outer_train if str(g.event_id) == event]
                base, _, gate = common.raw_scores(train, validation, depth, age, offsets, frame, columns, embeddings, common.SEED + outer_number * 10000 + inner_number)
                quiet = gate["quiet"]; qtrain = common.recency_values(train) >= float(gate["gate"]["threshold"])
                for g, value, is_quiet in zip(validation, base, quiet):
                    base_rows.append({"validation_event": event, "thread_id": str(g.thread_id), "expert": "quiet" if is_quiet else "active", "baseline_raw": float(value), "target": float(g.preventable_y.item()), "impact": float(g.preventable_impact.item())})
                if quiet.any():
                    train_frame = indexed.loc[common.ids(common.quiet_graphs(train, qtrain))].reset_index(); test_frame = indexed.loc[common.ids(common.quiet_graphs(validation, quiet))].reset_index()
                    for combo_number, groups in enumerate(combos, 1):
                        if combo_number % 25 == 0 or combo_number == len(combos): print(f"      heads {combo_number}/{len(combos)}", flush=True)
                        prediction = fit_score(train_frame, test_frame, groups, columns, embeddings, common.SEED + outer_number * 10000 + inner_number * 100 + combo_number, HEAD_TREES)
                        head_rows[groups].extend({"validation_event": event, "thread_id": tid, "head_raw": float(score)} for tid, score in zip(test_frame.thread_id, prediction))
            baseline = pd.DataFrame(base_rows)
            heads = {groups: pd.DataFrame(rows) for groups, rows in head_rows.items()}
            chosen_head, head_table = select_head(baseline, heads, outer_event); head_tables.append(head_table)
            baseline_oof = calibrator_frame(baseline.rename(columns={"baseline_raw": "raw_score"})[["validation_event", "thread_id", "expert", "raw_score", "target"]].to_dict(orient="records")); base_cal, _ = fit_calibrators(baseline_oof)
            base_oof = baseline_oof.copy(); base_oof["impact"] = np.expm1(base_oof.target); base_oof["score"] = apply_calibration(base_oof.raw_score.to_numpy(), base_oof.expert.eq("quiet").to_numpy(), base_cal); base_metric = reductions(base_oof, "score", False)
            configs = []
            if chosen_head is not None:
                merged = baseline.merge(heads[chosen_head], on=["validation_event", "thread_id"], how="left", validate="one_to_one")
                for weight, shift in BLEND_CONFIGS:
                    raw = merged.baseline_raw.to_numpy(np.float32).copy(); mask = merged.expert.eq("quiet").to_numpy(); raw[mask] = bounded_head_blend(raw[mask], merged.loc[mask, "head_raw"].to_numpy(), merged.loc[mask, "thread_id"].tolist(), weight, shift)
                    oof = calibrator_frame([{**row, "raw_score": float(value)} for row, value in zip(merged[["validation_event", "thread_id", "expert", "target"]].to_dict(orient="records"), raw)]); cal, _ = fit_calibrators(oof); oof["impact"] = np.expm1(oof.target); oof["score"] = apply_calibration(oof.raw_score.to_numpy(), oof.expert.eq("quiet").to_numpy(), cal); metric = reductions(oof, "score", False); feasible = all(metric[k] >= base_metric[k] - 1e-12 for k in (5, 10, 20, 50)); configs.append((weight, shift, cal, metric, feasible))
            feasible_configs = [x for x in configs if x[4]]
            selected = max(feasible_configs, key=lambda x: (x[3][1], x[3][3], x[3][50], -x[0])) if feasible_configs else None
            blend_tables.append(pd.DataFrame([{"outer_event": outer_event, "selected_head": "+".join(chosen_head) if chosen_head else "baseline", "weight": x[0], "maximum_rank_shift": x[1], "feasible": x[4], **{f"full_k{k}": x[3][k] for k in BUDGETS}} for x in configs]))
            print(f"    selected head={'+'.join(chosen_head) if chosen_head else 'baseline'}; blend={'baseline' if selected is None else selected[:2]}; scoring outer event...", flush=True)
            outer_base, outer_head, outer = common.raw_scores(outer_train, held, depth, age, offsets, frame, columns, embeddings, common.SEED + 900000 + outer_number)
            outer_quiet = outer["quiet"]; score = apply_calibration(outer_base, outer_quiet, base_cal); blend_score = score.copy()
            if selected is not None:
                _, _, cal, _, _ = selected; outer_head_values = fit_score(indexed.loc[common.ids(common.quiet_graphs(outer_train, common.recency_values(outer_train) >= float(outer["gate"]["threshold"])))].reset_index(), indexed.loc[common.ids(common.quiet_graphs(held, outer_quiet))].reset_index(), chosen_head, columns, embeddings, common.SEED + 990000 + outer_number, HEAD_TREES) if outer_quiet.any() else np.empty(0)
                raw = outer_base.copy(); raw[outer_quiet] = bounded_head_blend(outer_base[outer_quiet], outer_head_values, common.ids(common.quiet_graphs(held, outer_quiet)), selected[0], selected[1]); blend_score = apply_calibration(raw, outer_quiet, cal)
            scores = pd.DataFrame({"thread_id": common.ids(held), "event_id": [str(g.event_id) for g in held], "preventable_impact": [float(g.preventable_impact.item()) for g in held], "baseline_calibrated_score": score, "blend_calibrated_score": blend_score}); all_scores.append(scores); all_curves += curve_rows(scores, "baseline_calibrated_score", "calibrated_active_quiet_rf") + curve_rows(scores, "blend_calibrated_score", "bounded_quiet_head_blend")
            details.append({"outer_event": outer_event, "selected_head": list(chosen_head) if chosen_head else None, "selected_blend": list(selected[:2]) if selected else None}); pd.concat(all_scores).to_csv(output / "outer_scores_partial.csv", index=False); (output / "partial_result.json").write_text(json.dumps({"completed_outer_folds": details}, indent=2), encoding="utf-8")
        print("[3/6] Summarizing per-event intervention curves...", flush=True)
        scores = pd.concat(all_scores, ignore_index=True); curves = pd.DataFrame(all_curves); summary = curves.groupby(["model", "budget"], as_index=False).agg(mean_model_reduction=("model_reduction", "mean"), mean_oracle_efficiency=("oracle_efficiency", "mean"), eligible_events=("event_id", "nunique"))
        print("[4/6] Validating full-run output integrity...", flush=True)
        if scores.thread_id.duplicated().any() or not np.isfinite(scores[["baseline_calibrated_score", "blend_calibrated_score"]].to_numpy()).all(): raise ValueError("Invalid full-run score output")
        print("[5/6] Writing complete experiment evidence...", flush=True)
        scores.to_csv(output / "outer_scores.csv", index=False); curves.to_csv(output / "per_event_budget_curve.csv", index=False); summary.to_csv(output / "eligible_event_budget_summary.csv", index=False); pd.concat(head_tables).to_csv(output / "inner_head_selection.csv", index=False); pd.concat(blend_tables).to_csv(output / "inner_blend_selection.csv", index=False)
        print("[6/6] Finalizing reproducibility record...", flush=True)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "eligible_events": eligible, "fold_details": details, "output_files": ["outer_scores.csv", "per_event_budget_curve.csv", "eligible_event_budget_summary.csv", "inner_head_selection.csv", "inner_blend_selection.csv"]}); common.write_record(output, record); print(f"SUCCESS: full bounded quiet-head blend policy saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}); common.write_record(output, record); print(f"FAILURE: full bounded quiet-head blend policy preserved at: {output.resolve()}", flush=True); raise


if __name__ == "__main__":
    main()
