"""Fixed, exploratory full-run evaluation; never used in unit or profile stages."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median

import joblib

from triage_pipeline import core, methods, metrics, selection
from triage_pipeline.reporting import recommendation, render_report


def outer_groups(threads: list[core.Thread]) -> list[tuple[str, list[core.Thread], list[core.Thread]]]:
    groups = sorted({thread.group_id for thread in threads})
    return [(held, [thread for thread in threads if thread.group_id != held],
             [thread for thread in threads if thread.group_id == held]) for held in groups]


def nearest_training_groups(threads: list[core.Thread], count: int) -> set[str]:
    ordered = sorted({thread.group_id: thread.group_time for thread in threads}.items(),
                     key=lambda item: (item[1], item[0]))
    return {group for group, _ in ordered[-min(count, len(ordered)):]}


def make_rows(threads: list[core.Thread], clean: dict, cutoff_min: int,
              horizon_h: int, min_replies: int) -> list[dict]:
    cutoff, horizon = cutoff_min * 60, horizon_h * 3600
    rows = []
    for thread in threads:
        if not core.mature(thread, horizon):
            continue
        observed, _ = core.snapshot(thread, clean[thread.thread_id], cutoff)
        feature_row = core.features(thread, observed, cutoff)
        if not core.candidate(feature_row, min_replies):
            continue
        rows.append({"thread_id": thread.thread_id, "group_id": thread.group_id,
                     "cluster_id": thread.root_author_id,
                     "features": feature_row, "rules": core.rules(thread, observed, cutoff),
                     "target": core.target(thread, clean[thread.thread_id], cutoff, horizon)})
    return rows


def attach_scores(rows: list[dict], fitted) -> list[dict]:
    scored = methods.scores(rows, fitted)
    return [{**row, "scores": score} for row, score in zip(rows, scored)]


def all_metrics(rows: list[dict], fraction: float, include_model: bool = True,
                include_intervals: bool = False, progress=None, draws: int = 2000) -> dict:
    names = methods.METHODS if include_model else tuple(name for name in methods.METHODS if name != "M")
    result = {name: {"primary": metrics.grouped_metric(rows, name, fraction),
                     "top10": metrics.grouped_metric(rows, name, fraction, fixed_k=10)} for name in names}
    if include_intervals and rows:
        intervals = metrics.method_intervals(rows, names, fraction, draws=draws, progress=progress)
        for name in names:
            result[name]["primary_interval"] = intervals[name]
    return result


def select_from_scored(selection_rows: list[dict], train_count: int,
                       test_valid_groups: int, fraction: float, progress=None,
                       draws: int = 2000) -> tuple[dict, dict]:
    selection_metrics = {name: metrics.grouped_metric(selection_rows, name, fraction)
                         for name in methods.METHODS}
    valid = selection_metrics["U"]["valid_groups"]
    preliminary = selection.choose(selection_metrics, train_count, valid, test_valid_groups, None)
    ci = None
    if preliminary["status"] != "insufficient_data":
        best = max(selection.RULES,
                   key=lambda name: selection_metrics[name]["mean_capture"])
        ci = metrics.paired_cluster_ci(selection_rows, "M", best, fraction, draws=draws, progress=progress)
    choice = selection.choose(selection_metrics, train_count, valid, test_valid_groups, ci)
    return choice, selection_metrics


def _save_model(out: Path, label: str, fitted, train_rows: list[dict], config_hash: str) -> str:
    path = out / f"model_{label}.joblib"
    joblib.dump({"models": fitted, "features": core.FEATURES,
                "train_groups": sorted({row["group_id"] for row in train_rows}),
                "train_count": len(train_rows), "config_sha256": config_hash}, path)
    return path.name


def subset_metrics(rows: list[dict], flags: dict[str, dict[str, bool]] | None,
                   fraction: float) -> dict:
    names = sorted({name for per_thread in (flags or {}).values() for name in per_thread})
    return {name: {"candidate_threads": sum(bool(flags.get(row["thread_id"], {}).get(name)) for row in rows),
                   "metrics": all_metrics([row for row in rows if flags.get(row["thread_id"], {}).get(name)], fraction)}
            for name in names}


def segment_sizes(segments: dict, candidates: dict) -> dict:
    result = {}
    for name, threads in segments.items():
        counts = Counter(thread.group_id for thread in threads)
        sizes = sorted(counts.values())
        result[name] = {"groups": len(sizes), "threads": len(threads),
                        "candidates": len(candidates.get(name, [])),
                        "group_size_min": sizes[0] if sizes else None,
                        "group_size_median": median(sizes) if sizes else None,
                        "group_size_max": sizes[-1] if sizes else None,
                        "groups_below_one_percent_median": sum(size < median(sizes) * .01 for size in sizes) if sizes else 0}
    return result


def time_run(out: Path, rec: dict, threads: list[core.Thread], clean: dict, profile: dict,
             config_hash: str, subsets: dict[str, dict[str, bool]] | None = None,
             draws: int = 2000) -> dict:
    if profile["status"] != "complete":
        return {"status": "insufficient_data", "reason": profile["reason"],
                "segment_sizes": segment_sizes(core.raw_time_groups(threads)[0], {})}
    horizon, cutoff = profile["H_hours"], profile["T_minutes"]
    segments, split_report = core.time_segments(threads, horizon * 3600)
    rows = {name: make_rows(value, clean, cutoff, horizon, rec["config"]["min_replies"])
            for name, value in segments.items()}
    if not rows["train"]:
        return {"status": "insufficient_data", "reason": "no_training_candidates", "split": split_report,
                "candidate_counts": {name: len(value) for name, value in rows.items()},
                "segment_sizes": segment_sizes(segments, rows)}
    fraction = rec["config"]["top_fraction"]
    fitted = methods.fit_models(rows["train"],
        progress=lambda done, total: print(f"    fixed model seed {done}/{total}", flush=True))
    scored = {name: attach_scores(value, fitted) for name, value in rows.items()}
    test_valid = metrics.grouped_metric(scored["test"], "U", fraction)["valid_groups"]
    choice, selection_metrics = select_from_scored(scored["selection"], len(rows["train"]),
        test_valid, fraction, progress=lambda done, total: print(f"    selection bootstrap {done}/{total}", flush=True), draws=draws)
    test_metrics = all_metrics(scored["test"], fraction, include_intervals=True,
        progress=lambda done, total: print(f"    test bootstrap {done}/{total}", flush=True), draws=draws)
    subset_report = subset_metrics(scored["test"], subsets, fraction)
    learning = []
    group_count = len({thread.group_id for thread in segments["train"]})
    for completed, count in enumerate((1, 2, 4, 7, group_count), 1):
        chosen = nearest_training_groups(segments["train"], count)
        subset = [row for row in rows["train"] if row["group_id"] in chosen]
        if subset:
            partial = methods.fit_models(subset)
            learning.append({"groups_requested": count, "groups_used": len(chosen),
                "train_threads": len(subset),
                "selection_M": metrics.grouped_metric(attach_scores(rows["selection"], partial), "M", fraction),
                "test_M": metrics.grouped_metric(attach_scores(rows["test"], partial), "M", fraction)})
        print(f"    learning-curve size {completed}/5", flush=True)
    if choice["chosen"] == "M":
        rec["results"].setdefault("models", []).append(_save_model(out, "time", fitted, rows["train"], config_hash))
    return {"status": choice["status"], "profile": {"H_hours": horizon, "T_minutes": cutoff},
            "split": split_report, "candidate_counts": {key: len(value) for key, value in rows.items()},
            "selection_metrics": selection_metrics, "selection": choice,
            "test_metrics": test_metrics, "subset_test_metrics": subset_report,
            "segment_sizes": segment_sizes(segments, rows), "learning_curve": learning,
            "selected_test_metric": test_metrics[choice["chosen"]]["primary"] if choice["chosen"] else None}


def group_out_run(out: Path, rec: dict, threads: list[core.Thread], clean: dict,
                  profile: dict, config_hash: str, subsets: dict[str, dict[str, bool]] | None = None,
                  draws: int = 2000) -> dict:
    fraction = rec["config"]["top_fraction"]
    folds = []
    outer = outer_groups(threads)
    test_valid_total = 0
    for held, _, test in outer:
        parameters = profile["folds"][held]
        if parameters["status"] == "complete":
            test_rows = make_rows(test, clean, parameters["T_minutes"], parameters["H_hours"],
                                  rec["config"]["min_replies"])
            test_valid_total += int(bool(test_rows and sum(row["target"] for row in test_rows) > 0))
    for completed, (held, train, test) in enumerate(outer, 1):
        parameters = profile["folds"][held]
        if parameters["status"] != "complete":
            folds.append({"held_group": held, "status": "insufficient_data", "reason": parameters["reason"],
                          "segment_sizes": segment_sizes({"train": train, "selection": train, "test": test}, {})})
            continue
        cutoff, horizon = parameters["T_minutes"], parameters["H_hours"]
        train_rows = make_rows(train, clean, cutoff, horizon, rec["config"]["min_replies"])
        test_rows = make_rows(test, clean, cutoff, horizon, rec["config"]["min_replies"])
        if not train_rows:
            folds.append({"held_group": held, "status": "insufficient_data", "reason": "no_training_candidates",
                          "segment_sizes": segment_sizes({"train": train, "selection": train, "test": test},
                                                         {"train": train_rows, "test": test_rows})})
            continue
        inner_scored = []
        for inner_done, (inner_held, inner_train, inner_test) in enumerate(outer_groups(train), 1):
            inner_train_rows = make_rows(inner_train, clean, cutoff, horizon, rec["config"]["min_replies"])
            inner_test_rows = make_rows(inner_test, clean, cutoff, horizon, rec["config"]["min_replies"])
            if inner_train_rows:
                inner_models = methods.fit_models(inner_train_rows)
                inner_scored.extend(attach_scores(inner_test_rows, inner_models))
            print(f"    outer {completed}/{len(outer)} inner {inner_done}/{len(outer)-1}", flush=True)
        choice, selection_metrics = select_from_scored(inner_scored, len(train_rows), test_valid_total,
            fraction, progress=lambda done, total: print(f"    outer {completed}/{len(outer)} bootstrap {done}/{total}", flush=True), draws=draws)
        fitted = methods.fit_models(train_rows)
        scored_test = attach_scores(test_rows, fitted)
        test_metrics = all_metrics(scored_test, fraction,
            include_intervals=bool(test_rows),
            progress=lambda done, total: print(f"    outer {completed}/{len(outer)} test bootstrap {done}/{total}", flush=True), draws=draws)
        subset_report = subset_metrics(scored_test, subsets, fraction)
        if choice["chosen"] == "M":
            rec["results"].setdefault("models", []).append(_save_model(out, f"fold_{completed}", fitted,
                                                                       train_rows, config_hash))
        folds.append({"held_group": held, "status": choice["status"],
                      "H_hours": horizon, "T_minutes": cutoff, "train_candidates": len(train_rows),
                      "selection_metrics": selection_metrics, "selection": choice,
                      "test_metrics": test_metrics, "subset_test_metrics": subset_report,
                      "segment_sizes": segment_sizes({"train": train, "selection": train, "test": test},
                                                     {"train": train_rows, "selection": inner_scored,
                                                      "test": test_rows}),
                      "selected_test_metric": test_metrics[choice["chosen"]]["primary"] if choice["chosen"] else None})
        print(f"    outer-fold evaluation {completed}/{len(outer)}", flush=True)
    fixed = {name: [] for name in methods.METHODS}
    selected = []
    chosen_counts = Counter()
    for fold in folds:
        if "test_metrics" not in fold:
            continue
        for name in methods.METHODS:
            value = fold["test_metrics"][name]["primary"]["mean_capture"]
            if value is not None:
                fixed[name].append(value)
        chosen = fold["selection"]["chosen"]
        chosen_counts[chosen or "no_recommendation"] += 1
        if chosen:
            value = fold["test_metrics"][chosen]["primary"]["mean_capture"]
            if value is not None:
                selected.append(value)
    return {"status": "insufficient_data" if not folds or any(fold["status"] == "insufficient_data" for fold in folds) else "complete",
            "time_forward": False,
            "warning": "non-time-forward; cannot represent online performance",
            "folds": folds, "outer_test_valid_groups": test_valid_total,
            "selected_method_test_mean": sum(selected) / len(selected) if selected else None,
            "fixed_method_test_mean": {name: sum(values) / len(values) if values else None
                                       for name, values in fixed.items()},
            "selected_method_counts": dict(chosen_counts)}


def run_full(out: Path, rec: dict) -> None:
    from triage_pipeline.run_pipeline import cache_read, cleaned, latest_cache_record, save_json, sha256
    print("[2/5] loading verified consistency cache and frozen profile", flush=True)
    threads, replies, subsets, source = cache_read(latest_cache_record())
    rec["inputs"]["cache"] = source
    root = out.parents[1]
    target = __import__("os").environ.get("PROFILE_RUN")
    if target:
        path = Path(target)
        profile_path = path / "run_record.json" if path.is_dir() else path
    else:
        options = sorted(root.glob(f"profile_*/{rec['platform']}/run_record.json"), reverse=True)
        profile_path = next(path for path in options if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete")
    profile_record = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile_record["inputs"]["specification_sha256"] != rec["inputs"]["specification_sha256"]:
        raise ValueError("profile specification changed")
    if profile_record["config"] != rec["config"]:
        raise ValueError("profile configuration changed")
    rec["inputs"]["profile_record"] = {"path": str(profile_path.resolve()), "sha256": sha256(profile_path)}
    profile = dict(profile_record["results"]["profile"])
    no_nested = not any(reply.parent_id != reply.thread_id for reply in replies)
    if rec["config"]["split_mode"] == "time":
        profile["no_nested_replies"] = no_nested
    else:
        profile["folds"] = {name: {**value, "no_nested_replies": no_nested}
                            for name, value in profile.get("folds", {}).items()}
    clean, counts = cleaned(threads, replies)
    rec["results"]["cleaning_counts"] = counts
    print("[3/5] fixed fitting, nested selection and final evaluation", flush=True)
    config_path = Path(__file__).resolve().parent / "configs" / f"{rec['platform']}.json"
    config_hash = sha256(config_path)
    if rec["config"]["split_mode"] == "time":
        result = time_run(out, rec, threads, clean, profile, config_hash, subsets)
    else:
        result = group_out_run(out, rec, threads, clean, profile, config_hash, subsets)
    rec["results"]["evaluation"] = result
    print("[4/5] writing recommendation and one-page report", flush=True)
    save_json(out / "recommendation.json", recommendation(result, rec["config"]["split_mode"]))
    lines = render_report(result, rec["results"]["cleaning_counts"], profile,
                          rec["config"]["split_mode"])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


