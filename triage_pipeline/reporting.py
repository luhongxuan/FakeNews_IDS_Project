"""Platform-neutral presentation of preregistered results."""
from __future__ import annotations

import json


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _choice(choice: dict) -> dict:
    if not choice:
        return {"status": "insufficient_data", "selected_method": None,
                "reason": "profile_or_candidate_data_insufficient"}
    gates = choice.get("reason") if isinstance(choice.get("reason"), dict) else None
    point = choice.get("model_minus_best_rule_point")
    ci = choice.get("model_minus_best_rule_ci")
    return {"status": choice["status"], "selected_method": choice.get("chosen"),
            "reason": choice.get("reason") if choice["status"] == "insufficient_data" else
                      ("model_thresholds_passed" if choice.get("chosen") == "M" else "best_rule_retained"),
            "best_rule": choice.get("best_rule"), "model_minus_best_rule_point": point,
            "model_minus_best_rule_interval": ci,
            "thresholds": {"train_candidates_at_least_500": choice["status"] == "selected" if gates is None else not gates["train_candidates_lt_500"],
                           "selection_valid_groups_at_least_3": choice["status"] == "selected" if gates is None else not gates["selection_groups_lt_3"],
                           "test_valid_groups_at_least_3": choice["status"] == "selected" if gates is None else not gates["test_groups_lt_3"],
                           "model_gain_at_least_0_02": None if point is None else point >= .02,
                           "interval_lower_above_zero": None if ci is None else ci.get("lower") is not None and ci["lower"] > 0},
            "temporary_rules": choice.get("temporary_rules")}


def recommendation(result: dict, split_mode: str) -> dict:
    if split_mode == "time":
        choice = _choice(result.get("selection", {}))
        if result["status"] == "insufficient_data" and not result.get("selection"):
            choice["reason"] = result.get("reason", "profile_insufficient_data")
        return choice
    return {"status": result["status"], "per_fold": {
        fold["held_group"]: _choice(fold.get("selection", {})) | {
            "reason": fold.get("reason") if fold.get("selection") is None else _choice(fold["selection"])["reason"]}
        for fold in result.get("folds", [])},
        "selected_method_counts": result.get("selected_method_counts", {})}


def _metric_table(items: dict) -> list[str]:
    lines = ["| Method | Primary capture | 95% interval | Fixed top 10 | Spearman | Valid groups |",
             "|---|---:|---|---:|---:|---:|"]
    for name, item in items.items():
        primary = item["primary"]
        interval = item.get("primary_interval")
        bounds = f"[{interval['lower']}, {interval['upper']}]" if interval else "unavailable"
        lines.append(f"| {name} | {primary['mean_capture']} | {bounds} | "
                     f"{item['top10']['mean_capture']} | {primary['mean_spearman']} | "
                     f"{primary['valid_groups']} |")
    return lines


def _profile_lines(profile: dict) -> list[str]:
    lines = [f"H: {profile.get('H_hours')} h; T: {profile.get('T_minutes')} min; "
             f"L: {profile.get('L_seconds')} s; analysis threads: {profile.get('analysis_threads')}",
             f"Zero-reply fraction: {profile.get('zero_reply_fraction')}; "
             f"later-reply share from zero-reply threads: {profile.get('zero_reply_future_other_share')}",
             "| Candidate T (min) | Candidate threads | Later-reply coverage | Remaining share |",
             "|---:|---:|---:|---:|"]
    curve = profile.get("T_curve", {})
    for t, item in curve.items():
        lines.append(f"| {t} | {item['candidate_threads']} | {item['candidate_coverage']} | {item['remaining_share']} |")
    shares = sorted((item["remaining_share"] for item in curve.values() if item["remaining_share"] is not None), reverse=True)
    if len(shares) >= 2 and shares[0] - shares[1] < .01:
        lines.append("T 選擇接近平手")
    lines.append(f"Snapshot cleaning counts at T: {_json(profile.get('snapshot_cleaning_counts_at_T', {}))}")
    if profile.get("split"):
        lines.append(f"Boundary drops and segment starts: {_json(profile['split'])}")
    if profile.get("segments"):
        lines.append(f"Profile segment and maturity counts: {_json(profile['segments'])}")
    return lines


def _size_lines(sizes: dict) -> list[str]:
    lines = ["| Segment | Groups | Threads | Candidates | Group size min / median / max | Groups below 1% median |",
             "|---|---:|---:|---:|---|---:|"]
    for name, item in sizes.items():
        lines.append(f"| {name} | {item['groups']} | {item['threads']} | {item['candidates']} | "
                     f"{item['group_size_min']} / {item['group_size_median']} / {item['group_size_max']} | "
                     f"{item['groups_below_one_percent_median']} |")
    return lines


def _limits(profiles: list[dict], result: dict, split_mode: str) -> list[str]:
    limits = []
    if any(p.get("unknown_followup") for p in profiles):
        limits += ["Follow-up length unknown; target completeness cannot be verified."]
    if any(p.get("possibly_truncated_target") for p in profiles):
        limits += ["Target may be truncated."]
    if split_mode != "time":
        limits += ["非時間前向，不能代表上線表現。"]
    if any(p.get("no_nested_replies") for p in profiles):
        limits += ["No nested replies; depth features are constant."]
    sizes = result.get("segment_sizes", {})
    sizes = list(sizes.values()) + [item for fold in result.get("folds", []) for item in fold.get("segment_sizes", {}).values()]
    if any(item.get("groups_below_one_percent_median", 0) for item in sizes):
        limits += ["Some groups are below 1% of their segment's median size."]
    if result.get("status") == "insufficient_data" or any(f.get("status") == "insufficient_data" for f in result.get("folds", [])):
        limits += ["Insufficient data for at least one selection."]
    return limits


def render_report(result: dict, cleaning: dict, profile: dict, split_mode: str) -> list[str]:
    lines = ["# Propagation triage development report", "", f"Status: {result['status']}", "",
             "This pipeline ranks later other-author replies. It does not assess truth, harm, or intervention effects.",
             "Development test metrics are exploratory.", "", "## Cleaning", "", _json(cleaning), ""]
    profiles = [profile] if split_mode == "time" else list(profile.get("folds", {}).values())
    if split_mode == "time":
        lines += ["## Profile and coverage", "", *_profile_lines(profile), "", "## Segment sizes", "",
                  *_size_lines(result.get("segment_sizes", {})), "", "## Recommendation", "",
                  f"Selected method: {result.get('selection', {}).get('chosen')}; reason: {recommendation(result, split_mode).get('reason')}",
                  _json(recommendation(result, split_mode)), "", "## Test metrics", "",
                  *_metric_table(result.get("test_metrics", {})), "", "## Report-only subsets", "",
                  _json(result.get("subset_test_metrics", {})), "", "## Training-size curve", "",
                  _json(result.get("learning_curve", []))]
    else:
        lines += ["## Outer-fold summary", "", f"Selected-method mean: {result.get('selected_method_test_mean')}",
                  f"Fixed-method means: {_json(result.get('fixed_method_test_mean', {}))}",
                  f"Selected-method counts: {_json(result.get('selected_method_counts', {}))}"]
        for fold in result.get("folds", []):
            p = profile.get("folds", {}).get(fold["held_group"], {})
            lines += ["", f"### Fold {fold['held_group']}", "", *_profile_lines(p), "",
                      *_size_lines(fold.get("segment_sizes", {})), "",
                      f"Selected method and reason: {_json(_choice(fold.get('selection', {})))}",
                      f"Selected-method primary metric: {_json(fold.get('selected_test_metric'))}",
                      "", *_metric_table(fold.get("test_metrics", {})), "",
                      f"Report-only subsets: {_json(fold.get('subset_test_metrics', {}))}"]
    lines += ["", "## Limitations", "", *[f"- {item}" for item in _limits(profiles, result, split_mode)]]
    return lines
