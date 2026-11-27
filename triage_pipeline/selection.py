"""Preregistered method choice from selection data only."""
from __future__ import annotations

RULES = ("U", "R1", "R2")


def choose(selection_metrics: dict, train_candidates: int, selection_valid_groups: int,
           test_valid_groups: int, model_minus_rule_ci: dict | None) -> dict:
    available_rules = {name: selection_metrics[name]["mean_capture"] for name in RULES
                       if selection_metrics.get(name, {}).get("mean_capture") is not None}
    if train_candidates < 500 or selection_valid_groups < 3 or test_valid_groups < 3:
        return {"status": "insufficient_data", "chosen": None,
                "temporary_rules": available_rules if selection_valid_groups >= 1 else None,
                "reason": {"train_candidates_lt_500": train_candidates < 500,
                           "selection_groups_lt_3": selection_valid_groups < 3,
                           "test_groups_lt_3": test_valid_groups < 3}}
    if not available_rules:
        return {"status": "insufficient_data", "chosen": None, "temporary_rules": None,
                "reason": "no_valid_rule_group"}
    best_rule = max(RULES, key=lambda name: available_rules.get(name, float("-inf")))
    model_point = selection_metrics["M"]["mean_capture"]
    rule_point = available_rules[best_rule]
    if model_point is None:
        return {"status": "insufficient_data", "chosen": None, "temporary_rules": available_rules,
                "reason": "model_metric_unavailable"}
    adopt_model = (model_minus_rule_ci is not None and model_minus_rule_ci.get("lower") is not None and
                   model_minus_rule_ci["lower"] > 0 and model_point - rule_point >= .02)
    return {"status": "selected", "chosen": "M" if adopt_model else best_rule,
            "best_rule": best_rule, "model_minus_best_rule_point": model_point - rule_point,
            "model_minus_best_rule_ci": model_minus_rule_ci}
