# PROJECT_HANDOFF_20260831.md

## Purpose

This handoff records the model-result hierarchy and later author-aware PHEME
diagnostics completed after `PROJECT_HANDOFF_20260829.md`. It does not replace
or alter historical results.

The project remains an early detection / early intervention study: at each
cutoff, only information available at or before that time may enter features,
event context, or model inputs.

## Result hierarchy

“Best” must be qualified by evaluation setting.

| Setting | Best reported result | Meaning |
|---|---:|---|
| Oracle-rumour impact ranking | **0.1438 CRR@10** | Rank known rumour threads by preventable impact. |
| Latest author-aware partial-observation pipeline | 0.13849 CRR@10 | New pipeline with account, strict-past context, and discourse features. |
| End-to-end gate plus ranker | **0.0945 CRR@10** | Rank all observed threads without true rumour status. |

Oracle-rumour ranking and end-to-end evaluation have different candidate pools;
they must not be compared as though they were the same task.

## A. Highest saved oracle-rumour impact ranker

The highest reported result remains the strict 30-minute nested
text-augmented Random Forest:

| Method | Mean CRR@10 on seven eligible events |
|---|---:|
| **Nested text-augmented RF** | **0.1438** |
| Size baseline | 0.0977 |
| Random baseline | 0.0232 |

Result:

`graphsage_intervention_5/experiments/20260829_112412_731271_nested_text_impact_ranker_rf/result.json`

Protocol:

- strict 30-minute cutoff;
- nested Leave-One-Event-Out evaluation;
- RandomForestRegressor: 300 trees, `max_depth=8`, `min_samples_leaf=3`,
  `max_features=0.8`, seed 42;
- inner validation chooses propagation-only features or propagation plus
  train-fold-only PCA of frozen source/reply RoBERTa embeddings;
- budget: ten oracle-rumour threads per event;
- official mean excludes events with fewer than 100 candidate rumours.

Saved simulation impact over the seven eligible events:

| Policy | Blocked future propagation nodes/actions |
|---|---:|
| Nested text-augmented RF | **1,421** |
| Early-size baseline | 797 |
| Random selection | 204 |

This is a counterfactual simulation under the strong policy that all observed
nodes in each selected thread can be completely intervened on. A blocked
node/action is not a count of confirmed real-world people protected.

Nested validation selected text-augmented features only for Ferguson,
Ottawa-shooting, and Prince-Toronto; it selected propagation-only features for
the remaining eligible events. Text is conditionally useful, not a demonstrated
universal cross-event feature.

## B. Latest `graphsage_intervention_7` author-aware candidate

The later versioned partial-observation pipeline produced 6,341 PHEME records
at a strict 30-minute cutoff. It uses tree-referenced raw PHEME actions at or
before the cutoff, strictly earlier user history, and versioned derived
artifacts.

Nested variants:

1. role-aware base features;
2. role-aware base plus strict-past same-event context;
3. role-aware base plus strict-past context plus early discourse/semantic
   response features.

| Method | Mean CRR@10 on seven eligible events |
|---|---:|
| Nested role + strict-past context + discourse selection | **0.13849** |
| Fixed role-aware baseline | 0.12046 |

Results:

- `graphsage_intervention_7/experiments/20260830_212959_019175_nested_discourse_context_selection/result.json`
- `graphsage_intervention_7/experiments/20260830_151132_127932_author_role_ablation/result.json`

The selected Top-10 threads account for 1,378 future propagation nodes/actions
out of 11,640 preventable nodes/actions pooled across the seven eligible events
(11.84% pooled). The official event-equal metric is macro CRR@10 of 0.13849.
The nested selector chose the context-plus-discourse variant in every eligible
outer event.

This is the strongest candidate within the new author-aware pipeline, but it
does **not** exceed historical 0.1438. Snapshot construction, thread inclusion,
and feature definitions differ, so the 0.0053 difference is not a formal
same-data head-to-head comparison. Do not claim a new project state of the art.

## C. End-to-end deployment-oriented result

The relevant evaluated flow without oracle rumour status is:

```text
all observed threads
-> early text rumour gate
-> nested impact ranker
-> select ten threads
-> intervene on all observed nodes
```

| Setting | Mean CRR@10 on seven eligible events |
|---|---:|
| Text gate + nested ranker | **0.0945** |
| No gate; rank all observed threads | 0.0682 |
| Perfect true-rumour gate + ranker | 0.1453 |

Result:

`graphsage_intervention_5/experiments/20260829_113832_479907_nested_text_gate_text_ranker_cascade/result.json`

The gate improves all-thread ranking, but it still loses high-impact rumours
and admits non-rumours. The gap to the perfect-gate reference identifies early
candidate filtering as a major deployment bottleneck.

## D. 60-minute replication and rank-change diagnosis

The author-aware 60-minute nested result is 0.13520:

`graphsage_intervention_7/experiments/20260831_003925_210106_nested_discourse_context_selection_60min_v2/result.json`

It is a separate operational cutoff, not a direct numeric comparison with the
30-minute result. `preventable_impact` counts only future impact remaining
after intervention, so later intervention can only reduce or preserve the
remaining target. In the paired artifacts, 3,438 of 6,341 threads had lower
impact at 60 minutes and none had higher impact.

The post-hoc rank-change diagnostic found substantial Top-10 churn. Replayed
models were dominated by recent activity / recency
(`seconds_since_last_action`), not by a stable enquiry-pattern signal. Late
activity helped in some events and harmed ranking in others, so it is not a
universally transferable preventable-impact signal.

Diagnostic artifact:

`graphsage_intervention_7/experiments/20260831_011201_253882_diagnose_30_to_60_rank_changes/`

## Safe reporting language

> Under the historical strict 30-minute oracle-rumour protocol, the best saved
> nested text-augmented RF achieves mean CRR@10 of 0.1438 across seven eligible
> PHEME events. With a budget of ten threads per event, its simulation captures
> 1,421 future propagation nodes/actions, compared with 797 for an early-size
> baseline. The result is conditional on oracle rumour candidates and complete
> intervention of observed nodes; the end-to-end gate-plus-ranker result is
> lower (0.0945), and cross-event performance remains event-dependent.

## Research safety status

- Raw datasets, labels, event IDs, timestamps, and protected split definitions
  were not modified.
- Every listed author-aware artifact was written to a new versioned location.
- The 30- and 60-minute artifacts enforce their own cutoff.
- Nested outer-event results use inner events only for representation selection.
- Rank-change analysis uses targets only after scoring to describe errors, never
  to fit a model or choose a variant.
