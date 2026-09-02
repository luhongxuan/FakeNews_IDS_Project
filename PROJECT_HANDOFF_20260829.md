# Early Intervention Project Handoff — 2026-08-29

## Project objective and research constraints

This repository studies early misinformation-propagation intervention on PHEME.

For an observation cutoff `T` (primarily 30 minutes):

- Inputs may use only nodes, edges, text, account attributes, and temporal information available at or before `T`.
- Intervening on a selected thread means intervening on **all nodes already observed** in that thread by `T`.
- Evaluation counts future nodes that can actually be blocked through an unbroken chain of future nodes.
- If a propagation path passes through a node already observed by `T`, the blocking chain resets. That downstream future path is not counted as blocked.
- The intervention budget is top 10 threads per event.
- Main metric: `CRR@10 = blocked future nodes / all future nodes`.
- Evaluation uses leave-one-event-out (LOEO). A test event must never contribute to fitting, normalization, PCA, model selection, or checkpoint selection.

The root `AGENTS.md` was read and followed. Research validity takes priority over metric gains.

## Early model work and data issues

`graphsage_intervention_3` originally used a two-head GraphSAGE model:

- Head A: future-growth prediction.
- Head B: coverage / node intervention value.

The initial CharlieHebdo result was around `0.0527`, compared with an oracle reference near `0.2316`. Several loss, positive-weight, top-k-checkpoint, and two-stage variants did not resolve the gap.

The correct intervention label was clarified:

```text
preventable_impact = future nodes blocked after intervening on all observed
nodes, but only through continuous future-only causal paths.
```

The user also found that the old `pheme_reply_level_v3.csv` had malformed Twitter IDs after Excel treated large IDs as numbers. The corrected v4 data is now used.

## Leakage finding and correction

An older graph4 dataset construction path fitted numeric normalization across all events before LOEO. That creates test-distribution leakage risk.

All formal later comparisons therefore use fold-safe scripts:

- Raw node attributes are reconstructed from `graphsage_intervention_4/pheme_node_features_roberta_replyv4.csv`.
- Each fold's scaler is fitted only on training events.
- Text PCA is fitted only on training events.
- Test events are transform-only.
- Snapshot nodes and edges are validated against the cutoff.

Old globally normalized GNN results must not be treated as formal baselines.

## Current 30-minute task and pipeline

Main work is in `graphsage_intervention_5`.

The 30-minute ranker predicts `log1p(preventable_impact)`. Inputs use only the 30-minute snapshot:

- Reply counts in 0–10, 10–20, and 20–30 minutes.
- Recent 5/10-minute activity, time since last activity, and reply interval summaries.
- Observed graph structure: node count, depth, leaf fraction, and observed child counts.
- Observable follower summaries.
- Optional text: frozen source-tweet RoBERTa embedding plus early-reply embedding centroid, reduced with train-fold-only PCA.

The primary ranking model is a `RandomForestRegressor`. For every event, it ranks threads and selects the top 10. The intervention simulation then intervenes on all selected threads' observed nodes.

### A. Oracle-rumour candidate pool

This setting assumes true rumour threads are known, to isolate the quality of impact ranking.

Results over the seven eligible events (`n_test >= 100`):

| Method | Mean CRR@10 |
|---|---:|
| Nested text-augmented RF | **0.1438** |
| Fold-safe temporal RF | 0.1386 |
| Size baseline | 0.0977 |
| Random baseline | 0.0232 |

Best result:

`graphsage_intervention_5/experiments/20260829_112412_731271_nested_text_impact_ranker_rf/result.json`

### B. End-to-end rumour gate plus ranker

The deployment-oriented flow is:

```text
all observed threads
-> early text rumour gate P(rumour)
-> impact ranker
-> top 10 threads
-> intervene on all observed nodes
```

The gate uses source + early-reply frozen RoBERTa embeddings with Logistic Regression and train-fold-only preprocessing. Earlier results were approximately macro ROC-AUC `0.7167`, PR-AUC `0.7065`, precision `0.585`, and recall `0.808`.

End-to-end results:

| Setting | Mean CRR@10 |
|---|---:|
| Text gate + nested text/base ranker | **0.0945** |
| No gate; rank all threads | 0.0682 |
| Perfect true-rumour gate then rank | 0.1453 |

Result:

`graphsage_intervention_5/experiments/20260829_113832_479907_nested_text_gate_text_ranker_cascade/result.json`

Interpretation: the gate improves over no gate, but it still loses high-impact rumours and admits some non-rumours. Even with a perfect gate, ranker quality remains the main limitation.

## Models tested without improvement

Do not retune these from their outer-test results.

| Method | Mean CRR@10 |
|---|---:|
| Nested text-augmented RF | **0.1438** |
| Fold-safe temporal RF | 0.1386 |
| GraphSAGE pairwise ranker | 0.1328 |
| XGBoost `rank:pairwise` | 0.1312 |
| Size baseline | 0.0977 |

Other attempted directions included original multi-head GraphSAGE variants, weighted losses, top-k checkpointing, two-stage proxy approaches, HistGradientBoosting, dynamic structure features, semantic features, and nested selection variants. None delivered a robust improvement over the RF baseline.

Relevant results:

- GraphSAGE pairwise: `graphsage_intervention_5/experiments/20260829_154533_427025_fold_safe_graphsage_pairwise_ranker/result.json`
- XGBoost pairwise: `graphsage_intervention_5/experiments/20260829_123348_754861_nested_xgb_pairwise_impact_ranker/result.json`

## Oracle-vs-random and matched-control analysis

`graphsage_intervention_5/plot_oracle_vs_random_early_signals.py` compares theoretical oracle top-10 threads against event-stratified random selections. Oracle membership uses future `preventable_impact` only for post-hoc diagnosis; it is not a deployable feature.

Oracle threads often have:

- more observed nodes;
- more replies in minutes 20–30;
- deeper observed cascades;
- lower leaf fraction;
- slightly higher maximum followers, but not consistently higher mean followers.

This alone could reflect early size/activity confounding. Therefore `graphsage_intervention_5/analyze_matched_oracle_controls.py` matches each oracle thread to a same-event non-oracle thread with similar:

- `log_observed_nodes`;
- `late_reply_fraction` (20–30 minute reply fraction).

Across 70 matched pairs (7 events x 10 pairs), after matching early size and late activity, differences in maximum depth, leaf fraction, maximum children, maximum followers, and mean followers all had bootstrap 95% intervals crossing zero.

Core result:

> The main usable signal in existing 30-minute data is early size plus sustained late-window activity. After controlling for these, the remaining available structure and follower features do not reliably distinguish the highest preventable-impact threads.

Latest matched analysis:

`graphsage_intervention_5/analysis/20260829_182034_272004_matched_oracle_controls/`

## 60-minute version

`graphsage_intervention_6` is an isolated 60-minute version.

Files:

- `prepare_60min_preventable_impact_dataset.py`
- `run_fold_safe_temporal_rf_60min.py`

Definition:

- Cutoff is 3600 seconds.
- Every observed node at or before 60 minutes is intervenable.
- The target counts future nodes blocked only through continuous future-only paths.
- The dataset stores raw numeric node values; fold-specific normalization happens only in training.

### Orphan / forest correction

The reply table contains 84 threads with non-source nodes whose `parent_id` is absent. A thread may be a forest rather than a single tree.

The 60-minute builder treats every missing-parent component as an independent root:

- An observed orphan can block its own future descendants.
- A future orphan cannot be incorrectly counted as blocked by observed nodes.
- No rows are dropped or forcibly attached to the source.

60-minute data validation passed:

- Rumour graphs: 2,402.
- All-thread graphs: 6,425.
- Every snapshot node has `offset_sec <= 3600`.
- Temporal feature length is 13.
- Features are finite and `preventable_y = log1p(preventable_impact)`.

Manifest:

`graphsage_intervention_6/preventableimpact_60min.manifest.json`

### 60-minute RF result

| Method | Mean CRR@10 on eligible events |
|---|---:|
| Size baseline | **0.1225** |
| Fold-safe RF | 0.1118 |
| Random | 0.0214 |

Result:

`graphsage_intervention_6/experiments/20260829_204000_474163_fold_safe_temporal_rf_60min/result.json`

30-minute and 60-minute absolute CRR values are not directly comparable because their future denominators differ. However, each can be compared with its same-cutoff size baseline. The 60-minute RF loses to its size baseline, so merely extending the observation window did not solve cross-event ranking.

## Current honest conclusion

The strongest defensible conclusion is:

> Under strict 30-minute information constraints, using only PHEME propagation snapshots, frozen text embeddings, and basic account features, prevention-aware ranking beats random and early-size baselines. However, after early size and late activity are controlled, the available data lacks stable additional signals for reliably identifying the highest preventable-impact threads across events.

Do not claim that the system reliably identifies the most important threads in every event or is ready for deployment.

The 60-minute negative result supports the conclusion that simply delaying intervention does not resolve the cross-event generalization problem.

## Recommended next steps

1. Freeze the 30-minute experimental protocol and prepare a paper-quality methods/results section:
   - corrected v4 data;
   - forest/orphan rule;
   - preventable-impact definition;
   - LOEO split;
   - budget 10;
   - eligible-event rule;
   - CRR@10 definition.
2. Add formal statistics for the final comparison:
   - per-event table;
   - mean and confidence intervals;
   - paired comparison of best RF vs size and random.
3. Clearly separate oracle-rumour ranking from end-to-end gate-plus-ranker performance.
4. Report the matched-control and 60-minute negative results in the discussion, rather than hiding them.
5. Avoid more broad model changes using the same depth/follower/branch features. Material improvement likely requires a different dataset or genuinely new sources such as retweet/like signals, account history, follower/reply networks, or external credibility information.

## Environment change

With explicit user approval, `matplotlib 3.11.1` was installed in the repository virtual environment. No raw dataset, v4 CSV, protected label, predefined split, or existing experiment result was modified.
