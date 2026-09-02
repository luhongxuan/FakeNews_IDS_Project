# Early Intervention Project — Full Research Handoff (2026-08-31)

## Read this first

This is a factual handoff of the work completed through 2026-08-31. It keeps
successful, negative, exploratory, and invalidated findings separate. It does
**not** replace either historical handoff file:

- `PROJECT_HANDOFF_20260829.md`
- `PROJECT_HANDOFF_20260831.md`

The repository root `AGENTS.md` governs all future work. The central rule is
that methodological validity is more important than a higher metric: at a
cutoff T, model inputs may contain only information that existed at or before
T. Raw datasets, labels, event IDs, timestamps, protected splits, and old
experiment outputs must not be changed or overwritten.

## Research question

The project studies early misinformation intervention. Given a thread's early
propagation snapshot, rank threads within an event so a limited intervention
budget can prevent as much later propagation as possible.

The primary historical protocol is:

- dataset: PHEME;
- observation window: source-tweet time + 30 minutes;
- evaluation split: leave-one-event-out (LOEO);
- candidate pool: known rumour threads (an **oracle-rumour** setting);
- budget: select the top 10 threads in the held-out event;
- metric: `CRR@10`, selected threads' preventable future impact divided by the
  held-out event's total preventable future impact.

The deployment-oriented protocol is stricter: rank all observed threads after
an early text rumour gate. It must never be presented as the same task as the
oracle-rumour protocol.

## Target and intervention semantics

For a source-anchored cutoff T, the observable snapshot contains only actions
with timestamp `<= source_time + T`. The future suffix is used only as an
outcome during evaluation.

The project has used two related target implementations over time:

1. The earlier graph pipeline defined preventable impact using continuous
   future-only descendant paths after intervening on all observed nodes. A
   future chain passing through an already observed node was not counted as
   blocked.
2. The later author-aware partial-observation pipeline stores
   `preventable_impact = number of actions after the cutoff` for the thread;
   it corresponds to the explicit strong policy that intervention on a selected
   thread's observed portion prevents all of that thread's subsequent actions.

These definitions are not silently interchangeable. Any comparison across the
two pipelines must state the target implementation used. In the current PHEME
representation, event-level propagation is a collection of separate thread
trees: selecting a thread has no modeled cross-thread overlap. Therefore the
reported "blocked actions" are counterfactual dataset actions, not verified
real-world people protected.

## Important safety and data corrections already discovered

The following were genuine research-validity or data-quality concerns, and are
part of the reason earlier results must not all be treated as formal baselines.

- A historical PHEME reply CSV had malformed large Twitter IDs after an Excel
  numeric conversion; later work uses corrected reply-v4 data.
- An older graph4 construction fitted numeric normalization globally before
  LOEO, creating test-distribution leakage risk. Later formal fold-safe code
  fits scalers and text PCA on training events only, then transform-only on the
  held-out event.
- The rumour filtering condition (`is_rumour`) required review; an earlier
  unfiltered/incorrect path was identified during debugging. Candidate-pool
  definitions must be checked for every reported experiment.
- `compute_blocked_descendants` / intervention semantics required correction;
  target interpretation must be read from the relevant builder rather than
  inferred from an older report.
- CUDA reproducibility and incomplete execution visibility were concerns in
  earlier GNN runs. Recent RF diagnostics are deterministic where seeds are
  specified, but every new model must still record seed and hardware details.
- Event-context z-scores with only one prior event thread produced numerical
  extremes. A stable derived artifact sets that z-score to zero when fewer than
  two earlier threads exist. This is time-safe and did not improve CRR.

No known future-input leakage was introduced in the later author-aware,
discourse, text, context, or diagnostic artifacts: their features are built
from cutoff-observable data, while future impact is used only for targets and
post-hoc analysis. This should still be re-audited whenever code changes.

## Work chronology and results

### 1. Original GraphSAGE direction (versions 3–4)

The early system used a two-head GraphSAGE design:

- Head A: predict future growth / impact;
- Head B: choose useful nodes within a previously selected thread.

The initial Head-A performance was weak (for example CharlieHebdo around
0.0527 CRR@10 against an oracle reference around 0.2316). Weighted losses,
top-k checkpointing, pairwise ranking, and two-stage variants did not resolve
the cross-event gap. Head B was not continued because its node-level
intervention objective degraded Head A and created a different policy problem:
choosing a few nodes inside a thread is not comparable to selecting whole
threads under the original policy.

Conclusion: do not revive Head B merely as a performance fix. It needs a
separate, explicitly specified intervention policy (for example, limited nodes
per thread or no automatic source intervention).

### 2. Historical strongest directly reported 30-minute oracle-rumour result

`graphsage_intervention_5` contains the strongest saved result under its own
strict 30-minute nested LOEO protocol:

| Method | Mean CRR@10, 7 eligible events |
|---|---:|
| Nested text-augmented Random Forest | **0.1438** |
| Fold-safe temporal RF | 0.1386 |
| GraphSAGE pairwise ranker | 0.1328 |
| XGBoost pairwise ranker | 0.1312 |
| Early-size baseline | 0.0977 |
| Random baseline | 0.0232 |

Result: `graphsage_intervention_5/experiments/20260829_112412_731271_nested_text_impact_ranker_rf/result.json`

The nested RF predicts `log1p(preventable_impact)` from early temporal,
structure, follower, and optionally frozen RoBERTa source/reply features.
Text was selected only in some outer events, not universally. The result is a
real improvement over the same-protocol size and random baselines, but it is
not evidence of reliable real-world deployment.

Across seven eligible events, the saved simulation reports 1,421 selected
future actions versus 797 for early-size and 204 for random. This is a
counterfactual under complete intervention, not a count of real people saved.

### 3. End-to-end candidate filtering

The deployment-style cascade was:

```text
all observed threads -> early text rumour gate -> impact ranker -> top 10
```

| Setting | Mean CRR@10 |
|---|---:|
| Text gate + nested ranker | **0.0945** |
| No gate; rank all threads | 0.0682 |
| Perfect true-rumour gate + ranker | 0.1453 |

Result: `graphsage_intervention_5/experiments/20260829_113832_479907_nested_text_gate_text_ranker_cascade/result.json`

The gate helps relative to ranking all threads, but early candidate filtering
still misses impactful rumours. Oracle-rumour and end-to-end results must be
reported separately.

### 4. Early signal and matched-control diagnosis

An oracle-versus-random plot initially showed that high-impact PHEME threads
often had more observed nodes, more 20–30-minute actions, deeper snapshots,
and lower leaf fraction. A size-and-late-activity matched-control analysis
then found no reliably stable remaining differences in depth, child counts, or
follower statistics after matching. This suggested that simple early volume and
late activity were the main available signal in the old representation.

The important later correction is that one broad visualization called the top
**10 percent** of each event "critical". It is a valid descriptive top-decile
plot but is not evidence that the exact top 10 are easy to choose. Random
controls in that first plot also included some critical examples, making the
contrast conservative but not a clean exact-top-10 control.

Do not claim from that chart that a model should be able to solve exact
Top-10 selection.

### 5. 60-minute experiments

`graphsage_intervention_6` introduced a separate 60-minute protocol and
correctly handles missing-parent/orphan components as a forest.

| Method | Mean CRR@10, eligible events |
|---|---:|
| Size baseline | **0.1225** |
| Fold-safe RF | 0.1118 |
| Random | 0.0214 |

Later author-aware 60-minute nested work reported 0.13520:
`graphsage_intervention_7/experiments/20260831_003925_210106_nested_discourse_context_selection_60min_v2/result.json`

Thirty and sixty minute CRR values are not direct substitutes: later cutoff
means less future target remains. The 60-minute comparison is useful only
against other 60-minute policies. More waiting did not yield a stable
cross-event advantage over an appropriate baseline.

### 6. Author-aware, partial-observation pipeline (version 7)

New derived 30-minute artifacts were created without overwriting old data.
They contain 6,341 records and use source-anchored actions, cutoff-safe user
history, account/profile information, response roles, strict-past event
context, frozen time-safe text semantics, and early discourse indicators.

The strongest result within this new pipeline is:

| Method | Mean CRR@10, 7 eligible events |
|---|---:|
| Nested role + strict-past context + discourse selection | **0.13849** |
| Fixed role-aware RF baseline | 0.12046 |

Result: `graphsage_intervention_7/experiments/20260830_212959_019175_nested_discourse_context_selection/result.json`

This is a candidate improvement within the version-7 pipeline, but it does
not exceed the historical 0.1438. Definitions and artifact construction differ,
so the 0.0053 gap is not a formal same-data head-to-head comparison.

Feature ablations found no universal feature family:

- account profile helped some events and hurt others;
- user history helped strongly in a small event but did not transfer;
- role features and text/discourse did not create a consistent global lift;
- strict-past context was valid but did not solve the ranking bottleneck;
- 30-to-60 rank changes were largely tied to recent activity / recency, with
  high event dependence.

### 7. Twitter15 / Twitter16 external-data inspection

The local RumDetect2017 Twitter15/16 material was inspected through derived,
time-safe audits rather than used to relabel PHEME. The hydrated content offers
early text/action signals, although many records are unavailable/deleted and
the original tree data alone is insufficient for a complete source-anchored
early-intervention reconstruction.

Twitter15 has 1,473 prepared user-action snapshot records and Twitter16 has
808. Their labels are truth/rumour classes (`true`, `false`, `unverified`,
`non-rumor`), not PHEME's preventable future impact. They can support an
external representation or early-rumour-learning experiment, but cannot
directly supervise PHEME intervention value without defining a separate,
clearly labelled transfer protocol.

Useful audits:

- `graphsage_intervention_7/twitter15_16_user_action_role_availability.json`
- `graphsage_intervention_7/twitter15_16_content_repetition_by_label.json`
- `graphsage_intervention_8/experiments/20260831_013436_027054_twitter15_16_critical_vs_random/`
- `graphsage_intervention_8/experiments/20260831_082908_795961_pheme_twitter_early_signal_comparison/`

### 8. Exact top-10 versus broad high-impact group: key diagnosis

The central apparent contradiction was: charts showed visible early signal,
yet learned models could not accurately select the ten highest-impact threads.
The corrected exact-top-10 audit resolves much of it.

For seven eligible events, the best individual signals separate exact Top-10
threads from all other threads moderately well, but are nearly non-discriminative
when compared with the hard near-miss group ranked 11–40 by true future impact:

| Example feature | AUC: Top-10 vs all | AUC: Top-10 vs ranks 11–40 |
|---|---:|---:|
| `seconds_since_last_action` | 0.7345 | 0.5712 |
| event recency percentile | 0.7224 | 0.5598 |
| observed leaf fraction | 0.6744 | 0.5883 |
| mean observed children | 0.6689 | 0.5336 |
| 20–30 minute count | 0.6546 | 0.5219 |

Exact audit:
`graphsage_intervention_8/experiments/20260831_153132_180708_pheme_exact_top10_signal_validity/`

Figure:
`graphsage_intervention_8/experiments/20260831_153537_550326_pheme_top10_vs_next30_visualization/pheme_top10_vs_next30_gap.png`

Interpretation:

- There is a genuine **broad enrichment signal**: early activity and recency
  make a thread more likely to be in a broad high-impact band than an ordinary
  thread.
- There is not a stable, equally strong **tail-ordering signal** that tells
  rank 1–10 apart from rank 11–40 across unseen events.
- Future impact is heavy-tailed. Missing a small number of very large threads
  can substantially lower CRR even if several broadly high-impact threads were
  selected.
- Mean group differences and broad AUC do not guarantee the high precision
  required by a top-10 budget.

This does not mean the data has no information. It means the available
30-minute fields provide coarse triage information rather than stable
fine-grained ordering of the future-impact tail.

### 9. Baseline near-miss replay

The saved author-aware nested baseline was replayed only to diagnose its
already-selected predictions; no outer-test model selection was performed.
It selects 3–7 broad top-band candidates in many events where random would
select about one. It therefore does use the broad signal. However, it both:

1. includes 3–7 below-band false positives among its ten selections; and
2. often fails to order candidates *within* the high-impact band.

Within-top-band score-versus-impact Spearman correlation is negative or near
zero in Ferguson, Germanwings, and Ottawa; it is positive but still insufficient
in CharlieHebdo, Putin, Prince-Toronto, and Sydney.

Diagnostic result:
`graphsage_intervention_9/experiments/20260831_160313_480791_baseline_top_band_near_miss_diagnostic/`

This confirms the user's hypothesis only partially. The issue is not solely
that the first 45 threads are interchangeable: the model has broad enrichment,
but also misses broad-band cases and has unstable ordering inside that band.

### 10. Experiments that did not improve the ranking problem

These are negative results. Do not tune them using their held-out LOEO scores.

| Experiment | Result / finding |
|---|---|
| Response-interaction nested RF | 0.13439; below version-7 nested baseline. |
| Stable event z-score nested selection | 0.13849; numerical correction, no lift. |
| Stable-momentum MLP | 0.12399; below baseline. |
| Hard-negative RF (Top-10 vs ranks 11–40 only) | 0.05854; markedly worse. |
| Three-band RF (Top-10 / 11–40 / rest) | 0.05752; markedly worse. |
| Exploratory continuation features / rule benchmark | Useful descriptive signal, not a formal early-intervention improvement; conclusions must respect exact cutoff definitions. |

Hard-negative and three-band failures are informative. Discrete band labels
discard continuous impact magnitude, give very few positives per event, vary in
prevalence by event, and optimize absolute class probability rather than
relative held-out-event ordering. They do **not** change the actual evaluation
target: CRR still measures future impact across all selected whole threads.

Relevant results:

- `graphsage_intervention_8/experiments/20260831_140216_245612_nested_response_interaction_selection/`
- `graphsage_intervention_8/experiments/20260831_143526_912416_nested_stable_zscore_selection/`
- `graphsage_intervention_8/experiments/20260831_152322_490354_nested_stable_momentum_ranker/`
- `graphsage_intervention_9/experiments/20260831_154210_687812_loeo_hard_negative_ranker_rf/`
- `graphsage_intervention_9/experiments/20260831_154815_121171_loeo_three_band_ranker_rf/`

## Current honest model hierarchy

| Question | Best currently saved evidence |
|---|---|
| Historical strict 30-minute oracle-rumour ranking | version-5 nested text RF, 0.1438 CRR@10. |
| Newest strict partial-observation feature pipeline | version-7 context + discourse nested RF, 0.13849 CRR@10. |
| Deployment-like all-thread process | text-gate cascade, 0.0945 CRR@10. |
| Broad high-impact triage | clearly above random in many events, but not stable precise Top-10 ordering. |

No result supports claiming that the system reliably identifies the highest
preventable-impact threads across all unseen PHEME events or is ready for
real-world deployment.

## Newly proposed task: minimum intervention set

The original task fixes a budget of ten threads and asks to maximize prevented
impact. The user proposed a valid different policy question:

> To reduce at least R percent of an event's future impact, which set of
> threads should be intervened on while using the fewest interventions?

For a set S, a simple unit-cost formulation is:

```text
minimize    sum(cost_i for i in S)
subject to  sum(true_preventable_impact_i for i in S)
            / sum(true_preventable_impact_i for all candidates) >= R
```

With `cost_i = 1`, this means "use the smallest number of threads." A real
deployment variant can give threads different review/removal costs. This
constraint prevents the trivial policy of deleting everything: once the target
reduction is reached, extra removals are penalized.

This must be implemented as a **new protocol**, not substituted for CRR@10.
Suggested evaluation at multiple fixed targets (10%, 20%, 40%, 60%):

- `MinThreads@R`: number of model-selected threads needed to truly reach R;
- oracle minimum number at the same R;
- excess interventions relative to oracle;
- failure rate under an explicit maximum intervention budget;
- achieved reduction as a budget-efficiency curve.

The model should rank using only cutoff-safe predicted impact. True future
impact is used only to evaluate whether the target was achieved. It does not
remove the current prediction bottleneck, but it can distinguish "wrong
threads" from "too many interventions required to reach a useful reduction."

Because current PHEME threads are independent trees, this policy initially has
additive effect; it does not model duplicate reach or cross-thread network
effects. Such effects require a new graph/data definition and should be
reported separately.

## Recommended next actions for the next chat

1. Preserve the current historical and version-7 oracle-rumour baselines; do
   not overwrite, merge, or call either an official universal best result.
2. If pursuing the new minimum-intervention policy, first write a protocol
   document and a leakage-safe evaluator. It should reuse fixed PHEME labels,
   event splits, cutoff snapshots, and stored model scores; run a small oracle
   sanity check before any model training.
3. Report new policy metrics beside—not instead of—CRR@10.
4. If trying to improve prediction under the existing exact Top-10 protocol,
   avoid another arbitrary loss or broad feature bundle. The evidence points to
   missing discriminative information for near-miss threads. Promising, but
   distinct, directions are: time-safe claim/stance signals, source credibility
   and verified account history, actual follower/reply-network information,
   cross-thread claim repetition, or carefully separated transfer/pretraining
   using Twitter15/16 truth labels.
5. Never use future actions, full-tree statistics, final labels, or test-event
   normalization/model selection as inputs to an early model. Do not train long
   runs without a smoke test and explicit user approval.

## File and output conventions requested by the user

- User prefers executable, no-argument Python scripts with paths/configuration
  visibly defined in the file.
- New work should print progress to the terminal and print its exact output
  directory at completion; avoid opaque background execution.
- Versioned work was organized under `graphsage_intervention_7`, broad signal
  diagnostics under `graphsage_intervention_8`, and hard-negative / three-band
  work under `graphsage_intervention_9`.
- Every new derived artifact and experiment must use a fresh timestamped output
  directory. Do not delete failed or partial results; document which ones are
  invalid rather than overwriting them.

