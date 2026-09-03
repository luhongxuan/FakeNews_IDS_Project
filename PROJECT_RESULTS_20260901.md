# Early Intervention Research Results — 2026-09-01

Retained training and feature code is organized under `effective_models/`.
The exact move/defer record is in `REORGANIZATION_MANIFEST_20260902.md`.

## Purpose and common task definition

This project ranks misinformation-propagation threads for intervention using
only information observed by a fixed early cutoff.  A model outputs one scalar
priority score per thread; an operator may intervene on any Top-K threads
(Top-1, Top-10, Top-30, etc.).

For a source-anchored 30-minute snapshot, the outcome is:

```text
preventable_impact = number of future reachable propagation nodes blocked
                     when intervening on the source and every observed node
                     in a selected thread.
preventable_y = log1p(preventable_impact)
```

`preventable_impact` is an outcome/supervision signal only.  It is never a
model input.  Ranking performance at budget K is reported as captured future
preventable impact divided by total candidate-pool future preventable impact
(CRR/reduction@K).

## Headline results

| Status | Dataset/protocol | Model | Main result | What it establishes |
|---|---|---|---:|---|
| Highest viable PHEME result (legacy artifact lineage) | PHEME, 30 min, rumour-only, nested LOEO, 7 eligible events | v5 RF with inner selection between baseline and text-PCA features | **0.1467 reduction@10** | A legal early model captures meaningful additional future preventable impact across unseen events. |
| Closest later PHEME candidate | PHEME v5 protected rumour-only artifact, 30 min, outer LOEO, 7 eligible events | Train-only calibrated active/quiet two-expert RF | **0.1461 reduction@10** | Splitting quiet and active threads recovers the original v5 level and improves the paired global RF, but does not surpass v5. |
| Timing-aware PHEME candidate | PHEME corrected-v5 allowlist, 10--60 min in 10-minute steps, event-held-out OOF, fixed total Budget 50 | Balanced cumulative RF sequential policy | **0.3261 mean horizon reduction** | Under a common minute-10 denominator, repeated cumulative decisions improve over the replayed single-30-minute hybrid (0.2966); this metric is not directly interchangeable with reduction@10. |
| Most traceable current PHEME formal result | PHEME Graph7 schema-locked artifact, 30 min, rumour-only, nested LOEO, 7 eligible events | RF with nested selection among role/context/discourse bundles | **0.1385 CRR@10** | The result survives schema locking and inner-only representation selection. |
| Direct early-size comparison | PHEME Graph7, 30 min, LOEO | author-aware RF | **0.1192** vs early-size **0.1059** | A model can outperform simply selecting the largest early threads. |
| Twitter graph-only holdout | Twitter15/16, source-level fixed holdout, 30 min | RF with 15 graph/activity features | T15: 0.1288 vs Oracle 0.2383; T16: 0.2598 vs Oracle 0.4465 | Pure early propagation graphs contain usable signal, but source-level holdout is not PHEME-style event generalization. |
| Frozen Twitter policy test | Twitter15/16, strict rumour-only source holdout, 30 min | 31-feature scalar MLP with weak Oracle-order loss selected on validation | T15: **0.1986**; T16: **0.3292** reduction@10 | A small order weight helped validation; the frozen model produced a complete arbitrary-K priority ranking on test. |

Important: do **not** rank rows from different artifact lineages as an exact
leaderboard.  They differ in frozen data products and sometimes candidate-pool
policy.  Their value is evidence from complementary safety-checked protocols.

---

## A. PHEME: viable 30-minute intervention models

### A1. v5 frozen RF — highest observed viable result

**Result artifact**

- `effective_models/pheme_v5_text_rf/reference_result/20260831_223827_909201_v5_best_text_oof_min_interventions/result.json`
- OOF rankings: `effective_models/pheme_v5_text_rf/reference_result/20260831_223827_909201_v5_best_text_oof_min_interventions/oof_thread_scores.csv`

**Protocol**

- PHEME rumour-only candidate graphs.
- 30-minute cutoff.
- Nested leave-one-event-out (LOEO).
- At every outer event, inner events select `baseline` versus
  `text_augmented`; the outer event does not choose its own feature set.
- RF regression target: `log1p(v5 chain-future preventable_impact)`.
- The model predicts a scalar score and can therefore support arbitrary K.

**Feature sets**

Baseline: early temporal counts/recency/interarrival, observed-leaf and
children summaries, observed-node count, late activity fraction, max followers,
mean depth, and max depth.

Text-augmented: baseline plus train-fold-fitted 64-component text PCA.  The
inner loop selected the bundle separately per held-out event.

**Outer reduction@10 by eligible event**

| Event | Reduction@10 | Selected bundle |
|---|---:|---|
| CharlieHebdo | 0.1962 | baseline |
| Ferguson | 0.0797 | text-augmented |
| Germanwings-crash | 0.1377 | baseline |
| OttawaShooting | 0.0835 | text-augmented |
| PrinceToronto | 0.1324 | text-augmented |
| PutinMissing | 0.2806 | baseline |
| SydneySiege | 0.1166 | baseline |
| **Macro mean** | **0.1467** | — |

**Budget-efficiency result**

To reach 20% future-impact reduction, the model required on average 22.57
threads; the Oracle required 6.71.  This is a genuine remaining gap, but it
also demonstrates that the model supplies an actionable ranking rather than
only a fixed Top-10 answer.

**Caveat**

This is a valid, useful legacy result.  Its exact artifact lineage differs from
Graph7, so it must be presented beside—not silently merged with—the Graph7
number below.

### A2. Graph7 nested RF — strongest schema-locked formal confirmation

**Result artifact**

- `effective_models/pheme_graph7_discourse_rf/reference_result/20260830_212959_019175_nested_discourse_context_selection/result.json`
- Dataset: `effective_models/pheme_graph7_discourse_rf/artifacts/20260830_170000_early_discourse_response_v1`

**Protocol**

- 6,341 total threads; 2,373 rumour candidates; 9 events.
- Strict 30-minute schema-locked artifact with 52 approved early features.
- Nested LOEO; the outer held-out event is never used to select feature bundle.
- Eligible-event macro excludes the two small events (`ebola-essien`,
  `gurlitt`) according to the pre-existing candidate-count convention.
- RF: 300 trees, depth 8, minimum leaf 3, `max_features=0.8`, seed 42.

**Feature bundles selected in the inner loop**

1. Role base: activity, structure/time, author history/profile.
2. Plus strict-past event context.
3. Plus early discourse and frozen semantic features.

The final selected bundle was `plus_event_context_discourse` for 8/9 outer
events; `gurlitt` selected context without discourse.

**Result**

```text
Nested eligible-event macro CRR@10 = 0.1385
```

The directly recomputed same-pool early-size baseline for these seven eligible
events is 0.1033.  Thus this model improves captured future impact by 0.0352
absolute, or about 34.1% relative to selecting the largest early threads.

**Why this is valuable for the project**

It is the clearest current evidence that an early intervention ranker does more
than select the largest visible propagation tree.  It uses fixed schema,
strict cutoff provenance, event-disjoint evaluation, and inner-only feature
selection.

### A3. Graph7 author-aware RF — direct baseline comparison

**Result artifact**

- `effective_models/pheme_graph7_author_aware_rf/reference_result/20260830_140800_517954_author_aware_rf_partial_formal/result.json`

**Method**

RF on 15 pre-specified 30-minute features: observed actions/users/concentration,
three temporal bins, recency, observed tree shape, pre-cutoff user history, and
follower summaries.

**Eligible-event mean**

```text
Model CRR@10:        0.1192
Early-size CRR@10:   0.1059
Random CRR@10:       0.0390
```

This is a simpler, easily explainable demonstration that temporal/author
information adds value beyond early thread size.

### A4. Oracle-policy scalar MLP: important negative result

**Artifacts**

- 30 min: `research_scratch/legacy_full/graphsage_intervention_11/experiments/20260901_030603_692493_nested_oracle_policy_scalar/result.json`
- 60 min: `research_scratch/legacy_full/graphsage_intervention_11/experiments/20260901_110738_912447_nested_oracle_policy_scalar_60min/result.json`

**Method tested**

One score per thread from a 32-unit MLP, trained with Huber utility loss plus
within-event weighted pairwise Oracle-order loss.  Inner LOEO selected feature
bundle and order-loss weight.

**Result**

```text
30 min scalar hybrid: 0.1245, below the referenced v5 RF 0.1467.
60 min scalar hybrid: 0.1310, below the Graph7 60-min RF 0.1352.
```

This is a useful negative finding: merely adding Oracle-order supervision did
not close the gap.  The bottleneck is not solved simply by changing loss
function or using an MLP.

### A5. Safe account-age RF ablation

**Artifact**

- `research_scratch/legacy_full/graphsage_intervention_14/experiments/20260901_190241_925101_nested_pheme_v5_text_account_age`

**Training method**

- Protected v5 PHEME rumour-only candidate artifact, strict 30-minute cutoff.
- Nested event-separated LOEO.
- RandomForestRegressor: 300 trees, depth 8, minimum leaf 3,
  `max_features=0.8`, seed 42.
- Unchanged utility label: `log1p(preventable_impact)`.
- Inner events selected between a safe no-profile temporal/depth + text-PCA64
  bundle and the same bundle plus mean observed account age.
- Account age was computed only for accounts attached to observable snapshot
  nodes. Mutable engagement/profile fields were excluded.

**Result**

| Budget | Mean Oracle efficiency | Mean reduction |
|---:|---:|---:|
| 1 | 0.3435 | 0.0241 |
| 5 | 0.4063 | 0.0787 |
| 10 | 0.4428 | 0.1319 |
| 20 | 0.4814 | 0.2132 |
| 50 | 0.5994 | 0.4226 |
| 100 | 0.7209 | 0.6162 |

The account-age bundle was selected by the inner loop for seven of nine outer
events, but the final reduction@10 remained below the original v5 result.
The useful conclusion is that observed account age contains some transferable
signal, but is not a decisive replacement for the original v5 feature policy.

### A6. Recency-gated active/quiet two-expert RF

**Artifacts**

- Uncalibrated: `research_scratch/legacy_full/graphsage_intervention_14/experiments/20260901_203440_800572_paired_oof_pheme_quiet_expert`
- Train-only calibrated: `effective_models/pheme_active_quiet_calibrated_rf/reference_result/20260901_210513_260899_paired_oof_pheme_quiet_expert_calibration`

**Training method**

- Protected v5 rumour-only candidates, 30 minutes, outer LOEO.
- An unsupervised gate uses the current outer-train events' 60th percentile of
  `log1p_seconds_since_last_activity` to separate quiet and active threads.
- A separate RF utility regressor is fitted for each group. Both predict the
  unchanged `log1p(preventable_impact)` label.
- Shared inputs are safe temporal/depth features, train-fold text PCA, and
  observed account age. Retweet/favorite counts and mutable profile fields are
  excluded.
- The calibrated version learns positive-slope ridge mappings for the two
  experts from inner event-OOF predictions only, so held-out outcomes do not
  determine cross-expert score scale.

**Paired OOF result across seven eligible events**

| Budget | Global RF efficiency | Two experts | Calibrated two experts |
|---:|---:|---:|---:|
| 1 | 0.3435 | 0.2465 | 0.2465 |
| 3 | 0.4214 | 0.3920 | 0.3920 |
| 5 | 0.4063 | **0.4948** | **0.4948** |
| 10 | 0.4428 | 0.4685 | **0.4800** |
| 20 | 0.4814 | 0.5006 | **0.5063** |
| 50 | 0.5994 | **0.6143** | 0.6135 |
| 100 | **0.7209** | 0.7162 | 0.7089 |

At Budget 10, calibrated two experts reach `0.1461` mean reduction versus
`0.1319` for the paired global RF, an absolute gain of 0.0142.  This is the
strongest later PHEME candidate, but it is still just below the original v5
`0.1467`.  The main highlight is improved medium-budget coverage (K=5--50),
not better extreme-head ranking: K=1 and K=3 are worse than the global RF.

### A7. Quiet-expert alternatives and what they ruled out

All experiments below retain the same protected candidate pool, 30-minute
boundary, event-disjoint outer evaluation, train-fold recency gate, and future
utility as outcome only.

| Quiet model | Training idea | Useful result | Limitation |
|---|---|---|---|
| Direct source+reply semantic Ridge | Train-only scaler/PCA of source and observed-reply embeddings; inner LOEO chooses semantic variant | Clean test of semantic utility for quiet threads | Nested selection returned the ordinary quiet RF; semantic variants reduced full-policy reduction@10 to 0.1351 versus 0.1421. |
| Cross-fitted semantic residual | Learn semantic correction to an event-OOF RF utility residual, then apply only to quiet held-out rows | Full-policy reduction@10 rose from 0.1421 to **0.1483** | Gains were local: K=3, 20, 50 and 100 fell, so it is not a stable replacement. |
| Quiet-only semantic pairwise ranker | Within-event pairs teach source+reply semantics to order quiet threads | Quiet Top-1 Oracle efficiency rose from 0.0871 to **0.2224** | Quiet K=3--50 degraded substantially; better first choice did not produce a good full ranking. |
| Sparse 10-feature quiet RF | Retain only recency/activity/tree features and remove text/account-age noise | K=3 was effectively tied (0.1931 vs 0.1919) | K=1, 5, 10, 20 and 50 were worse; aggressive feature removal lost useful combined signal. |

These runs establish that the quiet-thread problem is not solved by more text,
pairwise loss, residual correction, or a very small hand-selected feature set
alone.  The next defensible hypothesis is nested selection over feature-group
**combinations**, rather than assuming every available view should always be
included.

### A8. Exhaustive nested quiet feature-group combinations

The follow-up evaluated all 255 non-empty combinations of eight legal
30-minute feature groups under nested event-separated selection. The selected
combination improved mean quiet Oracle efficiency at K=1 from 0.0871 to 0.2582
and at K=3 from 0.1919 to 0.2731, but degraded K=5, 10, 20, 50, and 100. Macro
Spearman correlation also fell from 0.3171 to 0.2995. The large K=1 gain was
especially sensitive to an exact Oracle hit on `putinmissing` and did not yield
a consistently better full ranking.

No stable cross-event subset emerged: topology and semantic-summary groups were
selected in five of seven folds, while account age was selected in none. This
is retained as exploratory negative evidence rather than an effective model.
The complete run is archived at
`research_scratch/legacy_full/graphsage_intervention_14/experiments/20260901_233016_536789_nested_pheme_quiet_feature_combinations`.

### A9. Multi-checkpoint cumulative RF and fixed sequential intervention

**Reference bundle**

- `effective_models/pheme_multicheckpoint_rf/reference_result/20260903_160025_balanced_cumulative_sequential_policy`

This candidate replaces a single decision at minute 30 with six decisions at
minutes 10, 20, 30, 40, 50, and 60. It uses the same 2,402-thread corrected-v5
rumour allowlist. The cumulative RF sees every legal observation available up
to the current checkpoint; the comparison Window/Delta RF sees the current and
previous ten-minute windows plus their changes. Both use pooled
elapsed-time-aware Random Forest regressors with 300 trees, depth 8, minimum
leaf 3, `max_features=0.8`, and seed 42. Every OOF prediction excludes the
entire held-out event from fitting.

The sequential policy fixes total Budget 50 in advance and allocates
`[9, 9, 8, 8, 8, 8]` interventions across the six checkpoints. Once selected,
a thread is removed from later candidate pools. Prevented nodes are counted at
the actual action time, and all policies divide by total future preventable
impact at minute 10. This common horizon penalizes late intervention and makes
the timing policies directly comparable to the replayed 30-minute policies.

| Fixed policy | Mean horizon reduction | Total blocked future nodes | Positive interventions | Mean action minute |
|---|---:|---:|---:|---:|
| Single-30 corrected-target v5 | 0.2748 | 3,790 | 301 / 350 | 30.0 |
| Single-30 cumulative RF | 0.2852 | 3,926 | 313 / 350 | 30.0 |
| Single-30 quiet-tier hybrid | 0.2966 | 3,995 | 315 / 350 | 30.0 |
| Balanced Window/Delta | 0.3129 | 4,570 | 298 / 350 | 34.2 |
| Balanced cumulative + one Window challenger per checkpoint | 0.3200 | 4,596 | 306 / 350 | 34.2 |
| **Balanced cumulative RF** | **0.3261** | **4,527** | **310 / 350** | **34.2** |
| Descriptive dynamic Oracle | 0.4928 | 7,356 | 346 / 350 | 34.2 |

Against the single-30-minute hybrid under this shared protocol, balanced
cumulative improves mean horizon reduction by 0.0295 absolute (about 9.95%
relative) and blocks 532 additional future nodes. It improves six of seven
eligible events and ties `putinmissing`; no event is worse. The Window
challenger has a larger raw blocked-node sum because event sizes differ, but a
lower event-macro reduction and three event regressions, so it is not selected
as the candidate.

The quiet-thread diagnostic remains a limitation rather than a solved result.
Window/Delta finds four more quiet Oracle Top-50 threads at minute 10 and two
more at minute 20, but loses more active hits. From minute 30 through minute 60,
cumulative finds at least as many quiet hits at every checkpoint and more
active hits. Cumulative still retrieves only 17--20 of 76--82 pooled quiet
Oracle Top-50 cases per later checkpoint, so the sequential gain comes mainly
from repeated earlier decisions rather than a decisive quiet-burst feature.

The balanced schedule and challenger count were fixed before outer-outcome
replay. They were not selected using held-out results. The dynamic Oracle is an
unavailable upper bound. Any attempt to optimize the quota schedule must use a
new nested policy-selection protocol; this result must not be used to tune and
then re-report a better outer schedule.

### A10. Next-10-minute wait-loss Hazard RF

**Reference bundle**

- `effective_models/pheme_multicheckpoint_rf/reference_result/20260903_172123_wait_loss_hazard_policy`

This exploratory follow-up separates intervention value from urgency. The
existing cumulative RF supplies the current utility score. A second pooled RF
predicts `log1p(next10_wait_loss)`, where wait loss is the nonnegative decrease
in corrected dynamic preventable impact between T and T+10 minutes. This label
is future supervision only; the model retains the same 57 cutoff-safe
cumulative inputs. The full evaluation fitted 56 inner and seven outer Hazard
models under event separation, then selected utility threshold, Hazard
threshold, and maximum budget from inner results only.

The Hazard prediction task is learnable across held-out events:

| Checkpoint | Spearman | Positive ROC-AUC | Positive PR-AUC | Top-20 wait-loss capture |
|---:|---:|---:|---:|---:|
| 10 min | 0.451 | 0.746 | 0.734 | 0.522 |
| 20 min | 0.401 | 0.732 | 0.604 | 0.527 |
| 30 min | 0.399 | 0.752 | 0.578 | 0.591 |
| 40 min | 0.362 | 0.754 | 0.510 | 0.640 |
| 50 min | 0.318 | 0.744 | 0.423 | 0.535 |

However, predictive signal did not translate into a better intervention
controller:

| Policy | Mean interventions | Mean horizon reduction | Total blocked future nodes | Mean action minute |
|---|---:|---:|---:|---:|
| Hazard target 20 | 30.14 | 0.1473 | 2,944 | 19.34 |
| Hazard target 25 | 36.57 | 0.2164 | 3,495 | 12.04 |
| Hazard target 30 | 39.43 | 0.2281 | 3,675 | 12.04 |
| Hazard cost 0.2 | 47.14 | 0.3242 | 3,780 | 10.11 |
| Hazard cost 0.05 / 0.1 | 49.29 | 0.3344 | 4,027 | 10.44 |
| **Fixed balanced cumulative 50** | **50.00** | 0.3261 | **4,527** | 34.20 |

The low-cost policies have slightly higher event-macro reduction but block 500
fewer total nodes than fixed balanced cumulative and lose on three of seven
events. Inner selection usually chooses Hazard threshold zero, while nonzero
gates still place most interventions at minute 10. The defensible conclusion is
that legal early features predict near-term continuation, but this does not by
itself identify when reserving an intervention slot is beneficial. This model
is retained as a timing-prediction diagnostic, not a replacement policy.

Because the Hazard hypothesis followed inspection of earlier outer PHEME
results, this is nested and leakage-safe but not an untouched confirmatory
test. A later burst-onset target must not be tuned and claimed as fresh evidence
on the same consumed outer events.

---

## B. What the PHEME diagnostics established

### B1. Oracle scorer and model scorer use the same intervention accounting

The audit `research_scratch/legacy_full/graphsage_intervention_12/audit_oracle_learnability_strict30.py`
replayed Oracle ranking by sorting the same candidate pool on
`preventable_impact`.  It passed exactly for every tested budget.  Both Oracle
and model sum `preventable_impact` over their selected threads and divide by
the same event total; there is no separate Oracle scoring formula.

### B2. Broad strict-30 feature search

**Artifact**

- `research_scratch/legacy_full/graphsage_intervention_13/experiments/20260901_121851_930507_strict30_comprehensive_oracle_feature_gap`

This audit fixed each Oracle Top-10 thread's nearest non-Oracle control using
the existing 52 Graph7 features, then compared 128 legal early features:

- 52 frozen Graph7 features;
- 5 time-safe NLI stance features;
- 71 newly reconstructed raw graph, timing, participation, text-structure,
  tweet-metadata, and account-age features.

The most consistent candidate signals were late activity, late unique-user
share, active leaves/frontier in the final ten minutes, text length, and NLI
neutrality.  Effects were modest (largest paired standardized effects around
0.3–0.47), not a single decisive hidden feature.  This explains why a legal
model can beat early size but still remain far below Oracle.

### B3. Important research interpretation

Some matched Oracle/control pairs have similar early features but large future
impact gaps.  This is evidence of conditional outcome uncertainty under a
30-minute observation window, not proof that no improvement is possible.  New
feature ideas must be tested through a fresh split-safe inner selection
protocol, not promoted because a post-hoc diagnostic looks favourable.

### B4. Quiet-thread identifiability screens

The quiet cohort was examined with broad safe scalar features, graph/time
representations, source/reply semantic summaries, embedding PCA, nonlinear
models, and capacity ablations.  The strongest descriptive signals were
recency/interarrival behavior and some source+reply semantic directions, but
performance varied sharply by event.  In particular, broad safe ExtraTrees
membership classification reached only macro LOEO ROC-AUC 0.5599 and PR-AUC
0.1731, while a PCA+linear snapshot-embedding view reached ROC-AUC 0.6769 but
did not translate into a consistently superior utility ranking.

The main research insight is that several features are weakly informative in
combination, yet no view separates quiet Oracle-head threads reliably across
events. Diagnostic separability must therefore not be confused with a
validated intervention-policy improvement.

---

## C. Twitter15/16 results

### C1. Protected graph-only RF holdout

**Result artifact**

- `effective_models/twitter_graph_only_rf/reference_result/20260901_013107_933844_twitter15_16_graph_only_ranker/run_record.json`

**Method**

- Source-anchored 30-minute Twitter Snowflake snapshot.
- Strictly forward observed forest; source and every observed node are
  intervened on.
- 15 legal graph/activity features only; no original label, text, user data,
  mutable counts, future nodes, or future edges.
- Fixed source-level train/validation/test split; RF depth selected on
  validation, then test evaluated once.

**Test results**

| Corpus | Candidates | Model CRR@10 | Oracle CRR@10 | Exact Top-10 overlap |
|---|---:|---:|---:|---:|
| Twitter15 | 237 | 0.1288 | 0.2383 | 3/10 |
| Twitter16 | 109 | 0.2598 | 0.4465 | 5/10 |

This result establishes that pure propagation-graph features have useful early
ranking signal.  It is an auxiliary source-level holdout, not event-separated
generalization and not directly comparable with PHEME LOEO.

### C2. Twitter graph-signal audit

**Artifact**

- `research_scratch/legacy_full/graphsage_intervention_13/experiments/20260901_124248_513683_twitter15_16_strict30_comprehensive_graph_oracle_gap`

Using fixed test-cohort Oracle/control matched pairs, the audit compared 55
strict-30 graph/timing features.  Unlike PHEME, Twitter showed clearer early
graph signals: observed size, shorter interarrival, max/mean/p90 depth, depth
variation, and several fine temporal bins.  Max depth was consistently
positive in both corpora.

This is exploratory only: because it inspected the fixed test cohort, it must
not be treated as a fresh test confirmation for features selected from it.

### C3. Graph14 head-focused scalar development run

**Artifacts**

- Feature artifact: `effective_models/twitter_weak_order_scalar_mlp/artifacts/20260901_133740_887252_twitter15_16_strict30_graph_head_features_v1`
- Development result: `research_scratch/legacy_full/graphsage_intervention_14/experiments/20260901_133933_282144_validation_head_focused_scalar_v1`

**Pre-specified feature bundle (31 features)**

- 10 existing early activity/recency/interarrival features.
- Six 5-minute bins, last-activity gap, interarrival p90, burstiness, and
  late-to-early ratio.
- Max/mean/std/p90/entropy of depth; leaf and late-leaf statistics; root
  children; branch HHI/Gini; width-to-depth ratio.

**Model and label**

```text
31 early features -> Linear(31,32) -> ReLU -> Dropout(0.05) -> scalar score
utility label = preventable_y = log1p(preventable_impact)
```

Twitter15 and Twitter16 use separate train-fitted scalers and separate scalar
models.  Four loss settings and three seeds were compared.  The intended
head-order and soft-Top-K losses were constructed only from training outcomes.

**Validation finding**

```text
Best multi-budget validation Oracle efficiency: 0.6445
Selected configuration: utility-only (order lambda = 0; soft-Top-K lambda = 0)
```

The tested order/soft-Top-K additions reduced validation performance.  At
validation Top-10, the selected utility-only ensemble captured:

| Corpus | Model blocked nodes | Oracle nodes | Oracle efficiency |
|---|---:|---:|---:|
| Twitter15 | 735 | 836 | 87.9% |
| Twitter16 | 446 | 916 | 48.7% |

It can output a full priority ranking and therefore supports arbitrary K.  The
ranking is stored in `validation_priority_ranking_any_k.csv`.

**Strict reporting caveat**

The training loss, scaler, and validation selection excluded test rows.
However, the implementation loaded the combined artifact and performed a
whole-artifact target-integrity assertion, which unnecessarily read the test
target.  This run is therefore development evidence, not a strictly isolated
formal evaluation.  It must be corrected before any formal test claim.

### C4. Strict weak-order scalar MLP and frozen test

**Artifacts**

- Validation selection: `effective_models/twitter_weak_order_scalar_mlp/reference_result/20260901_135117_434080_validation_head_focused_scalar_v2_strict_weak_order`
- Frozen test: `effective_models/twitter_weak_order_scalar_mlp/reference_result/20260901_135453_538898_test_head_focused_scalar_v2_strict_weak_order`

The corrected v2 loader skips test rows before parsing features or targets.
The same 31-feature, 32-unit scalar MLP was trained for 160 epochs with three
seeds. Validation compared order weights 0, 0.02, 0.05, and 0.1 with soft-Top-K
disabled. The selection criterion was mean Oracle efficiency over both corpora
and K={1,3,5,10,20,40}.

Validation selected the weak order weight `lambda=0.05` with mean efficiency
0.6661, compared with 0.6445 for utility-only. After freezing this choice, the
224 test threads were evaluated once:

| Corpus | Top-10 blocked / Oracle | Reduction@10 | Oracle efficiency@10 | Efficiency@40 |
|---|---:|---:|---:|---:|
| Twitter15 | 664 / 1,209 | **0.1986** | 0.5492 | 0.6952 |
| Twitter16 | 532 / 797 | **0.3292** | 0.6675 | 0.9149 |

This is the cleanest Twitter scalar-policy result. It shows that a small
Oracle-order term can help after validation tuning, while large order weights
previously overwhelmed utility learning. It does not prove superiority over a
utility-only final test because utility-only was not separately frozen and
tested under the same one-shot protocol.

### C5. Strict rumour-only v5-style Twitter RF

**Artifact**

- `research_scratch/legacy_full/graphsage_intervention_14/experiments/20260901_140245_040764_validation_v5_style_rf_strict_rumour_only`

This model transfers the v5 training style—not its PHEME data—to Twitter15/16:
RF utility regression on the same 31 strict 30-minute activity, fine-temporal,
and topology features. It uses three seeds, corpus-specific train-only scaling,
and excludes the non-rumour category according to the frozen artifact policy.

The mean multi-budget validation efficiency was 0.6026, below the utility-only
MLP's 0.6445. At K=10, RF efficiency was 0.7644 on Twitter15 and 0.4421 on
Twitter16. The useful finding is architectural: v5-style RF remained strong on
some budgets, but the scalar MLP generalized better on average with this
Twitter feature representation.

### C6. Extended 30-minute graph/time features

**Artifact**

- `research_scratch/legacy_full/graphsage_intervention_14/experiments/20260901_144351_171526_validation_extended_graph_time_scalar`

Additional train/validation-only graph/time combinations improved Twitter15
validation (Oracle efficiency 0.7931 at K=10 and 0.8239 at K=20), but did not
repair Twitter16 (0.4869 and 0.5017 respectively). This corpus asymmetry is a
central result: a feature can be genuinely useful without being stable across
Twitter15 and Twitter16.

### C7. Independent strict-60 scalar models

**Artifacts**

- Single scalar: `research_scratch/legacy_full/graphsage_intervention_14/experiments/20260901_145459_984192_validation_strict60_oracle_policy_scalar`
- Two branch: `research_scratch/legacy_full/graphsage_intervention_14/experiments/20260901_152404_223561_validation_strict60_two_branch_scalar`
- Activity-primary structural residual: `research_scratch/legacy_full/graphsage_intervention_14/experiments/20260901_152951_636516_validation_strict60_activity_primary_structure_residual`

A new 60-minute train/validation-only artifact was created; it did not reuse
30-minute outcomes. The utility target counts impact strictly after minute 60.
The single scalar uses 37 activity, fine-time, and topology features. Validation
again selected utility-only: mean efficiency 0.5652 versus 0.5536 for order
weight 0.05.

At K=10, the single scalar reached efficiency 0.8492 on Twitter15 and 0.4688 on
Twitter16. Separating activity and structure into two learned branches improved
some extreme budgets (for example Twitter15 K=1: 0.8934), but hurt the important
middle budgets. A bounded structural residual similarly helped some K=40 and
Twitter16 K=1 cases while degrading K=10/K=20. Therefore neither structural
decomposition is a general replacement for the single utility score.

### C8. Oracle Top-50 set-membership experiment

**Artifact**

- `research_scratch/legacy_full/graphsage_intervention_14/experiments/20260901_170302_640698_validation_strict60_oracle_top50_set`

This changes the learning question from exact ranking to identifying the
Oracle Top-50 set. On strict-60 validation, the predicted and Oracle sets
overlapped by 20/50 for Twitter15 and 38/50 for Twitter16. Jaccard scores were
0.2500 and 0.6129 respectively. Because a 50-of-71 selection is already easy
to overlap by chance on Twitter16 (expected intersection 35.21), the lift over
random was only 1.079 there. This experiment is useful for measuring candidate
retrieval, but must not be presented as equivalent to intervention ranking.

### C9. Twitter-trained scalar transfer and holdout

**Artifacts**

- PHEME zero-shot: `research_scratch/legacy_full/graphsage_intervention_11/experiments/20260901_125214_360130_twitter15_16_to_pheme_zero_shot_scalar`
- Twitter holdout: `research_scratch/legacy_full/graphsage_intervention_11/experiments/20260901_132348_008102_twitter15_16_holdout_intervention_scalar`

The 32-unit scalar MLP was trained only on Twitter15/16 train rows. Validation
selected Oracle-order weight 0.75 from {0, 0.25, 0.75}; source IDs shared with
PHEME were removed before training. On PHEME rumour-only zero-shot evaluation,
mean eligible-event reduction@10 was 0.0936 with Oracle efficiency 0.3445. This
is evidence of limited cross-corpus transfer, not a replacement for PHEME LOEO.

On the Twitter source-level test split, the same training family reached:

| Corpus | Reduction@10 | Oracle efficiency@10 |
|---|---:|---:|
| Twitter15 | 0.1734 | 0.4797 |
| Twitter16 | 0.3304 | 0.6700 |

The Twitter holdout confirms that the scalar model learns an actionable signal
within its source corpora, while the PHEME zero-shot drop quantifies the domain
shift.

---

## D. Material suitable for the project presentation/report

1. **A real operational task, not only rumour classification.**  The system
   ranks threads by estimated intervention priority and supports an operator's
   budget K, rather than making a binary rumour prediction only.

2. **Strict early-observation discipline.**  Snapshots use only nodes/edges at
   or before the cutoff; future propagation contributes only to outcome
   construction and evaluation.

3. **Demonstrated value beyond a naive policy.**  On the Graph7 formal PHEME
   protocol, the nested model obtains 0.1385 versus 0.1033 for selecting the
   largest early threads, a 34.1% relative increase in captured future impact.

4. **A strongest viable result with transparent qualification.**  The v5 RF
   reaches 0.1467 reduction@10 across seven eligible unseen events.  Present it
   as the highest observed viable result under its legacy frozen artifact
   lineage, not as a silently interchangeable number with Graph7.

5. **External graph-only evidence.**  Twitter15/16 demonstrates that early
   structural signatures—especially depth and temporal density—can be more
   pronounced in another corpus, supporting the plausibility of graph-aware
   early intervention.

6. **Honest limits are a strength.**  Oracle remains substantially better,
   scalar Oracle-policy loss did not automatically improve PHEME, and PHEME
   has matched pairs with similar early observations but divergent futures.
   The project therefore reports a real, bounded predictive contribution rather
   than overclaiming counterfactual certainty.

7. **Expert specialization helps at practical middle budgets.**  The
   train-only calibrated active/quiet RF improves the paired global RF from
   0.1319 to 0.1461 reduction@10 and improves Oracle efficiency from K=5 to
   K=50. It does not improve K=1/K=3 and does not exceed the original v5
   0.1467, so it is a candidate mechanism rather than a promoted baseline.

8. **Small ranking supervision can work when carefully weighted.**  On the
   strict Twitter validation protocol, order weight 0.05 beat utility-only;
   larger order weights and soft-Top-K losses were harmful. The frozen test
   ranking then blocked 19.9% and 32.9% of future impact at K=10 on Twitter15
   and Twitter16 respectively.

## E. Recommended next steps

1. Complete the nested PHEME quiet feature-group combination search. It tests
   all 255 non-empty combinations of activity level, temporal dynamics,
   topology, text surface, sentiment, account age, semantic summaries, and
   source+reply embedding PCA. Each outer event uses inner LOEO for selection.
   The experiment is currently running and therefore has no result in this
   report yet.
2. Compare its completed OOF curve against the paired global RF, current quiet
   RF, calibrated two-expert RF, and original v5 at every pre-specified budget;
   do not select only the most favourable K.
3. Treat the Twitter frozen test as consumed. Do not tune new Twitter features
   against those 224 outcomes or reuse that cohort as fresh confirmation.
4. Preserve v5 and Graph7 as parallel reference lines; do not overwrite or
   silently replace either baseline.
5. If no nested feature combination improves the quiet expert consistently,
   report the result as evidence that the current strict-30 observable views
   are insufficient for stable cross-event quiet-thread ranking.

## Research safety record

- Raw datasets, labels, event IDs, timestamps, protected splits, and previous
  experiment directories were not changed by the audits or Graph14 work.
- Every reported intervention target is outcome-only and is not a feature.
- Graph7 formal PHEME runs use event-separated LOEO and train-only transforms.
- Twitter source-level results must be labeled auxiliary; they are not a claim
  of cross-event generalization.
- The Graph14 development caveat above is intentionally retained for
  reproducibility and methodological transparency.
- The later strict weak-order v2 validation skipped Twitter test rows before
  parsing features or targets; its selected configuration was frozen before
  the one-time test evaluation.
- The PHEME two-expert gate and calibration are determined from outer-train or
  inner-OOF data only. Held-out outcomes are evaluation-only.
- Mutable engagement counters and mutable profile fields were deliberately
  excluded from the safe account-age, two-expert, semantic, residual, pairwise,
  sparse, and feature-combination experiments.
