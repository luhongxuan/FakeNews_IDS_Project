
# AGENTS.md

## Project Purpose

This repository is a research project for early detection / early intervention
of misinformation propagation.

The main goal is to develop, train, evaluate, and compare models that can make
predictions using only information available at an early observation time.

Correct research methodology is more important than improving a metric.

---

# 1. General Working Rules

Before modifying code:

1. Read the relevant files first.
2. Understand the current data flow and training pipeline.
3. Identify which files need to be changed.
4. Make the smallest reasonable change.
5. Do not refactor unrelated code unless explicitly requested.
6. Do not silently change experiment assumptions.

When a task is ambiguous, prefer inspecting the repository before making assumptions.

Do not claim that code, tests, training, or evaluation succeeded unless they were
actually executed successfully.

---

# 2. Research Safety Rules

Research validity has higher priority than model performance.

Never improve performance by changing the evaluation conditions.

Do not:

- modify labels to improve results
- move samples between train / validation / test sets
- regenerate dataset splits without explicit approval
- use test data for model selection
- use future information in early detection experiments
- compare models using different dataset splits without clearly reporting it
- change metric definitions silently
- remove difficult samples or failed cases only to improve metrics
- overwrite previous experiment results

If any potential data leakage is found, stop and report it before continuing.

---

# 3. Early Detection / Early Intervention Rules

For an observation cutoff at time T:

Only information available at or before T may be used.

Information created after T must not be used in:

- node features
- edge construction
- graph statistics
- temporal features
- model inputs
- normalization statistics
- labels derived from future observations
- feature engineering

The full propagation graph must never be used accidentally when training or
evaluating an early-observation model.

When creating temporal snapshots, verify that:

- every node timestamp <= cutoff
- every edge only connects observable nodes
- no future feature is included
- snapshot construction is deterministic for the same input and cutoff

If these conditions cannot be verified, report the issue instead of continuing
with training.

---

# 4. Dataset Protection

Treat existing datasets and predefined dataset splits as protected research assets.

Do not modify, delete, rename, regenerate, or overwrite them unless the user
explicitly requests it.

Especially protect:

- raw datasets
- processed datasets used by existing experiments
- labels
- train / validation / test split definitions
- event IDs
- timestamps

New derived datasets may be created when necessary, but they must be written to
a new location and must not overwrite the original data.

Always preserve traceability from derived data back to the original event/sample.

---

# 5. Train / Validation / Test Rules

Train, validation, and test data must remain separated.

Do not use:

- test labels during training
- test performance to tune hyperparameters
- test performance to choose checkpoints
- test samples for feature normalization
- information from the same event across different splits when the experiment
  requires event-separated evaluation

Use validation data for model selection.

Use the test set only for final evaluation unless the experiment specification
explicitly defines another protocol.

If overlap between splits is detected, report it immediately.

---

# 6. Model Development Rules

Codex may:

- add a new model
- modify model architecture
- modify loss functions
- add features
- add configuration files
- improve training code
- improve evaluation code
- add tests
- add logging
- fix bugs

Codex must NOT modify dataset definitions or evaluation conditions just because
a model performs poorly.

When modifying a model:

1. State the hypothesis behind the modification.
2. Keep the change focused.
3. Record the important hyperparameters.
4. Preserve the previous model or experiment configuration.
5. Compare the new model against an appropriate baseline.

Avoid combining many unrelated architectural changes in a single experiment.

Prefer one clear experimental change at a time.

---

# 7. Experiment Reproducibility

Every important experiment should record:

- experiment name
- model name
- dataset
- dataset split
- observation window / cutoff
- random seed
- hyperparameters
- training configuration
- evaluation metrics
- final results

Set random seeds where applicable.

Do not overwrite an existing experiment directory.

Create a new experiment directory or identifier for every meaningful run.

If an experiment fails, keep enough information to understand why it failed.

### Required Script Progress and Saved-Run Reporting

Any script that creates a derived dataset, trains a model, runs an evaluation,
or produces a diagnostic / visualization must make its progress visible in the
terminal. Do not leave the user with a single long-running status line or an
opaque background process.

At a minimum, such a script must:

- print a clear start message and the intended output directory before costly
  work begins;
- print meaningful stage transitions (for example: loading, validating,
  snapshot construction, training fold / epoch progress, evaluation, writing
  results);
- for long loops, print bounded periodic progress with completed and total
  units, so the user can confirm it is still running;
- save a run record in its new output directory containing the configuration,
  input artifact paths / versions, split, cutoff, seed where applicable,
  metrics, and final status;
- preserve failure information and the output directory when a run fails, when
  practical; and
- print an unambiguous final line on success or failure that includes the exact
  absolute path of the saved output directory.

For direct user execution, prefer self-contained no-argument scripts with
visible configuration at the top of the file, unless command-line parameters
are specifically requested. Do not silently run expensive work in the
background.

### Assistant Execution and User Handoff Policy

Codex should execute short, bounded checks itself when they are needed to make
progress. This includes syntax checks, unit tests, small data/leakage audits,
smoke runs, and simple validation or evaluation scripts that are clearly not
long-running and do not perform full model training.

For these short runs, Codex must:

- actually execute the check instead of asking the user to run it;
- inspect the saved result and terminal output;
- fix in-scope failures and rerun the relevant short check when safe;
- use the result to decide and perform the next safe pipeline step; and
- report exactly what was executed and what passed or failed.

Codex must not start a full model training run, exhaustive or nested search,
large dataset build, or other expected long-running computation. Unless the
user explicitly overrides this rule for a particular run, Codex must stop
before that stage and give the user:

- the exact script to run;
- the exact foreground terminal command;
- the model/configuration, dataset, split, cutoff, and expected outputs;
- a short statement explaining why the run is considered long-running; and
- the successful smoke, safety, and leakage checks that justify starting it.

The long-running script must be prepared for direct foreground execution by the
user. Do not launch it in the background, and do not start it merely to see
whether it works.

Every research script run, including smoke, validation, evaluation, diagnostic,
visualization, dataset construction, and training, must leave a complete run
record in a new output directory. The record must contain at least start and
completion timestamps, current status (`running`, `complete`, or `failed`),
configuration, exact input paths or artifact identifiers, split, cutoff, seed
where applicable, metrics/results, output file inventory, and failure details
when applicable. A successful terminal message without saved result files is
not sufficient evidence of completion.

Scripts intended for user execution must flush terminal output and show:

- a start line and absolute output directory before work begins;
- numbered stage progress such as `[2/6]`;
- fold, epoch, combination, batch, or item progress as completed/total for long
  loops, printed at bounded intervals;
- evaluation and result-writing stages; and
- one unambiguous final `SUCCESS` or `FAILURE` line containing the absolute
  output directory.

---

# 8. Baseline Protection

Existing baseline results are reference results.

Do not overwrite or silently replace a baseline.

A new model that performs better should be reported as a candidate improvement.

Do not promote a new result to the official baseline unless explicitly requested.

When comparing models, ensure that they use the same:

- dataset
- split
- observation window
- evaluation protocol

Otherwise clearly state that the results are not directly comparable.

---

# 9. Training Safety Pipeline

Do not immediately start a long full-training run after modifying code.

Use the following sequence:

1. Inspect
2. Implement
3. Static / syntax checks
4. Unit tests
5. Data and leakage validation
6. Smoke training
7. Full training
8. Evaluation
9. Experiment comparison
10. Report

If any stage fails, fix or report the failure before moving to the next stage.

---

# 10. Smoke Test Rules

Before full training, perform a small smoke test whenever possible.

The smoke test should use:

- a small subset of training data
- very few epochs
- the normal data pipeline
- the normal forward / loss / backward pipeline

Verify at least:

- dataset can be loaded
- tensor / graph shapes are valid
- forward pass succeeds
- loss is finite
- backward pass succeeds
- predictions have the expected shape
- validation code runs
- metrics can be calculated

A successful smoke test does NOT prove that the model is correct.
It only verifies that the pipeline can execute.

Smoke tests are short checks and should normally be executed by Codex. After a
smoke failure, Codex should inspect the error, make the smallest in-scope fix,
and rerun the smoke test. After success, Codex may continue through other short
validation steps, but must stop and hand off before full or long-running model
training as required by the Assistant Execution and User Handoff Policy.

---

# 11. Full Training

Do not start full, expensive, or expected long-running model training. Prepare
and validate the script, then hand the exact foreground command to the user for
manual execution. Only run such training when the user explicitly overrides
this repository rule for that specific run.

Before full training, report:

- model
- configuration
- dataset
- split
- observation window
- expected command
- expected progress display and output directory behavior
- completed smoke, safety, and leakage checks

Do not silently start multiple expensive experiments.

---

# 12. Evaluation Rules

Do not rely only on accuracy.

Use the metrics already defined by the project whenever possible.

For classification tasks, relevant metrics may include:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC

For ranking / intervention tasks, use the project's defined ranking or correlation
metrics when applicable.

Do not add or remove evaluation metrics merely to make results look better.

Report negative results as well as positive results.

---

# 13. File Modification Rules

Before editing:

- inspect the relevant file
- understand its callers and dependencies

Prefer modifying only files directly related to the requested task.

Do not:

- delete large directories
- delete datasets
- delete checkpoints
- delete experiment history
- perform destructive Git operations
- rewrite unrelated parts of the repository

without explicit approval.

---

# 14. Dependency Rules

Do not install new packages without explicit approval.

If a new dependency is necessary:

1. explain why it is needed
2. check whether an existing dependency can solve the problem
3. report the package name
4. wait for approval before installing it

Do not silently upgrade existing dependencies.

---

# 15. Command Safety

Safe read-only inspection commands may be executed as needed.

Examples include:

- listing files
- searching code
- reading files
- git status
- git diff

Be cautious with commands that:

- modify many files
- install software
- download files
- access external networks
- delete files
- reset Git state
- launch expensive training jobs

Never use destructive commands such as hard reset or recursive deletion unless
the user explicitly requests and approves them.

---

# 16. Network Access

Do not send project data, dataset contents, experiment outputs, credentials, or
private files to external services, except for the narrow user-owned RunPod
inference exception below.

Do not access the network unless it is necessary for the requested task and
permitted by the current environment.

Never expose:

- API keys
- passwords
- tokens
- environment secrets
- private dataset contents outside the explicitly permitted cutoff-safe RunPod
  payload described below

### User-Owned RunPod Inference Exception

PHEME Agent experiments may send only the minimum cutoff-safe inference payload
to a RunPod endpoint owned and explicitly enabled by the user for that
experiment. This exception is for model inference only; it does not authorize
uploading or synchronizing the raw dataset, repository, database, experiment
directories, or unrelated project files.

Every experiment using this exception must:

- require an explicit configuration flag such as `ALLOW_REMOTE=true`;
- send only source text, replies, timestamps, graph summaries, and derived
  features observable at or before the configured observation cutoff;
- exclude post-cutoff content, future-growth or preventable-impact labels,
  Oracle membership, final veracity, dataset split metadata, and other
  evaluation answers from the inference payload;
- never transmit API keys, passwords, tokens, `.env` contents, credentials, or
  private infrastructure details as part of prompts or saved public outputs;
- use access control and encrypted transport for the RunPod endpoint;
- record the remote-use authorization, provider, model, cutoff, exact payload
  field names, prompt/schema version or hash, cache hit/miss counts, and actual
  inference-call count in the experiment run record;
- redact endpoint credentials and query tokens from logs and run records; and
- keep model, prompt, decoding configuration, and evaluation protocol fixed
  across directly compared events or clearly report any difference.

User authorization for this RunPod exception does not relax any early-detection,
split-separation, leakage, reproducibility, or evaluation rule elsewhere in this
file.

---

# 17. Coding Quality

Prefer:

- clear functions
- small focused modules
- descriptive variable names
- type hints when appropriate
- reusable utilities
- configuration instead of hardcoded experiment parameters

Avoid:

- duplicated logic
- unexplained magic numbers
- hidden fallback behavior
- broad exception handling that hides errors
- hardcoded absolute paths
- unnecessary large refactors

Do not change working code only for stylistic reasons during a research experiment.

---

# 18. Required Final Report

After completing a coding task, report:

## What changed

List the files modified and the purpose of each change.

## Why

Explain the hypothesis or reason for the change.

## Validation

Report which tests or checks were actually executed.

## Experiment

If training was performed, report the experiment configuration.

## Results

Report relevant metrics without exaggerating improvements.

## Research Safety

Confirm whether:

- dataset split was unchanged
- labels were unchanged
- observation cutoff rules were respected
- no known future-information leakage was introduced

## Remaining Issues

List known limitations, failures, or uncertainties.

## Recommended Next Step

Suggest the next reasonable experiment, but do not automatically start it unless
requested.
