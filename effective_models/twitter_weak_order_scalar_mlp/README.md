# Twitter15/16 Weak-order Scalar MLP

- Frozen test reduction@10: Twitter15 **0.1986**, Twitter16 **0.3292**.
- Model: 31 inputs -> Linear(32) -> ReLU -> Dropout(0.05) -> scalar score.
- Target: tail-weighted Smooth-L1 on `log1p(preventable_impact)` plus a weak
  within-train Oracle-head order loss.
- Validation selected order weight 0.05 from {0, 0.02, 0.05, 0.1}; test was
  evaluated once after freezing.

The strict feature artifact is stored locally under `artifacts/`, and the
frozen validation/test outputs are under `reference_result/`.
The original test has already been consumed and must not be reused for tuning.

```powershell
.\venv\Scripts\python.exe .\effective_models\twitter_weak_order_scalar_mlp\training\run_smoke.py
.\venv\Scripts\python.exe .\effective_models\twitter_weak_order_scalar_mlp\training\run_full.py
```

Both commands use `training/common.py`. The full command only reproduces the
already-frozen test protocol; its result must not be used for further tuning.
