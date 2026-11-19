# DermAI Milestone 2 — Validation Model Comparison

> Models ranked by **val macro-F1** (higher is better).
> The **best model** (highlighted with ✅) should then be evaluated
> once on the test split using `--run_test`.

## Summary Table

| Model | Params (M) | Train (min) | Accuracy | Bal-Acc | Macro-F1 | Wt-F1 | Macro-AUC | MCC | mel-Sensitivity | mel-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| ✅ **resnet50** | 23.52 | 2.7 | 0.4589 | 0.1494 | 0.1355 | 0.4769 | 0.5648 | 0.0392 | 0.2235 | 0.5224 |

## Per-Class F1 (val)

| Model | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| ✅ **resnet50** | 0.0000 | 0.0000 | 0.1120 | 0.0212 | 0.1514 | 0.6643 | 0.0000 |

---
**Best model by val macro-F1:** `resnet50`

> [!IMPORTANT]
> Run the test split **only once** for `{best_model}` using:
> ```bash
> python -m ml.evaluation.compare_models --run_test ...
> ```

