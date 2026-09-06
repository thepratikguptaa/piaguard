# What is in this folder

Several evaluation runs live here. Only some of them are current. Check
`run_metadata.json` in any folder before quoting a number — it records the model,
the aggregation, θ and the timestamp, and `"mock": true` means the numbers are fake.

| Folder | Model | Aggregation / windows | Use it? |
|---|---|---|---|
| `./` (top level) | gpt2 | `topk_mean`, `(1, 3)` | **Yes — these are the headline numbers.** Tables 1–4 and Figures 1–5 of the report. |
| `gpt2-medium_topk/` | gpt2-medium | `topk_mean`, `(1, 3)` | **Yes** — the model-size comparison (Section 6.3 of the guide). |
| `w3/` | gpt2 | `topk_mean`, `(3,)` | **Yes, for the window-size trade-off only** (Section 6.2): 2.6× faster L2 and better AUROC, one attack less recall. Run with `--no-ablation`. |
| `agg_robust_z/`, `agg_max/`, `agg_topk_mean/` | gpt2 | as named, `(1, 3)` | **Yes, but only for the aggregation sweep** (Section 6.7). Run with `--no-ablation`, so no window-size rows. |
| `gpt2/`, `gpt2-medium/` | gpt2, gpt2-medium | `robust_z`, `(1, 3)` | **No — superseded.** Produced before the aggregation defect was found. Kept only as the "before" evidence in Section 6.7. Do not report these as results. |
| any `SMOKE_*` file | mock | — | **Never.** Produced by `MockCausalLM`, which fakes surprisal with a hash function. Plumbing check only. |

Regenerate the headline numbers with:

```bash
python scripts/evaluate.py --model gpt2 --target-fpr 0.01 --batch-size 32
```

If that aborts saying the model could not be loaded, the interpreter you ran it with
has no `torch` — the error names it and the command to check. That guard exists because
mock tables have been mistaken for results before; it is not a bug.

See `docs/PIAGuard_Explained.pdf` for what every table and figure means.
