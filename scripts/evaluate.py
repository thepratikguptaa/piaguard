"""Full evaluation: calibration, ablation, baseline comparison, figures.

    python scripts/evaluate.py --model gpt2 --target-fpr 0.01
    python scripts/evaluate.py --model gpt2 --extra-csv path/to/benchmark.csv

Writes to results/:
    table1_main_results.csv     proposed pipeline vs baselines
    table2_ablation.csv         contribution of each layer
    table3_per_family.csv       recall by attack taxonomy class
    table4_latency.csv          per-layer latency budget
    per_prompt_scores.csv       raw scores (for your own plots / appendix)
    fig1_roc.png  fig2_score_distribution.png
    fig3_per_family_recall.png  fig4_ablation.png  fig5_latency.png
    run_metadata.json           model, thresholds, dataset sizes, config

If torch/transformers or the model weights are unavailable the run falls back to
MockCausalLM and every output file is prefixed SMOKE_ - those numbers are a
plumbing check, not results. Do not put them in the report.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from piaguard import (PipelineConfig, PIAGuardPipeline, load_lm, MockCausalLM,
                      load_eval_set, calibration_split)
from piaguard.dataset import summarise
from piaguard.baselines import KeywordFilter, PerplexityFilter, RandomBaseline
from piaguard import metrics as M


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2", help="HF causal LM used for scoring")
    p.add_argument("--data", default=None, help="eval CSV (default: seed set)")
    p.add_argument("--extra-csv", nargs="*", default=None, help="additional labelled CSVs")
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--window-sizes", type=int, nargs="+", default=[1, 3])
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-positions", type=int, default=None,
                   help="cap on masked variants per prompt (default: DetectorConfig, 96). "
                        "Dominates runtime - each variant is one forward pass.")
    p.add_argument("--no-ablation", action="store_true",
                   help="skip the window-size ablation, which re-scores the whole test "
                        "set once per window config and is ~70%% of total runtime")
    p.add_argument("--outdir", default=None)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(outdir, exist_ok=True)

    cfg = PipelineConfig()
    cfg.detector.model_name = args.model
    cfg.detector.window_sizes = tuple(args.window_sizes)
    cfg.detector.max_tokens = args.max_tokens
    cfg.detector.batch_size = args.batch_size
    if args.max_positions is not None:
        cfg.detector.max_positions = args.max_positions
    cfg.gate.target_fpr = args.target_fpr
    cfg.seed = args.seed

    print("=" * 74)
    print("PIAGuard evaluation")
    print("=" * 74)

    data = load_eval_set(args.data, args.extra_csv)
    print(summarise(data))

    lm = load_lm(cfg.detector)
    is_mock = isinstance(lm, MockCausalLM)
    prefix = "SMOKE_" if is_mock else ""
    if is_mock:
        print("\n*** MOCK MODEL - outputs are a plumbing check, NOT results ***\n")
    else:
        print(f"\nscoring model: {lm.name} on {getattr(lm, 'device', '?')}\n")

    pipe = PIAGuardPipeline(lm, cfg)

    # ---------- 1. calibrate theta on clean prompts only ----------
    cal, test = calibration_split(data, frac=0.4, seed=args.seed)
    print(f"[1/5] calibrating theta on {len(cal)} clean prompts (never scored at test time)")
    t0 = time.time()
    cal_res = pipe.analyse_many([e.text for e in cal])
    cal_fused = [r.gate.fused_score for r in cal_res]
    theta = pipe.gate.calibrate(cal_fused, target_fpr=args.target_fpr)
    print(f"      theta_block={theta['theta_block']:.3f}  theta_review={theta['theta_review']:.3f}"
          f"  (target FPR {args.target_fpr:.1%})  [{time.time() - t0:.1f}s]")

    # ---------- 2. score the test set ----------
    print(f"[2/5] scoring {len(test)} held-out prompts")
    t0 = time.time()
    res = pipe.analyse_many([e.text for e in test])
    print(f"      done in {time.time() - t0:.1f}s")

    y_true = [e.label for e in test]
    fam = [e.family for e in test]
    fused = [r.gate.fused_score for r in res]
    det_only = [r.detect.score for r in res]
    san_only = [r.sanitize.risk for r in res]
    y_pred = [1 if r.blocked else 0 for r in res]

    per_prompt = pd.DataFrame([{
        "id": e.id, "family": e.family, "label": e.label,
        **r.to_row(),
    } for e, r in zip(test, res)])
    per_prompt.to_csv(os.path.join(outdir, f"{prefix}per_prompt_scores.csv"), index=False)

    # ---------- 3. baselines ----------
    print("[3/5] running baselines")
    texts = [e.text for e in test]
    ppl = PerplexityFilter(lm, max_tokens=args.max_tokens).score_many(texts)
    rnd = RandomBaseline(args.seed).score_many(texts)

    systems = {
        "Proposed (L1+L2+L3 fused)": fused,
        "Loss-shift only (L2)": det_only,
        "Rule/keyword filter (L1)": san_only,
        "Perplexity threshold": ppl,
        "Random": rnd,
    }

    rows = []
    for name, scores in systems.items():
        m = M.tpr_at_fpr(y_true, scores, args.target_fpr)
        rows.append({
            "System": name,
            f"TPR @ FPR<={args.target_fpr:.0%}": round(m["tpr"], 4),
            "FPR": round(m["fpr"], 4),
            "Precision": round(m["precision"], 4),
            "F1": round(m["f1"], 4),
            "Balanced acc.": round(m["balanced_accuracy"], 4),
            "AUROC": round(M.auroc(y_true, scores), 4),
            "theta": round(m["threshold"], 4),
        })
    table1 = pd.DataFrame(rows)
    table1.to_csv(os.path.join(outdir, f"{prefix}table1_main_results.csv"), index=False)

    # operating-point metrics for the calibrated pipeline (theta from clean set)
    op = M.classification_metrics(y_true, y_pred)

    # ---------- 4. ablation ----------
    print("[4/5] ablation")
    abl_rows = []
    for name, scores in [
        ("L1 only (rules)", san_only),
        ("L2 only (loss shift)", det_only),
        ("L1 + L2 (fused, w=1.5)", fused),
    ]:
        m = M.tpr_at_fpr(y_true, scores, args.target_fpr)
        abl_rows.append({"Configuration": name,
                         "TPR": round(m["tpr"], 4), "FPR": round(m["fpr"], 4),
                         "F1": round(m["f1"], 4),
                         "AUROC": round(M.auroc(y_true, scores), 4)})
    # window-size ablation on the detector alone - re-scores every test prompt once
    # per config, so it costs about as much as steps 1-3 put together
    window_configs = [] if args.no_ablation else [(1,), (3,), (1, 3)]
    if args.no_ablation:
        print("      window-size ablation skipped (--no-ablation)")
    for ws in window_configs:
        pipe.detector.cfg.window_sizes = ws
        sc = [pipe.detector.score(pipe.sanitizer.run(t).text).score for t in texts]
        m = M.tpr_at_fpr(y_true, sc, args.target_fpr)
        abl_rows.append({"Configuration": f"L2 window sizes {ws}",
                         "TPR": round(m["tpr"], 4), "FPR": round(m["fpr"], 4),
                         "F1": round(m["f1"], 4),
                         "AUROC": round(M.auroc(y_true, sc), 4)})
    pipe.detector.cfg.window_sizes = tuple(args.window_sizes)
    table2 = pd.DataFrame(abl_rows)
    table2.to_csv(os.path.join(outdir, f"{prefix}table2_ablation.csv"), index=False)

    # ---------- 5. per-family, latency, figures ----------
    print("[5/5] per-family breakdown, latency, figures")
    pf = M.per_family_recall(fam, y_true, y_pred)
    table3 = pd.DataFrame([{"Attack family": k, "n": v["n"],
                            "Recall (detected)": round(v["recall"], 4)}
                           for k, v in pf.items()])
    table3.to_csv(os.path.join(outdir, f"{prefix}table3_per_family.csv"), index=False)

    lat_rows = []
    for layer in ["l1_sanitize", "l2_detect", "l3_gate", "total"]:
        vals = [r.timings_ms.get(layer, 0.0) for r in res]
        s = M.latency_stats(vals)
        lat_rows.append({"Stage": layer, **{k: round(v, 2) for k, v in s.items()}})
    table4 = pd.DataFrame(lat_rows)
    table4.to_csv(os.path.join(outdir, f"{prefix}table4_latency.csv"), index=False)

    sub = f"{lm.name}, seed set n={len(test)}" + ("  [MOCK - not results]" if is_mock else "")
    M.plot_roc({"Proposed (fused)": (y_true, fused),
                "Loss-shift only": (y_true, det_only),
                "Rule filter only": (y_true, san_only),
                "Perplexity": (y_true, ppl)},
               os.path.join(outdir, f"{prefix}fig1_roc.png"), subtitle=sub)
    M.plot_score_distribution(fused, y_true, pipe.gate.cfg.theta_block,
                              os.path.join(outdir, f"{prefix}fig2_score_distribution.png"))
    if len(table3):
        M.plot_bars(list(table3["Attack family"]), list(table3["Recall (detected)"]),
                    os.path.join(outdir, f"{prefix}fig3_per_family_recall.png"),
                    "Detection recall by attack family", "Recall", ylim=(0, 1.05))
    M.plot_bars(list(table2["Configuration"]), list(table2["TPR"]),
                os.path.join(outdir, f"{prefix}fig4_ablation.png"),
                f"Ablation: TPR at FPR<={args.target_fpr:.0%}", "TPR", ylim=(0, 1.05))
    M.plot_bars([r["Stage"] for r in lat_rows], [r.get("mean_ms", 0) for r in lat_rows],
                os.path.join(outdir, f"{prefix}fig5_latency.png"),
                "Mean latency per prompt by stage", "milliseconds", fmt="{:.1f}")

    meta = {
        "model": lm.name, "mock": is_mock, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_calibration": len(cal), "n_test": len(test),
        "n_attacks_test": int(sum(y_true)), "target_fpr": args.target_fpr,
        "theta": theta, "operating_point": {k: round(v, 4) for k, v in op.items()},
        "detector": {"window_sizes": list(cfg.detector.window_sizes),
                     "max_tokens": cfg.detector.max_tokens,
                     "aggregation": cfg.detector.aggregation,
                     "mask_strategy": cfg.detector.mask_strategy,
                     "max_positions": cfg.detector.max_positions},
        "window_ablation": not args.no_ablation,
        "sanitizer_weight": cfg.gate.sanitizer_weight,
    }
    with open(os.path.join(outdir, f"{prefix}run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---------- console summary ----------
    print("\n" + "=" * 74)
    print("TABLE 1 - main results")
    print(table1.to_string(index=False))
    print("\nTABLE 2 - ablation")
    print(table2.to_string(index=False))
    print("\nTABLE 3 - per attack family (at calibrated theta)")
    print(table3.to_string(index=False))
    print("\nTABLE 4 - latency (ms)")
    print(table4.to_string(index=False))
    print("\nCalibrated operating point: "
          f"TPR={op['tpr']:.3f}  FPR={op['fpr']:.3f}  F1={op['f1']:.3f}")
    print(f"\nfiles written to {os.path.abspath(outdir)}")
    if is_mock:
        print("\n*** SMOKE_ prefix => mock model. Re-run with a real model for reportable numbers. ***")


if __name__ == "__main__":
    main()
