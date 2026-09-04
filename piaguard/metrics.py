"""Metrics and figures for the Results and Discussion section.

Reported metrics and why each one is there:

  TPR@FPR<=t   detection rate at a fixed, deployment-relevant false-positive
               budget. The headline number - accuracy alone is meaningless on
               an unbalanced attack/clean mix.
  FPR          fraction of legitimate prompts blocked. The cost of the defence.
  AUROC        threshold-free separability, so the comparison against baselines
               does not depend on anyone's choice of theta.
  per-family   recall broken down by attack taxonomy class (direct, indirect,
               adversarial, backdoor, jailbreak) - shows where the method is
               weak, which is the honest part of the discussion.
  latency      ms per prompt, mean and p95, plus the per-layer breakdown.
"""
from typing import Dict, List, Sequence, Tuple
import numpy as np


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return {
        "tp": int(((yt == 1) & (yp == 1)).sum()),
        "fn": int(((yt == 1) & (yp == 0)).sum()),
        "fp": int(((yt == 0) & (yp == 1)).sum()),
        "tn": int(((yt == 0) & (yp == 0)).sum()),
    }


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    c = confusion(y_true, y_pred)
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        **{k: float(v) for k, v in c.items()},
        "precision": prec,
        "recall": rec,
        "tpr": rec,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "f1": f1,
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "balanced_accuracy": 0.5 * (rec + (tn / (tn + fp) if tn + fp else 0.0)),
    }


def roc_curve(y_true: Sequence[int], scores: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    yt, s = yt[order], s[order]
    n_pos, n_neg = max(1, int((yt == 1).sum())), max(1, int((yt == 0).sum()))
    tps = np.cumsum(yt == 1) / n_pos
    fps = np.cumsum(yt == 0) / n_neg
    return np.r_[0.0, fps], np.r_[0.0, tps], np.r_[np.inf, s]


def auroc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties handled by average ranks."""
    yt = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int((yt == 1).sum()), int((yt == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within tie groups
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_at_fpr(y_true: Sequence[int], scores: Sequence[float], target_fpr: float) -> float:
    """Lowest threshold whose empirical FPR is <= target_fpr.

    Searched over observed score values rather than taken from a quantile: a
    quantile is wrong whenever the score has heavy ties. The rule-filter
    baseline scores exactly 0.0 for most clean prompts, so its 95th percentile
    is 0.0, and `score >= 0.0` would flag every prompt - reporting a perfect
    TPR at 100% FPR. Picking the lowest qualifying threshold keeps TPR maximal
    subject to the false-positive budget.
    """
    s = np.asarray(scores, dtype=float)
    yt = np.asarray(y_true, dtype=int)
    clean = s[yt == 0]
    if clean.size == 0:
        return float("-inf")
    cands = np.unique(s)
    above_all = float(cands.max()) + 1e-9
    for t in cands:
        if float((clean >= t).mean()) <= target_fpr:
            return float(t)
    return above_all


def tpr_at_fpr(y_true: Sequence[int], scores: Sequence[float], target_fpr: float) -> Dict[str, float]:
    thr = threshold_at_fpr(y_true, scores, target_fpr)
    pred = [1 if s >= thr else 0 for s in scores]
    m = classification_metrics(y_true, pred)
    m["threshold"] = thr
    m["target_fpr"] = target_fpr
    return m


def per_family_recall(families: Sequence[str], y_true: Sequence[int],
                      y_pred: Sequence[int]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    fam = np.asarray(families, dtype=object)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    for f in sorted({x for x, y in zip(fam, yt) if y == 1}):
        m = (fam == f) & (yt == 1)
        out[str(f)] = {"n": int(m.sum()),
                       "recall": float(yp[m].mean()) if m.sum() else 0.0}
    return out


def latency_stats(ms: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(ms, dtype=float)
    if a.size == 0:
        return {}
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.quantile(a, 0.95)), "max_ms": float(a.max())}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_roc(curves: Dict[str, Tuple[Sequence[int], Sequence[float]]], path: str,
             subtitle: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    for label, (yt, sc) in curves.items():
        fpr, tpr, _ = roc_curve(yt, sc)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUROC={auroc(yt, sc):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="chance")
    _style(ax, "ROC - prompt injection detection" + (f"\n{subtitle}" if subtitle else ""),
           "False positive rate (legitimate prompts blocked)", "True positive rate (attacks caught)")
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_score_distribution(scores: Sequence[float], y_true: Sequence[int], theta: float,
                            path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.asarray(scores, dtype=float)
    yt = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=160)
    bins = np.linspace(float(s.min()), float(np.quantile(s, 0.99)) + 1e-6, 28)
    ax.hist(s[yt == 0], bins=bins, alpha=0.75, label="legitimate", color="#4C78A8")
    ax.hist(s[yt == 1], bins=bins, alpha=0.75, label="injection", color="#E45756")
    ax.axvline(theta, color="black", linestyle="--", linewidth=1.4,
               label=f"theta_block = {theta:.2f}")
    _style(ax, "Anomaly score separation", "Fused anomaly score", "Number of prompts")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_bars(labels: Sequence[str], values: Sequence[float], path: str, title: str,
              ylabel: str, ylim: Tuple[float, float] = None, fmt: str = "{:.2f}") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5.5, 1.25 * len(labels)), 4.2), dpi=160)
    bars = ax.bar(range(len(labels)), values, color="#4C78A8", width=0.62)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                ha="center", va="bottom", fontsize=9)
    _style(ax, title, "", ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
