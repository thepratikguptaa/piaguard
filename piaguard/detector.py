"""Layer 2 - Injection Detection by masked-span loss shift.

Intuition
---------
An injected instruction is a *trigger*: a short span that the surrounding text
does not predict, and that dominates how the model continues. If you delete that
span, the model's mean cross-entropy over what remains drops sharply. Ordinary
prompts have no such span - removing any one piece changes the loss only a little.

Method
------
    L0        = mean NLL of the full (sanitised) prompt
    L(i,w)    = mean NLL of the prompt with the w tokens at position i removed
    d(i,w)    = L0 - L(i,w)                       (positive => that span was costly)
    score     = (max d - median d) / (1.4826 * MAD + eps)

The robust z-score is the important detail: raw max-shift depends on prompt
length and topic, so a single global threshold does not transfer. Dividing by
the spread of the *same prompt's* shifts makes the score self-normalising, which
is what lets one calibrated theta hold across prompt lengths.

This extends UniGuardian's single-token masking with multi-token windows
(window_sizes), because real injected instructions are phrases, not single
tokens - masking one token of "ignore previous instructions" leaves the rest of
the trigger intact and the loss barely moves.

Cost
----
All masked variants for a prompt are evaluated in batched forward passes, so
cost is O(n_variants / batch_size) passes, no fine-tuning and no gradients:
the defence is training-free and model-agnostic.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np

EPS = 1e-6


@dataclass
class Span:
    start: int
    end: int
    shift: float
    text: str = ""


@dataclass
class DetectResult:
    score: float
    baseline_nll: float
    top_spans: List[Span] = field(default_factory=list)
    n_variants: int = 0
    n_tokens: int = 0

    def to_dict(self) -> Dict:
        return {
            "detector_score": round(self.score, 4),
            "baseline_nll": round(self.baseline_nll, 4),
            "n_tokens": self.n_tokens,
            "n_variants": self.n_variants,
            "top_span": self.top_spans[0].text if self.top_spans else "",
            "top_shift": round(self.top_spans[0].shift, 4) if self.top_spans else 0.0,
        }


class LossShiftDetector:
    def __init__(self, lm, config=None):
        from .config import DetectorConfig
        self.lm = lm
        self.cfg = config or DetectorConfig()

    # ---------- variant construction ----------
    def _variants(self, ids: List[int]) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        seqs, spans = [], []
        n = len(ids)
        # "replace" keeps the sequence length constant, so the mean NLL is not
        # affected by the length change that deletion causes. It needs a filler
        # that carries no meaning: repeating ids[0] would splice the prompt's own
        # first token across the span and inject fresh signal.
        mask_id = getattr(self.lm, "mask_token_id", ids[0]) \
            if self.cfg.mask_strategy != "delete" else None
        for w in self.cfg.window_sizes:
            if w >= n:
                continue
            stride = 1 if w == 1 else max(1, w // 2)   # overlap wider windows
            for i in range(0, n - w + 1, stride):
                if self.cfg.mask_strategy == "delete":
                    seqs.append(ids[:i] + ids[i + w:])
                else:  # replace the span with a neutral mask token (keeps length)
                    seqs.append(ids[:i] + [mask_id] * w + ids[i + w:])
                spans.append((i, i + w))
        # cost cap: keep an evenly spread subset
        if len(seqs) > self.cfg.max_positions:
            keep = np.linspace(0, len(seqs) - 1, self.cfg.max_positions).astype(int)
            seqs = [seqs[k] for k in keep]
            spans = [spans[k] for k in keep]
        return seqs, spans

    # ---------- aggregation ----------
    def _aggregate(self, shifts: np.ndarray) -> float:
        if shifts.size == 0:
            return 0.0
        if self.cfg.aggregation == "max":
            return float(shifts.max())
        if self.cfg.aggregation == "topk_mean":
            k = min(self.cfg.topk, shifts.size)
            return float(np.sort(shifts)[-k:].mean())
        # robust_z (default)
        med = float(np.median(shifts))
        mad = float(np.median(np.abs(shifts - med)))
        scale = 1.4826 * mad + EPS
        return float((shifts.max() - med) / scale)

    def score(self, text: str) -> DetectResult:
        ids = self.lm.tokenize(text)[: self.cfg.max_tokens]
        if len(ids) < 4:
            return DetectResult(score=0.0, baseline_nll=0.0, n_tokens=len(ids))

        baseline = float(self.lm.mean_nll([ids])[0])
        seqs, spans = self._variants(ids)
        if not seqs:
            return DetectResult(score=0.0, baseline_nll=baseline, n_tokens=len(ids))

        losses = np.empty(len(seqs), dtype=np.float32)
        bs = max(1, self.cfg.batch_size)
        for b in range(0, len(seqs), bs):
            losses[b:b + bs] = self.lm.mean_nll(seqs[b:b + bs])

        shifts = baseline - losses                     # positive => costly span
        score = self._aggregate(shifts)

        order = np.argsort(shifts)[::-1][:3]
        top = [Span(spans[j][0], spans[j][1], float(shifts[j]),
                    self.lm.decode(ids[spans[j][0]:spans[j][1]])) for j in order]

        return DetectResult(score=score, baseline_nll=baseline, top_spans=top,
                            n_variants=len(seqs), n_tokens=len(ids))

    def score_batch(self, texts: List[str]) -> List[DetectResult]:
        return [self.score(t) for t in texts]
