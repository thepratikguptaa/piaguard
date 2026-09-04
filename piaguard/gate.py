"""Layer 3 - Decision Gate.

Fuses the two independent signals into one verdict:

    fused = detector_score + sanitizer_weight * sanitizer_risk

and maps it to ALLOW / REVIEW / BLOCK. The REVIEW band exists because a binary
gate forces every borderline prompt into either a false block (a user loses
service) or a false allow (an attack lands); REVIEW routes it to a stricter
system prompt, a smaller max-token budget, or a human, depending on deployment.

Thresholds are not hand-picked. `calibrate` sets theta from the empirical
quantile of fused scores on a CLEAN-ONLY calibration set at a chosen false
positive rate. That way theta is stated as a deployment SLA ("1% of legitimate
traffic gets blocked") rather than a magic number, and it can be re-derived for
any new model or traffic mix without touching the code.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import numpy as np


class Verdict(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass
class GateResult:
    verdict: Verdict
    fused_score: float
    detector_score: float
    sanitizer_risk: float
    theta_block: float
    theta_review: float
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.verdict is Verdict.BLOCK

    def to_dict(self) -> Dict:
        return {
            "verdict": self.verdict.value,
            "fused_score": round(self.fused_score, 4),
            "theta_block": round(self.theta_block, 4),
            "reason": self.reason,
        }


class DecisionGate:
    def __init__(self, config=None):
        from .config import GateConfig
        self.cfg = config or GateConfig()

    def fuse(self, detector_score: float, sanitizer_risk: float) -> float:
        return float(detector_score + self.cfg.sanitizer_weight * sanitizer_risk)

    def decide(self, detector_score: float, sanitizer_risk: float,
               sanitizer_families: Optional[List[str]] = None) -> GateResult:
        fused = self.fuse(detector_score, sanitizer_risk)

        if fused >= self.cfg.theta_block:
            verdict = Verdict.BLOCK
        elif self.cfg.use_three_way and fused >= self.cfg.theta_review:
            verdict = Verdict.REVIEW
        else:
            verdict = Verdict.ALLOW

        bits = []
        if detector_score >= self.cfg.theta_review:
            bits.append(f"loss-shift anomaly {detector_score:.2f}")
        if sanitizer_families:
            bits.append("patterns: " + ",".join(sanitizer_families))
        reason = "; ".join(bits) or "no anomaly above threshold"

        return GateResult(verdict, fused, detector_score, sanitizer_risk,
                          self.cfg.theta_block, self.cfg.theta_review, reason)

    # ---------- calibration ----------
    def calibrate(self, clean_fused_scores: List[float],
                  target_fpr: Optional[float] = None) -> Dict[str, float]:
        """Set theta_block at the (1 - target_fpr) quantile of clean scores.

        theta_review is placed at the 85th percentile of clean scores, i.e. the
        band the top ~15% of legitimate-but-unusual prompts fall into.
        """
        fpr = target_fpr if target_fpr is not None else self.cfg.target_fpr
        arr = np.asarray(clean_fused_scores, dtype=np.float64)
        if arr.size == 0:
            raise ValueError("calibration set is empty")
        self.cfg.theta_block = float(np.quantile(arr, 1.0 - fpr))
        self.cfg.theta_review = float(np.quantile(arr, 0.85))
        self.cfg.target_fpr = fpr
        return {
            "theta_block": self.cfg.theta_block,
            "theta_review": self.cfg.theta_review,
            "target_fpr": fpr,
            "n_calibration": int(arr.size),
        }
