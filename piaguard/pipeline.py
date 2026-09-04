"""End-to-end orchestration of the four defence layers.

    prompt -> [L1 Sanitizer] -> [L2 LossShiftDetector] -> [L3 DecisionGate]
                                                              |
                                          BLOCK <-------------+-------------> ALLOW / REVIEW
                                                                                    |
                                                                              LLM response
                                                                                    |
                                                                        [L4 OutputValidator]
                                                                                    |
                                                                          safe response to user

Every stage is timed so the Results section can report a real latency budget and
show which layer dominates cost (it is L2, by an order of magnitude).
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Any, List
import time

from .config import PipelineConfig
from .sanitizer import Sanitizer, SanitizeResult
from .detector import LossShiftDetector, DetectResult
from .gate import DecisionGate, GateResult, Verdict
from .validator import OutputValidator, ValidateResult

BLOCK_MESSAGE = ("This request was blocked by the input-integrity check "
                 "(possible prompt injection).")


@dataclass
class PipelineResult:
    prompt: str
    verdict: str
    sanitize: SanitizeResult = None
    detect: DetectResult = None
    gate: GateResult = None
    validate: Optional[ValidateResult] = None
    response: str = ""
    timings_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.verdict == Verdict.BLOCK.value

    def to_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"prompt": self.prompt[:160], "verdict": self.verdict}
        if self.sanitize:
            row.update({"sanitizer_" + k: v for k, v in self.sanitize.to_dict().items()})
        if self.detect:
            row.update(self.detect.to_dict())
        if self.gate:
            row.update(self.gate.to_dict())
        if self.validate:
            row.update(self.validate.to_dict())
        row.update({"t_" + k: round(v, 2) for k, v in self.timings_ms.items()})
        return row


class PIAGuardPipeline:
    def __init__(self, lm, config: Optional[PipelineConfig] = None,
                 system_prompt: str = "",
                 llm_fn: Optional[Callable[[str], str]] = None):
        """
        lm          : BaseLM used by the detector (scoring model - can be small)
        llm_fn      : the protected application's generation function. Optional:
                      leave it None to evaluate detection only.
        system_prompt: used by L4 for leak detection (should contain the canary)
        """
        self.cfg = config or PipelineConfig()
        self.sanitizer = Sanitizer(self.cfg.sanitizer)
        self.detector = LossShiftDetector(lm, self.cfg.detector)
        self.gate = DecisionGate(self.cfg.gate)
        self.validator = OutputValidator(self.cfg.validator, system_prompt=system_prompt)
        self.system_prompt = system_prompt
        self.llm_fn = llm_fn

    def analyse(self, prompt: str) -> PipelineResult:
        """Layers 1-3 only. This is what the detection metrics are computed on."""
        t: Dict[str, float] = {}

        t0 = time.perf_counter()
        san = self.sanitizer.run(prompt)
        t["l1_sanitize"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        det = self.detector.score(san.text)
        t["l2_detect"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        gate = self.gate.decide(det.score, san.risk, san.families)
        t["l3_gate"] = (time.perf_counter() - t0) * 1000

        t["total"] = sum(t.values())
        return PipelineResult(prompt=prompt, verdict=gate.verdict.value,
                              sanitize=san, detect=det, gate=gate, timings_ms=t)

    def run(self, prompt: str) -> PipelineResult:
        """Full path including generation and Layer 4."""
        res = self.analyse(prompt)

        if res.gate.verdict is Verdict.BLOCK:
            res.response = BLOCK_MESSAGE
            return res

        if self.llm_fn is None:
            return res

        t0 = time.perf_counter()
        raw = self.llm_fn(res.sanitize.text)
        res.timings_ms["llm"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        val = self.validator.run(raw, system_prompt=self.system_prompt)
        res.timings_ms["l4_validate"] = (time.perf_counter() - t0) * 1000
        res.timings_ms["total"] = sum(v for k, v in res.timings_ms.items() if k != "total")

        res.validate = val
        res.response = val.redacted_output if not val.safe else raw
        if not val.safe:
            res.verdict = Verdict.REVIEW.value if res.verdict == Verdict.ALLOW.value else res.verdict
        return res

    def analyse_many(self, prompts: List[str], progress: bool = True) -> List[PipelineResult]:
        out = []
        n = len(prompts)
        for i, p in enumerate(prompts, 1):
            out.append(self.analyse(p))
            if progress and (i % 10 == 0 or i == n):
                print(f"  scored {i}/{n}", flush=True)
        return out
