"""Central configuration for the PIAGuard pipeline.

Everything tunable lives here so the evaluation scripts can sweep parameters
without touching the layer implementations.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple
import json


@dataclass
class SanitizerConfig:
    normalize_unicode: bool = True
    strip_invisible: bool = True
    fold_homoglyphs: bool = True
    collapse_whitespace: bool = True
    decode_obfuscation: bool = True   # inspect base64 / hex / rot13 blobs
    # "flag" keeps the text intact and only reports hits (recommended: lower FPR).
    # "redact" replaces matched spans with [REDACTED] before the text reaches the LLM.
    mode: str = "flag"
    max_len: int = 8000


@dataclass
class DetectorConfig:
    model_name: str = "gpt2"
    window_sizes: Tuple[int, ...] = (1, 3)   # single-token and 3-token span masking
    mask_strategy: str = "delete"            # "delete" | "replace"
    aggregation: str = "robust_z"             # "robust_z" | "max" | "topk_mean"
    topk: int = 3
    batch_size: int = 16
    max_tokens: int = 128       # truncate long prompts (cost control)
    max_positions: int = 96     # cap masked variants per prompt
    device: str = "auto"
    fp16: bool = True


@dataclass
class GateConfig:
    # theta is calibrated by scripts/calibrate.py against a clean-only set.
    theta_block: float = 3.5
    theta_review: float = 2.0
    target_fpr: float = 0.01
    # Sanitizer evidence is fused with the detector score, not used alone.
    sanitizer_weight: float = 1.5
    use_three_way: bool = True


@dataclass
class ValidatorConfig:
    canary: str = "SMIT-PBL13-CANARY-7Q4Z"
    system_prompt_ngram: int = 8      # n-gram overlap size for leak detection
    ngram_overlap_threshold: int = 2  # >= this many shared n-grams => leak
    check_secrets: bool = True
    check_compliance_markers: bool = True


@dataclass
class PipelineConfig:
    sanitizer: SanitizerConfig = field(default_factory=SanitizerConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    seed: int = 13

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "PipelineConfig":
        with open(path) as f:
            raw = json.load(f)
        return PipelineConfig(
            sanitizer=SanitizerConfig(**raw["sanitizer"]),
            detector=DetectorConfig(**{**raw["detector"],
                                       "window_sizes": tuple(raw["detector"]["window_sizes"])}),
            gate=GateConfig(**raw["gate"]),
            validator=ValidatorConfig(**raw["validator"]),
            seed=raw.get("seed", 13),
        )
