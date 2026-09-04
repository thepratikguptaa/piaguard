"""Baselines the proposed pipeline is compared against.

Without these the Results section is a single column of numbers with nothing to
argue against. Each baseline stands in for a real class of deployed defence:

  KeywordFilter      the rule-based filter (NeMo Guardrails-style blocklists,
                     regex WAF rules). Cheap, and the class of defence HOUYI is
                     shown to bypass. Implemented here as Layer 1's rules used
                     as a hard gate - so the ablation "L1 alone" is exactly this.
  PerplexityFilter   flag a prompt if its mean NLL is unusually high. The
                     standard training-free detector before loss-shift methods.
                     Fails on fluent injections, which read perfectly normally.
  RandomBaseline     sanity floor. If a method does not beat this, nothing else
                     in the table means anything.
"""
from typing import List, Sequence
import numpy as np

from .sanitizer import Sanitizer


class KeywordFilter:
    """Rule-based blocklist used as a hard gate (score = sanitizer risk)."""

    name = "Keyword/rule filter"

    def __init__(self, config=None):
        self.san = Sanitizer(config)

    def score(self, text: str) -> float:
        return self.san.run(text).risk

    def score_many(self, texts: Sequence[str]) -> List[float]:
        return [self.score(t) for t in texts]


class PerplexityFilter:
    """Mean NLL of the prompt under the scoring model, no masking."""

    name = "Perplexity threshold"

    def __init__(self, lm, max_tokens: int = 128):
        self.lm = lm
        self.max_tokens = max_tokens

    def score(self, text: str) -> float:
        ids = self.lm.tokenize(text)[: self.max_tokens]
        if len(ids) < 2:
            return 0.0
        return float(self.lm.mean_nll([ids])[0])

    def score_many(self, texts: Sequence[str]) -> List[float]:
        return [self.score(t) for t in texts]


class RandomBaseline:
    name = "Random"

    def __init__(self, seed: int = 13):
        self.rng = np.random.default_rng(seed)

    def score_many(self, texts: Sequence[str]) -> List[float]:
        return list(self.rng.random(len(texts)))
