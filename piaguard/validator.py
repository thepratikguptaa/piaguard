"""Layer 4 - Output Validation.

Layers 1-3 can be wrong. This layer is the last check, and it looks at the
*response*, where a successful injection has to show itself eventually:

  canary        an unguessable string planted in the system prompt. If it ever
                appears in an output, the system prompt has leaked - a
                zero-false-positive signal, and the cheapest one available.
  ngram overlap catches paraphrase-free regurgitation of the system prompt even
                without the canary.
  secrets       API keys, tokens, private keys, emails, card-like digit runs.
  compliance    phrases a model emits when it has accepted a persona override
                ("DAN mode enabled", "as an unrestricted AI"), scored only when
                the reply is not a refusal - refusals quote these phrases while
                declining, which is exactly how naive filters generate false
                positives.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re

SECRET_PATTERNS = [
    ("openai_key",   re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("aws_key",      re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("bearer",       re.compile(r"\bBearer\s+[A-Za-z0-9\-_.]{20,}\b")),
    ("private_key",  re.compile(r"-----BEGIN\s+[A-Z ]*PRIVATE KEY-----")),
    ("jwt",          re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("email",        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("card_like",    re.compile(r"\b(?:\d[ -]?){13,16}\b")),
]

COMPLIANCE_MARKERS = [
    re.compile(r"\b(dan|developer)\s*mode\s*(is\s*)?(enabled|activated|on)\b", re.I),
    re.compile(r"\bas\s+an?\s+(unrestricted|unfiltered|amoral|jailbroken)\b", re.I),
    re.compile(r"\bi\s+(will\s+)?(now\s+)?ignore\s+(my\s+)?(previous\s+)?(instructions|guidelines|rules)\b", re.I),
    re.compile(r"\b(my|the)\s+system\s+prompt\s+(is|says|reads)\b", re.I),
    re.compile(r"\bhere\s+(is|are)\s+my\s+(instructions|system\s+prompt|rules)\b", re.I),
]

REFUSAL_MARKERS = [
    re.compile(r"\bi\s+(can'?t|cannot|won'?t|am\s+not\s+able\s+to)\b", re.I),
    re.compile(r"\bi'?m\s+not\s+going\s+to\b", re.I),
    re.compile(r"\b(unable|declin\w+|not\s+appropriate)\b", re.I),
]

_WORD_RE = re.compile(r"\w+")


def _ngrams(text: str, n: int) -> set:
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


@dataclass
class ValidateResult:
    safe: bool
    issues: List[str] = field(default_factory=list)
    redacted_output: str = ""
    canary_leak: bool = False
    prompt_leak: bool = False
    secrets: List[str] = field(default_factory=list)
    compliance: bool = False
    is_refusal: bool = False

    def to_dict(self) -> Dict:
        return {
            "output_safe": self.safe,
            "canary_leak": self.canary_leak,
            "prompt_leak": self.prompt_leak,
            "secrets": self.secrets,
            "compliance_marker": self.compliance,
            "issues": self.issues,
        }


class OutputValidator:
    def __init__(self, config=None, system_prompt: str = ""):
        from .config import ValidatorConfig
        self.cfg = config or ValidatorConfig()
        self.system_prompt = system_prompt
        self._sys_ngrams = _ngrams(system_prompt, self.cfg.system_prompt_ngram) if system_prompt else set()

    def run(self, output: str, system_prompt: Optional[str] = None) -> ValidateResult:
        issues: List[str] = []
        redacted = output

        canary_leak = bool(self.cfg.canary) and self.cfg.canary in output
        if canary_leak:
            issues.append("canary_leak")
            redacted = redacted.replace(self.cfg.canary, "[REDACTED-CANARY]")

        sys_ngrams = (_ngrams(system_prompt, self.cfg.system_prompt_ngram)
                      if system_prompt else self._sys_ngrams)
        prompt_leak = False
        if sys_ngrams:
            shared = sys_ngrams & _ngrams(output, self.cfg.system_prompt_ngram)
            if len(shared) >= self.cfg.ngram_overlap_threshold:
                prompt_leak = True
                issues.append(f"system_prompt_leak(ngrams={len(shared)})")

        secrets: List[str] = []
        if self.cfg.check_secrets:
            for label, rx in SECRET_PATTERNS:
                if rx.search(redacted):
                    secrets.append(label)
                    redacted = rx.sub(f"[REDACTED-{label.upper()}]", redacted)
            if secrets:
                issues.append("secret_exposure:" + ",".join(secrets))

        is_refusal = any(rx.search(output[:400]) for rx in REFUSAL_MARKERS)
        compliance = False
        if self.cfg.check_compliance_markers and not is_refusal:
            compliance = any(rx.search(output) for rx in COMPLIANCE_MARKERS)
            if compliance:
                issues.append("persona_override_marker")

        return ValidateResult(
            safe=not issues, issues=issues, redacted_output=redacted,
            canary_leak=canary_leak, prompt_leak=prompt_leak,
            secrets=secrets, compliance=compliance, is_refusal=is_refusal,
        )
