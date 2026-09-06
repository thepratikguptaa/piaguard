"""Layer 1 - Input Sanitization.

Two jobs, deliberately separated:

1. NORMALISE   - collapse the encoding tricks that let an identical attack string
                 slip past a literal string match (invisible characters, homoglyphs,
                 NFKC-equivalent lookalikes, base64/hex/rot13 wrappers).
2. FLAG        - report which known injection patterns matched, and where.

Design note for the panel: this layer does NOT block on its own, and by default
does not rewrite the prompt either. Rule-based filters are exactly the component
the literature shows is bypassable (HOUYI), so using it as a gate would inherit
that weakness. Here it contributes *evidence* (a risk score) that the Decision
Gate fuses with the behavioural signal from Layer 2. The normalisation half is
what actually matters: it removes the cheap evasions before Layer 2 tokenises.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import base64
import binascii
import codecs
import re
import unicodedata

# Invisible / control characters used to hide instructions inside otherwise clean text.
INVISIBLE = "".join([
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",   # zero-width family
    "\u00ad",                                            # soft hyphen
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",   # bidi overrides
    "\u2066", "\u2067", "\u2068", "\u2069",
])
INVISIBLE_RE = re.compile(f"[{re.escape(INVISIBLE)}]")
TAG_CHARS_RE = re.compile(r"[\U000E0000-\U000E007F]")  # unicode tag block

# Cyrillic / Greek lookalikes folded to ASCII so "іgnore" == "ignore".
HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ј": "j", "һ": "h", "ԁ": "d", "ɑ": "a", "ο": "o",
    "ν": "v", "τ": "t", "ι": "i", "κ": "k", "ρ": "p", "ѐ": "e", "ⅼ": "l",
    "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
}

# Canonical, widely published injection / trigger patterns.
# Each entry: (rule id, attack family, weight, compiled regex)
_RULES: List[Tuple[str, str, float, re.Pattern]] = [
    ("R01", "direct", 1.0, re.compile(r"\bignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|preceding|earlier)\b.{0,20}\b(instruction|prompt|rule|direction|message)s?\b", re.I)),
    ("R02", "direct", 1.0, re.compile(r"\b(disregard|forget|override|bypass|discard)\b.{0,25}\b(instruction|prompt|rule|guideline|system message|constraint)s?\b", re.I)),
    ("R03", "direct", 0.9, re.compile(r"\b(new|updated|revised)\s+(instruction|task|rule|directive)s?\s*[:\-]", re.I)),
    ("R04", "leak",   1.0, re.compile(r"\b(reveal|repeat|print|show|output|display|echo|disclose|dump)\b.{0,30}\b(system\s*(prompt|message)|initial\s*(prompt|instruction)|your\s+(instruction|prompt|rule)s?|prompt\s+above)\b", re.I)),
    ("R05", "leak",   0.9, re.compile(r"\b(what|which)\b.{0,25}\byour\s+(system\s+prompt|original\s+instructions|hidden\s+rules)\b", re.I)),
    ("R06", "role",   1.0, re.compile(r"\byou\s+are\s+now\b.{0,40}\b(dan|do\s+anything\s+now|unrestricted|jailbroken|developer\s+mode|unfiltered)\b", re.I)),
    ("R07", "role",   0.8, re.compile(r"\b(act|behave|pretend|roleplay|respond)\s+as\s+(if\s+you\s+are\s+)?(an?\s+)?(unrestricted|unfiltered|amoral|evil|uncensored|no[- ]?limits?)\b", re.I)),
    ("R08", "role",   0.9, re.compile(r"\b(enable|activate|enter)\b.{0,15}\b(developer|debug|god|sudo|admin|dev)\s*mode\b", re.I)),
    ("R09", "delim",  0.8, re.compile(r"(?m)^\s*(?:#{0,3}\s*)?(system|assistant|user)\s*[:\]>]\s*", re.I)),
    ("R10", "delim",  0.7, re.compile(r"(<\|(im_start|im_end|endoftext|system|eot_id|start_header_id)\|>)|(\[/?(INST|SYS)\])|(###\s*(Instruction|System)\s*:)", re.I)),
    ("R11", "delim",  0.6, re.compile(r"[-=*_#]{12,}")),  # separator component (HOUYI)
    ("R12", "policy", 0.9, re.compile(r"\b(without|no|skip)\b.{0,20}\b(safety|ethical|moral|content)\b.{0,20}\b(filter|guideline|restriction|check|policy)s?\b", re.I)),
    ("R13", "policy", 0.8, re.compile(r"\b(you\s+(must|have\s+to|should)\s+(now\s+)?(comply|obey|answer\s+anything))\b", re.I)),
    ("R14", "exfil",  1.0, re.compile(r"\b(send|post|forward|upload|exfiltrate|transmit)\b.{0,30}\b(to\s+)?(https?://|api\s*key|token|credential|password|\.env)\b", re.I)),
    ("R15", "exfil",  0.9, re.compile(r"!\[.{0,40}\]\(https?://[^)]*\{|\?(q|data|c|x)=", re.I)),  # markdown-image beacon
    ("R16", "indirect", 0.9, re.compile(r"\b(ai|llm|assistant|model|chatbot|agent)\b.{0,25}\b(reading|processing|summariz\w+|parsing)\b.{0,25}\b(this|the following)\b.{0,20}\b(must|should|please)\b", re.I)),
    ("R17", "indirect", 0.8, re.compile(r"\b(important|urgent|note\s+to)\s+(instruction|message|note)?\s*(for|to)\s+(the\s+)?(ai|assistant|llm|model|agent)\b", re.I)),
    ("R18", "obfusc", 0.7, re.compile(r"\b(decode|decrypt|de-?obfuscate|base\s*64|rot\s*13)\b.{0,30}\b(then|and)\b.{0,20}\b(execute|follow|obey|run|do)\b", re.I)),
]

_B64_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
_HEX_RE = re.compile(r"\b(?:[0-9a-fA-F]{2}\s?){12,}\b")
_INJECT_HINT_RE = re.compile(r"ignore|instruction|system prompt|disregard|jailbreak|you are now|reveal", re.I)


@dataclass
class RuleHit:
    rule_id: str
    family: str
    weight: float
    # Offsets into the normalised prompt, or None when the hit came from a
    # decoded payload that has no location there (e.g. a whole-text rot13).
    span: Optional[Tuple[int, int]]
    snippet: str


@dataclass
class SanitizeResult:
    text: str                       # normalised (and optionally redacted) prompt
    original: str
    hits: List[RuleHit] = field(default_factory=list)
    transforms: List[str] = field(default_factory=list)
    decoded_payloads: List[str] = field(default_factory=list)
    risk_divisor: float = 3.0

    @property
    def risk(self) -> float:
        """Saturating risk score in [0, 1] from matched rule weights."""
        if not self.hits:
            return 0.0
        total = sum(h.weight for h in self.hits)
        families = len({h.family for h in self.hits})
        return min(1.0, (total + 0.5 * (families - 1)) / self.risk_divisor)

    @property
    def families(self) -> List[str]:
        return sorted({h.family for h in self.hits})

    def to_dict(self) -> Dict:
        return {
            "risk": round(self.risk, 4),
            "n_hits": len(self.hits),
            "rules": [h.rule_id for h in self.hits],
            "families": self.families,
            "transforms": self.transforms,
            "n_decoded": len(self.decoded_payloads),
        }


class Sanitizer:
    """Layer 1. Normalise encoding tricks, then flag known injection patterns."""

    def __init__(self, config=None):
        from .config import SanitizerConfig
        self.cfg = config or SanitizerConfig()

    # ---------- normalisation ----------
    def _normalize(self, text: str) -> Tuple[str, List[str]]:
        applied: List[str] = []
        out = text[: self.cfg.max_len]

        if self.cfg.strip_invisible:
            new = TAG_CHARS_RE.sub("", INVISIBLE_RE.sub("", out))
            if new != out:
                applied.append("strip_invisible")
            out = new

        if self.cfg.normalize_unicode:
            new = unicodedata.normalize("NFKC", out)
            if new != out:
                applied.append("nfkc")
            out = new

        if self.cfg.fold_homoglyphs:
            new = "".join(HOMOGLYPHS.get(ch, ch) for ch in out)
            if new != out:
                applied.append("fold_homoglyphs")
            out = new

        if self.cfg.collapse_whitespace:
            new = re.sub(r"[ \t\u00a0]{2,}", " ", re.sub(r"\n{3,}", "\n\n", out)).strip()
            if new != out:
                applied.append("collapse_whitespace")
            out = new

        return out, applied

    # ---------- obfuscation ----------
    def _decode_payloads(self, text: str) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
        """Decode base64/hex/rot13 blobs and keep any that look like instructions.

        The decoded text is never executed or forwarded - it is only inspected so
        an obfuscated payload raises the same flags as a plaintext one.

        Returns (decoded_text, blob_span) pairs, where blob_span locates the
        ENCODED blob inside `text`. Redaction needs that location: offsets found
        inside the decoded string do not address the same characters, so using
        them would redact unrelated text (and leave the payload itself intact).
        rot13 rewrites the whole prompt and has no single blob, so its span is
        None and it contributes evidence only.
        """
        found: List[Tuple[str, Optional[Tuple[int, int]]]] = []
        for m in _B64_RE.finditer(text):
            blob = m.group(0)
            try:
                dec = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False)
                s = dec.decode("utf-8", errors="strict")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                continue
            if _INJECT_HINT_RE.search(s):
                found.append((s, m.span()))
        for m in _HEX_RE.finditer(text):
            try:
                s = bytes.fromhex(re.sub(r"\s", "", m.group(0))).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                continue
            if _INJECT_HINT_RE.search(s):
                found.append((s, m.span()))
        try:
            rot = codecs.encode(text, "rot13")
            if _INJECT_HINT_RE.search(rot) and not _INJECT_HINT_RE.search(text):
                found.append((rot, None))
        except Exception:
            pass
        return found

    # ---------- rule matching ----------
    @staticmethod
    def _match(text: str) -> List[RuleHit]:
        hits: List[RuleHit] = []
        for rid, family, weight, rx in _RULES:
            m = rx.search(text)
            if m:
                hits.append(RuleHit(rid, family, weight, m.span(),
                                    text[m.start():min(m.end(), m.start() + 80)]))
        return hits

    @staticmethod
    def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping/adjacent spans so redaction cannot interleave them."""
        merged: List[Tuple[int, int]] = []
        for s, e in sorted(spans):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def run(self, text: str) -> SanitizeResult:
        norm, applied = self._normalize(text)
        decoded = self._decode_payloads(norm) if self.cfg.decode_obfuscation else []

        hits = self._match(norm)
        for payload, blob_span in decoded:            # flags from hidden payloads too
            for h in self._match(payload):
                h.rule_id += "*"                      # '*' = matched after decoding
                # h.span indexes the DECODED text, which is not a slice of norm.
                # Re-anchor it to the encoded blob so redaction removes the
                # payload rather than whatever happens to sit at those offsets.
                h.span = blob_span
                hits.append(h)

        out = norm
        if self.cfg.mode == "redact":
            spans = [h.span for h in hits
                     if h.span is not None and 0 <= h.span[0] < h.span[1] <= len(norm)]
            if spans:
                for s, e in reversed(self._merge_spans(spans)):
                    out = out[:s] + "[REDACTED]" + out[e:]
                applied.append("redact")

        # de-duplicate rule ids, keep first occurrence
        seen, uniq = set(), []
        for h in hits:
            if h.rule_id not in seen:
                seen.add(h.rule_id)
                uniq.append(h)

        return SanitizeResult(text=out, original=text, hits=uniq,
                              transforms=applied,
                              decoded_payloads=[p for p, _ in decoded],
                              risk_divisor=self.cfg.risk_divisor)
