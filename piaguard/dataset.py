"""Loading and splitting the evaluation data.

Expected CSV schema (any extra columns are ignored):

    id, label, family, source, text
      label  1 = injection / attack, 0 = legitimate
      family taxonomy class: clean | direct | indirect | adversarial |
             backdoor | jailbreak | obfuscated

`--extra-csv` on the evaluate script accepts any file with at least `text` and
`label`; missing `family` is filled with "unlabelled". That is the hook for a
public benchmark - keep the seed set for per-family coverage and report the
benchmark numbers as the headline.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import csv
import os
import numpy as np

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "eval_set.csv")


@dataclass
class Example:
    text: str
    label: int
    family: str = "unlabelled"
    id: str = ""
    source: str = ""


def load_csv(path: str) -> List[Example]:
    out: List[Example] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        if "text" not in cols or "label" not in cols:
            raise ValueError(f"{path}: needs at least 'text' and 'label' columns, got {reader.fieldnames}")
        for i, row in enumerate(reader, 1):
            text = (row[cols["text"]] or "").strip()
            if not text:
                continue
            raw = str(row[cols["label"]]).strip().lower()
            label = 1 if raw in {"1", "true", "yes", "injection", "attack", "malicious"} else 0
            fam = (row.get(cols.get("family", ""), "") or "").strip() or ("clean" if label == 0 else "unlabelled")
            out.append(Example(text=text, label=label, family=fam,
                               id=(row.get(cols.get("id", ""), "") or f"{os.path.basename(path)}_{i}"),
                               source=(row.get(cols.get("source", ""), "") or os.path.basename(path))))
    return out


def load_eval_set(path: Optional[str] = None, extra: Optional[List[str]] = None) -> List[Example]:
    data = load_csv(path or DEFAULT_PATH)
    for p in (extra or []):
        data += load_csv(p)
    return data


def calibration_split(data: List[Example], frac: float = 0.4, seed: int = 13
                      ) -> Tuple[List[Example], List[Example]]:
    """Hold out a CLEAN-ONLY calibration set; everything else is the test set.

    Calibrating theta on clean prompts only, and on prompts never scored at test
    time, is what stops the threshold from being fitted to the attacks it is
    then evaluated on. State this in the report - a panel will ask.
    """
    rng = np.random.default_rng(seed)
    clean = [e for e in data if e.label == 0]
    attacks = [e for e in data if e.label == 1]
    idx = rng.permutation(len(clean))
    n_cal = max(1, int(round(frac * len(clean))))
    cal = [clean[i] for i in idx[:n_cal]]
    test = [clean[i] for i in idx[n_cal:]] + attacks
    return cal, test


def summarise(data: List[Example]) -> str:
    from collections import Counter
    c = Counter(e.family for e in data)
    n_pos = sum(e.label for e in data)
    lines = [f"{len(data)} prompts  ({n_pos} attacks / {len(data) - n_pos} legitimate)"]
    lines += [f"    {k:12s} {v}" for k, v in sorted(c.items())]
    return "\n".join(lines)
