"""Language-model backends for the Layer 2 detector.

The detector only needs three things from a model, so both backends implement
the same tiny interface:

    tokenize(text)          -> List[int]
    decode(ids)             -> str
    mean_nll(list_of_ids)   -> np.ndarray   (one mean negative log-likelihood per sequence)

HFCausalLM   real backend (GPT-2 / Mistral / LLaMA via transformers)
MockCausalLM offline, deterministic, no torch. For unit tests and CI only -
             numbers produced with it are NOT experimental results.
"""
from typing import List, Sequence
import hashlib
import numpy as np


class BaseLM:
    name = "base"

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def mean_nll(self, sequences: List[List[int]]) -> np.ndarray:
        raise NotImplementedError


class HFCausalLM(BaseLM):
    """Causal LM wrapper. Computes per-sequence mean NLL in batches.

    Works with any decoder-only checkpoint: gpt2, gpt2-medium,
    mistralai/Mistral-7B-Instruct-v0.2, meta-llama/Llama-3.2-1B, ...
    """

    def __init__(self, model_name: str = "gpt2", device: str = "auto", fp16: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.name = model_name

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(device).eval()
        self.pad_id = self.tok.pad_token_id

    def tokenize(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids))

    def mean_nll(self, sequences: List[List[int]]) -> np.ndarray:
        """Mean token-level cross-entropy for each sequence (length-normalised)."""
        torch = self.torch
        if not sequences:
            return np.zeros(0, dtype=np.float32)

        maxlen = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), maxlen), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(sequences), maxlen), dtype=torch.long)
        for i, s in enumerate(sequences):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        ids, attn = ids.to(self.device), attn.to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=attn).logits.float()

        # next-token prediction: logits[:, :-1] predict ids[:, 1:]
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = ids[:, 1:].unsqueeze(-1)
        tok_nll = -logprobs.gather(-1, target).squeeze(-1)          # (B, L-1)
        mask = attn[:, 1:].float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (tok_nll * mask).sum(dim=1).div(denom).cpu().numpy().astype(np.float32)


class MockCausalLM(BaseLM):
    """Deterministic stand-in used when torch/weights are unavailable.

    Assigns every whitespace token a fixed pseudo-random surprisal, with a bump
    for tokens that appear in known trigger phrases, so the masking machinery,
    calibration and metrics code can be exercised end to end offline.
    """

    name = "mock"
    TRIGGERS = {
        "ignore", "disregard", "override", "bypass", "forget", "instructions",
        "instruction", "system", "prompt", "reveal", "dan", "jailbroken",
        "unrestricted", "developer", "sudo", "exfiltrate", "credentials",
    }

    def __init__(self, seed: int = 13):
        self.seed = seed
        self._vocab: dict = {}
        self._inv: dict = {}

    def _id(self, token: str) -> int:
        if token not in self._vocab:
            tid = len(self._vocab) + 1
            self._vocab[token] = tid
            self._inv[tid] = token
        return self._vocab[token]

    def tokenize(self, text: str) -> List[int]:
        return [self._id(t) for t in text.split()]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._inv.get(i, "<unk>") for i in ids)

    def _surprisal(self, tid: int) -> float:
        token = self._inv.get(tid, "")
        h = hashlib.md5(f"{self.seed}:{token.lower()}".encode()).digest()
        base = 2.0 + (h[0] / 255.0) * 2.0
        if token.strip(".,:;!?\"'").lower() in self.TRIGGERS:
            base += 6.0
        return base

    def mean_nll(self, sequences: List[List[int]]) -> np.ndarray:
        out = []
        for s in sequences:
            out.append(float(np.mean([self._surprisal(t) for t in s])) if s else 0.0)
        return np.asarray(out, dtype=np.float32)


def load_lm(cfg) -> BaseLM:
    """Load the configured HF model, falling back to the mock if unavailable."""
    try:
        return HFCausalLM(cfg.model_name, device=cfg.device, fp16=cfg.fp16)
    except Exception as exc:  # torch/transformers missing, or no network for weights
        print(f"[models] HF backend unavailable ({type(exc).__name__}: {exc}).")
        print("[models] Falling back to MockCausalLM - plumbing only, NOT valid results.")
        return MockCausalLM()
