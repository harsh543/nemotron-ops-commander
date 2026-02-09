"""
LLM inference — local GPU preferred, HF Inference API as fallback.

When running on a GPU Space (T4/A10), loads the model directly onto the GPU
for fast inference (~200-500ms).  Falls back to the remote HF Inference API
only when local GPU loading is impossible (no GPU, no torch, gated model
without token).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = os.environ.get("MODEL_ID", "nvidia/Nemotron-Mini-4B-Instruct")

# Models to try loading *locally* on the Space GPU (in priority order).
# Nemotron-Mini-4B needs ~8 GB fp16 — fits easily on a T4 (16 GB).
LOCAL_CANDIDATES = [
    MODEL_ID,
    "microsoft/Phi-3-mini-4k-instruct",   # 3.8B, open, fast
]

# Models to try via the *remote* Inference API (fallback).
REMOTE_CANDIDATES = [
    "nvidia/Nemotron-Mini-4B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3-mini-4k-instruct",
]

# Short timeout for remote API calls so the fallback chain doesn't stall.
_API_TIMEOUT = 30


class LLMClient:
    """Unified LLM client — local GPU or remote API."""

    def __init__(self) -> None:
        self.backend: str = "none"          # "local" | "api"
        self.active_model: str = MODEL_ID
        # local-mode state
        self._model = None
        self._tokenizer = None
        self._device: Optional[str] = None
        # api-mode state
        self._api_client = None

        self._initialize()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        if self._try_local_load():
            return
        self._setup_api_fallback()

    def _try_local_load(self) -> bool:
        """Attempt to load a model on the local GPU."""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.info("No CUDA GPU detected — skipping local load")
                return False
        except ImportError:
            logger.info("PyTorch not installed — skipping local load")
            return False

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        for model_id in LOCAL_CANDIDATES:
            try:
                logger.info("Loading %s locally on GPU ...", model_id)
                t0 = time.time()

                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_id, token=HF_TOKEN, trust_remote_code=True,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    token=HF_TOKEN,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self._device = "cuda"
                self.backend = "local"
                self.active_model = model_id

                elapsed = time.time() - t0
                logger.info(
                    "Loaded %s on GPU in %.1fs  (VRAM: %.1f GB)",
                    model_id, elapsed,
                    torch.cuda.memory_allocated() / 1e9,
                )
                return True

            except Exception as exc:
                logger.warning("Failed to load %s locally: %s", model_id, exc)
                # Free any partially-loaded state
                self._model = None
                self._tokenizer = None
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

        return False

    def _setup_api_fallback(self) -> None:
        from huggingface_hub import InferenceClient

        self._api_client = InferenceClient(token=HF_TOKEN, timeout=_API_TIMEOUT)
        self.backend = "api"
        self.active_model = REMOTE_CANDIDATES[0]
        logger.info("Using remote HF Inference API (fallback)")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Tuple[str, float]:
        """Generate text.  Returns *(text, latency_ms)*."""

        if self.backend == "local":
            return self._generate_local(prompt, system_prompt, max_tokens, temperature)
        return self._generate_api(prompt, system_prompt, max_tokens, temperature)

    # ── local GPU path ────────────────────────────────────────────────

    def _generate_local(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float,
    ) -> Tuple[str, float]:
        import torch

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Use the tokenizer's built-in chat template when available,
        # otherwise fall back to a simple concatenation.
        try:
            text_input = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text_input = (
                f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            )

        inputs = self._tokenizer(
            text_input, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self._device)

        start = time.time()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9,
                do_sample=temperature > 0,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        latency_ms = (time.time() - start) * 1000
        return result.strip(), latency_ms

    # ── remote API path ───────────────────────────────────────────────

    def _generate_api(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float,
    ) -> Tuple[str, float]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        last_error: Optional[Exception] = None

        for model_id in REMOTE_CANDIDATES:
            try:
                response = self._api_client.chat_completion(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = response.choices[0].message.content
                latency_ms = (time.time() - start) * 1000
                self.active_model = model_id
                return text.strip(), latency_ms
            except Exception as exc:
                last_error = exc
                logger.warning("Remote model %s failed: %s", model_id, exc)
                continue

        raise RuntimeError(
            f"All remote models failed. Last error: {last_error}\n"
            "Set HF_TOKEN as a Space secret for higher rate limits."
        )

    # ------------------------------------------------------------------

    def get_active_model(self) -> str:
        tag = "local GPU" if self.backend == "local" else "remote API"
        return f"{self.active_model} ({tag})"


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
