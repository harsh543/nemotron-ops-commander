"""
LLM inference — Nebius Token Factory preferred, local GPU / HF Inference API as fallback.

Nebius Token Factory (https://tokenfactory.nebius.com) is a hosted,
OpenAI-compatible inference API. Unlike this Space's free shared T4 GPU,
it serves requests concurrently instead of queuing them one at a time, so
it is tried first whenever NEBIUS_API_KEY is configured. When it is not
configured, the client falls back to loading a model directly onto the
Space's GPU (~200-500ms/request but single-threaded), and finally to the
remote HF Inference API if no GPU is available.
"""

from __future__ import annotations

import spaces
import logging
import os
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = os.environ.get("MODEL_ID", "nvidia/Nemotron-Mini-4B-Instruct")

# Nebius Token Factory — OpenAI-compatible hosted inference.
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
NEBIUS_BASE_URL = os.environ.get(
    "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
)
# Nano is a small, non-reasoning Nemotron variant: fast and gives a direct
# answer. The catalog also has "Nemotron-3_5-Lightning", a reasoning model
# that emits its chain-of-thought inline in `content` — usable, but needs
# a much larger max_tokens budget and output-stripping, so it's not the
# default for a live demo. Swap via env var to any ID from `GET /v1/models`.
NEBIUS_MODEL = os.environ.get(
    "NEBIUS_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
)
# $/token — the Token Factory API does not return per-model pricing, so
# these are unset by default. Set them from your Nebius billing dashboard
# if you want the benchmark to report a dollar cost instead of just tokens.
NEBIUS_PROMPT_COST_PER_TOKEN = float(os.environ.get("NEBIUS_PROMPT_COST_PER_TOKEN", 0) or 0)
NEBIUS_COMPLETION_COST_PER_TOKEN = float(os.environ.get("NEBIUS_COMPLETION_COST_PER_TOKEN", 0) or 0)

# Models to try loading *locally* on the Space GPU (in priority order).
# Nemotron-Mini-4B-Instruct — 4B, efficient for T4 GPU with excellent instruction-following
LOCAL_CANDIDATES = [
    MODEL_ID,
    "nvidia/Nemotron-Mini-4B-Instruct",        # 4B, efficient instruction-following
    "meta-llama/Meta-Llama-3.1-8B-Instruct",  # 8B, 128k context, strong performance (fallback)
    "microsoft/Phi-3-mini-4k-instruct",        # 3.8B, open, fast
]

# Models to try via the *remote* Inference API (fallback).
# Ensure MODEL_ID is first so the UI matches the Space setting.
REMOTE_CANDIDATES = list(
    dict.fromkeys(
        [
            MODEL_ID,
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "nvidia/Nemotron-Mini-4B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "HuggingFaceH4/zephyr-7b-beta",
            "microsoft/Phi-3-mini-4k-instruct",
        ]
    )
)

# Short timeout for remote API calls so the fallback chain doesn't stall.
_API_TIMEOUT = 30


class LLMClient:
    """Unified LLM client — local GPU or remote API."""

    def __init__(self) -> None:
        self.backend: str = "none"          # "nebius" | "local" | "api"
        self.active_model: str = MODEL_ID
        self.last_usage: Optional[dict] = None  # tokens/cost from the last Nebius call
        # local-mode state
        self._model = None
        self._tokenizer = None
        self._device: Optional[str] = None
        # api-mode state
        self._api_client = None
        # nebius-mode state
        self._nebius_client = None

        self._initialize()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        if self._try_nebius():
            return
        if self._try_local_load():
            return
        self._setup_api_fallback()

    def _try_nebius(self) -> bool:
        """Attempt to configure the Nebius Token Factory backend."""
        if not NEBIUS_API_KEY:
            logger.info("NEBIUS_API_KEY not set — skipping Nebius Token Factory")
            return False

        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed — skipping Nebius Token Factory")
            return False

        self._nebius_client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY)
        self.backend = "nebius"
        self.active_model = NEBIUS_MODEL
        logger.info("Using Nebius Token Factory (%s)", NEBIUS_MODEL)
        return True

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
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    trust_remote_code=True,
                ).eval().to("cuda")
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

        if self.backend == "nebius":
            return self._generate_nebius(prompt, system_prompt, max_tokens, temperature)
        if self.backend == "local":
            return self._generate_local(prompt, system_prompt, max_tokens, temperature)
        return self._generate_api(prompt, system_prompt, max_tokens, temperature)

    # ── Nebius Token Factory path ───────────────────────────────────────

    def _generate_nebius(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float,
    ) -> Tuple[str, float]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        response = self._nebius_client.chat.completions.create(
            model=NEBIUS_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = (time.time() - start) * 1000

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        has_pricing = NEBIUS_PROMPT_COST_PER_TOKEN > 0 or NEBIUS_COMPLETION_COST_PER_TOKEN > 0
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round(
                prompt_tokens * NEBIUS_PROMPT_COST_PER_TOKEN
                + completion_tokens * NEBIUS_COMPLETION_COST_PER_TOKEN,
                8,
            )
            if has_pricing
            else None,
        }

        text = response.choices[0].message.content
        if text is None:
            finish_reason = response.choices[0].finish_reason
            logger.warning(
                "Nebius returned empty content (finish_reason=%s, max_tokens=%d) — "
                "likely truncated before emitting visible output",
                finish_reason, max_tokens,
            )
            text = ""
        return text.strip(), latency_ms

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
        result = self._tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
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
        tag = {"nebius": "Nebius Token Factory", "local": "local GPU"}.get(
            self.backend, "remote API"
        )
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
