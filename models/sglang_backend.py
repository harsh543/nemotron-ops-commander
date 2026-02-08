"""SGLang backend helpers."""

from __future__ import annotations

import structlog
import sglang as sgl

logger = structlog.get_logger(__name__)


def init_sglang_runtime(model_path: str, device: str):
    """Initialize SGLang runtime and set as default backend."""

    runtime = sgl.Runtime(model_path=model_path, device=device)
    sgl.set_default_backend(runtime)
    logger.info("sglang.runtime.init", model_path=model_path, device=device)
    return runtime
