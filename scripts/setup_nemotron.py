"""
Setup script for Nemotron-Ops-Commander.

Downloads the model, verifies GPU, builds the RAG index, and validates
everything is ready for demo.

Usage:
    python scripts/setup_nemotron.py              # Full setup
    python scripts/setup_nemotron.py --cpu         # CPU-only mode
    python scripts/setup_nemotron.py --skip-model   # Skip model download
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _green(t: str) -> str: return f"\033[92m{t}\033[0m"
def _red(t: str) -> str: return f"\033[91m{t}\033[0m"
def _yellow(t: str) -> str: return f"\033[93m{t}\033[0m"
def _bold(t: str) -> str: return f"\033[1m{t}\033[0m"
def _dim(t: str) -> str: return f"\033[2m{t}\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = _green("✓") if ok else _red("✗")
    msg = f"  {icon} {label}"
    if detail:
        msg += f" {_dim(detail)}"
    print(msg)
    return ok


def step_check_python() -> bool:
    v = sys.version_info
    return check("Python", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}")


def step_check_gpu() -> bool:
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return check("NVIDIA GPU", True, f"{name} ({mem:.1f} GB)")
        else:
            return check("NVIDIA GPU", False, "Not found — will use CPU (slower)")
    except ImportError:
        return check("NVIDIA GPU", False, "torch not installed")


def step_check_deps() -> bool:
    missing = []
    for pkg in ["fastapi", "uvicorn", "pydantic", "structlog", "gradio",
                 "httpx", "chromadb", "sentence_transformers", "yaml"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        return check("Dependencies", False, f"Missing: {', '.join(missing)}")
    return check("Dependencies", True, "All core packages found")


def step_download_model(cpu_only: bool = False) -> bool:
    model_name = "nvidia/Nemotron-Mini-4B-Instruct"
    print(f"\n  Downloading {_bold(model_name)}...")
    print(f"  {_dim('This may take 5-15 minutes on first run.')}")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_name)
        return check("Model download", True, f"Cached at {path}")
    except ImportError:
        print(f"  {_yellow('Install huggingface_hub:')} pip install huggingface_hub")
        # Fallback: just verify transformers can find it
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(model_name)
            return check("Model download", True, "Available via transformers")
        except Exception as e:
            return check("Model download", False, str(e))
    except Exception as e:
        return check("Model download", False, str(e))


def step_init_model(cpu_only: bool = False) -> bool:
    print(f"\n  Initializing Nemotron {'(CPU)' if cpu_only else '(GPU)'}...")
    try:
        if cpu_only:
            os.environ["NEMOTRON_DEVICE"] = "cpu"
            os.environ["NEMOTRON_USE_SGLANG"] = "false"

        from models.nemotron_client import NemotronClient, NemotronConfig

        config = NemotronConfig(
            device="cpu" if cpu_only else "cuda",
            use_sglang=not cpu_only,
        )
        client = NemotronClient(config)
        return check("Model init", True, f"{config.model_name} on {config.device}")
    except Exception as e:
        return check("Model init", False, str(e))


def step_build_rag_index() -> bool:
    print(f"\n  Building RAG index from 30 incidents...")
    try:
        from rag.indexer import IncidentIndexer
        data_dir = Path(__file__).resolve().parent.parent / "data" / "sample_incidents"
        indexer = IncidentIndexer(data_dir)
        count = indexer.run(reset=True)
        return check("RAG index", True, f"{count} incidents indexed into ChromaDB")
    except Exception as e:
        return check("RAG index", False, str(e))


def step_check_env() -> bool:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        return check(".env file", True, "Found")
    else:
        example = Path(__file__).resolve().parent.parent / ".env.example"
        if example.exists():
            import shutil
            shutil.copy(example, env_file)
            return check(".env file", True, "Created from .env.example")
        return check(".env file", False, "Missing — copy .env.example to .env")


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup Nemotron-Ops-Commander")
    parser.add_argument("--cpu", action="store_true", help="CPU-only mode (no GPU required)")
    parser.add_argument("--skip-model", action="store_true", help="Skip model download/init")
    parser.add_argument("--skip-rag", action="store_true", help="Skip RAG index build")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  {_bold('Nemotron-Ops-Commander Setup')}")
    print(f"{'=' * 60}\n")

    results = []

    # Checks
    print(_bold("Pre-flight checks:"))
    results.append(step_check_python())
    results.append(step_check_gpu())
    results.append(step_check_deps())
    results.append(step_check_env())

    # Model
    if not args.skip_model:
        print(f"\n{_bold('Model setup:')}")
        results.append(step_download_model(cpu_only=args.cpu))
        results.append(step_init_model(cpu_only=args.cpu))
    else:
        print(f"\n  {_dim('Skipping model setup')}")

    # RAG
    if not args.skip_rag:
        print(f"\n{_bold('Knowledge base:')}")
        results.append(step_build_rag_index())
    else:
        print(f"\n  {_dim('Skipping RAG index')}")

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    if all(results):
        print(f"  {_green('✓ All checks passed!')} ({passed}/{total})")
        print(f"\n  {_bold('Ready to run:')}")
        print(f"    1. Start server:  uvicorn api.main:app --port 8000")
        print(f"    2. Run demo:      python scripts/demo.py")
        print(f"    3. Open UI:       python ui/gradio_app.py")
    else:
        print(f"  {_yellow(f'{passed}/{total} checks passed')}")
        print(f"\n  Fix the issues above, then re-run this script.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
