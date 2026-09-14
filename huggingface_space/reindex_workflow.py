"""
reindex_workflow.py -- Render Workflows track.

Re-indexes the RAG knowledge base in batches, so a failure partway
through a large archive resumes cleanly instead of restarting from
scratch. Unlike bolting a workflow onto a fast synchronous request, this
is a genuine fit: re-indexing is real background work, and losing
partial progress on a batch failure is a real cost worth avoiding.

index_batch_step is rigged to fail on its first attempt for one specific
batch. Because Render Workflows replays a retried parent's already-
completed ctx.run() calls from history instead of re-executing them,
retrying reindex_knowledge_base re-runs only that one batch -- earlier
batches are not re-embedded or re-upserted.
"""

from __future__ import annotations

import json
from pathlib import Path

from render import Retry, TaskContext, Workflows

app = Workflows()

DATA_DIR = Path(__file__).parent / "data" / "sample_incidents"
BATCH_SIZE = 5

# Keyed by "{run_key}:{batch_index}" so concurrent runs don't collide.
# Reliable within a run because Render Workflows keeps one live process
# per app.start() -- a short-backoff retry stays on this process.
_batch_attempts: dict[str, int] = {}


def _load_incidents() -> list[dict]:
    incidents = []
    for path in sorted(DATA_DIR.glob("*.json")):
        try:
            incidents.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return incidents


@app.task(name="index_batch_step")
async def index_batch_step(ctx: TaskContext, run_key: str, batch_index: int, incidents: list[dict]) -> int:
    # Nebius-hosted embeddings, not a local model -- this Workflow service
    # has the same 512MB RAM ceiling as the web service (see
    # rag_engine_render.py), which a local embedding model already OOM'd.
    from rag_engine_render import _embed, get_rag_engine

    key = f"{run_key}:{batch_index}"
    attempt = _batch_attempts.get(key, 0) + 1
    _batch_attempts[key] = attempt
    if batch_index == 1 and attempt == 1:
        raise RuntimeError(f"Simulated transient failure indexing batch {batch_index} -- demonstrates resume")

    engine = get_rag_engine()
    ids, documents, metadatas = [], [], []
    for inc in incidents:
        doc_text = f"{inc.get('summary', '')} {inc.get('root_cause', '')} {inc.get('resolution', '')}"
        ids.append(str(inc["id"]))
        documents.append(doc_text)
        metadatas.append(
            {
                "title": inc.get("summary", inc.get("title", "")),
                "source": inc.get("source", "internal"),
                "severity": inc.get("severity", "unknown"),
                "resolution": inc.get("resolution", ""),
                "service": inc.get("service", ""),
                "tags": ",".join(inc.get("tags", [])),
            }
        )
    embeddings = _embed(documents)
    engine.collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(incidents)


@app.task(
    name="reindex_knowledge_base",
    retry=Retry(max_retries=3, wait_duration_ms=1000, backoff_scaling=2),
)
async def reindex_knowledge_base(ctx: TaskContext, run_key: str) -> dict:
    incidents = _load_incidents()
    batches = [incidents[i : i + BATCH_SIZE] for i in range(0, len(incidents), BATCH_SIZE)]

    total = 0
    for i, batch in enumerate(batches):
        count = await ctx.run(index_batch_step, run_key, i, batch)
        total += count

    return {"total_indexed": total, "batches": len(batches)}


app.start()
