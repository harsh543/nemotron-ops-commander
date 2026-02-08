"""Build RAG index from sample incidents."""

from __future__ import annotations

from pathlib import Path

from rag.indexer import IncidentIndexer


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data" / "sample_incidents"
    indexer = IncidentIndexer(data_dir)
    indexer.run()


if __name__ == "__main__":
    main()
