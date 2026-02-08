"""Download and warm up Nemotron model."""

from __future__ import annotations

from models.nemotron_client import NemotronClient, NemotronConfig


def main() -> None:
    client = NemotronClient(NemotronConfig())
    print("Model initialized", client.config.model_name)


if __name__ == "__main__":
    main()
