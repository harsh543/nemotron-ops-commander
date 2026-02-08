"""Gradio demo UI for Nemotron Ops Commander."""

from __future__ import annotations

import json
from typing import Dict

import gradio as gr
import httpx

API_URL = "http://localhost:8000/analyze/"
API_KEY = "change-me"


def analyze_logs(log_text: str) -> Dict:
    """Call the log analysis endpoint and return results."""

    payload = {
        "logs": [
            {
                "timestamp": "now",
                "source": "demo",
                "message": line,
                "labels": {},
            }
            for line in log_text.splitlines()
            if line.strip()
        ],
        "system": "demo",
        "environment": "local",
    }

    headers = {"X-API-Key": API_KEY}
    with httpx.Client(timeout=30) as client:
        response = client.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


def build_ui() -> gr.Blocks:
    """Build the Gradio UI."""

    with gr.Blocks(title="Nemotron Ops Commander") as demo:
        gr.Markdown("# Nemotron Ops Commander\nAI-powered incident response assistant.")

        with gr.Row():
            log_input = gr.Textbox(lines=12, label="Paste logs")
            output = gr.JSON(label="Analysis Result")

        analyze_btn = gr.Button("Analyze")
        analyze_btn.click(fn=analyze_logs, inputs=[log_input], outputs=[output])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
