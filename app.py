"""Điểm vào cho Hugging Face Spaces và chạy GUI cục bộ."""

import os

from viegrader.gui import build_gui
from viegrader.gui.app import CSS

demo = build_gui()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True, css=CSS,
    )
