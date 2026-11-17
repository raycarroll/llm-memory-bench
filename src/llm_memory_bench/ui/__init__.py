"""Streamlit UI for defining and inspecting benchmark runs."""

from pathlib import Path
import sys


def main() -> None:
    """Launch the app through Streamlit's server entry point.

    Calling ``render_app`` directly puts Streamlit into bare mode, where
    session state and ScriptRunContext are unavailable.
    """
    from streamlit.web.cli import main as streamlit_main

    app_path = Path(__file__).with_name("app.py")
    # The transformers package exposes many optional vision modules through
    # lazy imports. Streamlit's source watcher probes those modules and logs
    # misleading missing-torchvision tracebacks, although this app only uses
    # text embeddings. The UI is restarted explicitly after code changes.
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.fileWatcherType=none",
        *sys.argv[1:],
    ]
    streamlit_main()


__all__ = ["main"]
