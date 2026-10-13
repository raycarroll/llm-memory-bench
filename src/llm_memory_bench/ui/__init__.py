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
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    streamlit_main()


__all__ = ["main"]
