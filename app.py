"""Streamlit Application Entrypoint.

Delegates execution to src/streamlit_app.py for standard deployments.
"""

from pathlib import Path
import runpy
import sys

# Ensure src is on Python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

app_path = src_dir / "streamlit_app.py"
runpy.run_path(str(app_path), run_name="__main__")

