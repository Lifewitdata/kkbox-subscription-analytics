"""Execute the actual Streamlit entry point using Streamlit's headless AppTest."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_renders_without_uncaught_exceptions():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception, [exception.message for exception in app.exception]
    assert any("KKBox Subscription Analytics" in title.value for title in app.title)
    assert len(app.metric) >= 2
