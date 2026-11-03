"""Start the actual Streamlit server and verify its HTTP health endpoint."""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765
HEALTH_URL = f"http://127.0.0.1:{PORT}/_stcore/health"


def main() -> None:
    command = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
        "--server.headless=true",
        f"--server.port={PORT}",
        "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 45
        last_error = None
        while time.time() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(
                    f"Streamlit exited with code {process.returncode}.\n{output[-8000:]}"
                )
            try:
                with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    if response.status == 200 and body.strip().lower() == "ok":
                        print(f"Streamlit server health check passed: {HEALTH_URL}")
                        return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            time.sleep(0.5)
        output = process.stdout.read() if process.stdout else ""
        raise TimeoutError(
            f"Streamlit did not become healthy. Last error: {last_error}\n{output[-8000:]}"
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
