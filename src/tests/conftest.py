# src/tests/conftest.py
from __future__ import annotations
import shutil
from pathlib import Path
import pytest

# This autouse fixture runs for every test and overwrites any stubbed files
@pytest.fixture(autouse=True)
def ensure_real_sources(mini_workspace):
    """
    The test harness is writing minimal stubs into the temporary workspace's src/.
    Overwrite those with the real project files so tests exercise the actual code.
    """
    root = mini_workspace["root"]  # provided by existing fixture
    ws_src = root / "src"
    ws_src.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]  # .../src/tests -> .../src
    real_src = project_root  # points to .../src

    # Ensure API package exists
    (ws_src / "api").mkdir(parents=True, exist_ok=True)

    # Copy the two scripts under test, overwriting any stub
    shutil.copy2(real_src / "api" / "validate_cli.py", ws_src / "api" / "validate_cli.py")
    shutil.copy2(real_src / "reports_dashboard.py", ws_src / "reports_dashboard.py")

