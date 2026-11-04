"""The version of this code, recorded in everything AnnoLoop writes (exported images, models, scores)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def code_commit() -> str | None:
    """``git describe --always --dirty`` of this repository; None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
