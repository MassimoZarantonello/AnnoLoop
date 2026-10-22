"""Machine-specific paths, read from environment variables or from the ``.env`` file at the repo root.

Each machine has its own ``.env`` (not in git; ``.env.example`` lists the variables). Configs refer to the
variables as ``${MACROSEG_RUNS}``, so the same config works on every machine. A relative path is taken
from the repo root. A variable already set in the environment wins over ``.env``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENV_FILE = REPO / ".env"
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_env() -> None:
    """Read ``.env`` into the environment, without overriding variables that are already set."""
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=False)


def _value(name: str) -> str:
    """The raw value of variable ``name``; a missing or empty variable is an error."""
    load_env()
    text = os.environ.get(name, "").strip()
    if not text:
        raise KeyError(f"{name} is not set: add it to {ENV_FILE} (see .env.example).")
    return text


def path(name: str) -> Path:
    """Variable ``name`` as a local path (relative paths start at the repo root)."""
    return _local(_value(name))


def expand(text: str) -> str:
    """Replace every ``${NAME}`` in ``text`` with its value; relative values start at the repo root."""
    return _VARIABLE.sub(lambda match: str(_local(_value(match.group(1)))), text)


def _local(text: str) -> Path:
    p = Path(text).expanduser()
    return p if p.is_absolute() else REPO / p
