"""Config + secret loading.

No secret is ever hardcoded: values come from the environment, loaded from a
local .env (gitignored) via python-dotenv when present. On the GPU platforms the
same variables are injected by the platform secret manager instead.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """Load .env (if present) into os.environ. Real values stay out of git."""
    load_dotenv(REPO_ROOT / ".env")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. Relative paths resolve against the repo root."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def require_env(name: str) -> str:
    """Return an env var or raise a clear error pointing at .env.example."""
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. Add it to .env "
            f"or the platform secret manager (see .env.example)."
        )
    return val


def get_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)
