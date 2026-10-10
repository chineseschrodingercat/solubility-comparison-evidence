"""Small versioned JSON/TOML configuration with paths relative to its file."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any

from .contracts import ContractError


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".toml":
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    else:
        config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version") != 1 or not isinstance(config.get("paths", {}), dict):
        raise ContractError("Config requires version=1 and a paths object")
    for key, value in config.get("paths", {}).items():
        if not isinstance(value, str):
            raise ContractError(f"Config path {key!r} must be text")
        location = Path(value).expanduser()
        config["paths"][key] = (
            location
            if location.is_absolute()
            else (path.resolve().parent / location).resolve()
        )
    return config
