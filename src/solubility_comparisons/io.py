"""Explicit input paths and fresh output directories; no implicit workspace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import ContractError


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ContractError(f"Non-finite JSON constant {value} in {path.name}")

    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=reject_constant)
        except (ValueError, ContractError) as exc:
            raise ContractError(
                f"Invalid JSON at {path.name}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ContractError(f"Expected an object at {path.name}:{line_number}")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def fresh_directory(path: Path) -> Path:
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    # Runtime names identify files without leaking a developer's personal paths.
    return [
        dict(name=path.name, sha256=sha256(path), bytes=path.stat().st_size)
        for path in paths
    ]
