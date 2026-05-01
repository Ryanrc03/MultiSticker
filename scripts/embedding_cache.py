"""Small disk cache helpers for expensive frozen embedding passes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _update_json(hasher, value) -> None:
    hasher.update(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    hasher.update(b"\n")


def digest_rows(rows: Sequence[dict], fields: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    _update_json(hasher, {"count": len(rows), "fields": list(fields)})
    for row in rows:
        _update_json(
            hasher,
            {
                "sample_id": row.get("sample_id", ""),
                "label_id": row.get("label_id", ""),
                "values": [row.get(field, "") for field in fields],
            },
        )
    return hasher.hexdigest()[:20]


def digest_stickers(sticker_ids: Sequence[str], image_paths: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    _update_json(hasher, {"count": len(sticker_ids)})
    for sticker_id, image_path in zip(sticker_ids, image_paths):
        path = Path(image_path)
        try:
            stat = path.stat()
            file_state = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except FileNotFoundError:
            file_state = {"missing": True}
        _update_json(
            hasher, {"sticker_id": sticker_id, "path": str(path), "file": file_state}
        )
    return hasher.hexdigest()[:20]


def cache_path(cache_dir: Path, prefix: str, metadata: dict) -> Path:
    hasher = hashlib.sha256()
    _update_json(hasher, metadata)
    return cache_dir / f"{prefix}_{hasher.hexdigest()[:20]}.npz"


def load_npz(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp_path, **arrays)
    generated_path = tmp_path
    if not generated_path.exists():
        generated_path = Path(str(tmp_path) + ".npz")
    generated_path.replace(path)


def describe_cache(path: Path) -> str:
    return str(path)
