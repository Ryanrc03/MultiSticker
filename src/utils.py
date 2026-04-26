from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: str) -> None:
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return np.random.default_rng(seed)


def batched_indices(size: int, batch_size: int, order: np.ndarray) -> Iterator[np.ndarray]:
    for start in range(0, size, batch_size):
        yield order[start : start + batch_size]


def now() -> float:
    return time.time()


def elapsed_seconds(start_time: float) -> float:
    return time.time() - start_time


def round_dict(values: Dict[str, Any], digits: int = 4) -> Dict[str, Any]:
    rounded = {}
    for key, value in values.items():
        if isinstance(value, float):
            rounded[key] = round(value, digits)
        else:
            rounded[key] = value
    return rounded


def normalize_rows(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, eps, None)
