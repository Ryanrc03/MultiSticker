from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageSequence

from .utils import ensure_dir, round_dict, save_json


SUPPORTED_MEDIA = {".png", ".gif", ".webm"}
DEFAULT_SUPPORTED_MEDIA = tuple(sorted(SUPPORTED_MEDIA))
VIDEO_MEDIA = {".webm"}
MULTI_FRAME_MEDIA = {".gif", ".webm"}


@dataclass
class UStickerPaths:
    project_root: str
    scratch_root: str = "/scratch/rl182/meme"
    usticker_root: str = "/scratch/rl182/meme/u-sticker"
    session_memory_override: str | None = None
    sample_intent_override: str | None = None
    intent_cluster_override: str | None = None
    manifest_override: str | None = None
    run_name: str | None = None

    @property
    def cache_root(self) -> str:
        return str(Path(self.scratch_root) / "usticker_igsr")

    @property
    def data_root(self) -> str:
        return ensure_dir(str(Path(self.cache_root) / "data"))

    @property
    def model_root(self) -> str:
        return ensure_dir(str(Path(self.cache_root) / "models"))

    @property
    def sticker_root(self) -> str:
        return ensure_dir(str(Path(self.cache_root) / "stickers" / "final_stickers"))

    @property
    def run_root(self) -> str:
        return ensure_dir(str(Path(self.cache_root) / "runs"))

    @property
    def llm_root(self) -> str:
        return ensure_dir(str(Path(self.cache_root) / "llm"))

    @property
    def _artifact_suffix(self) -> str:
        return f"_{self.run_name}" if self.run_name else ""

    @property
    def zip_path(self) -> str:
        return str(Path(self.usticker_root) / "u-sticker-combined.zip")

    @property
    def domain_map_path(self) -> str:
        return str(Path(self.usticker_root) / "idx_to_domain.txt")

    @property
    def hf_home(self) -> str:
        return ensure_dir(str(Path(self.scratch_root) / "models" / "hf"))

    @property
    def torch_home(self) -> str:
        return ensure_dir(str(Path(self.scratch_root) / "models" / "torch"))

    @property
    def vllm_home(self) -> str:
        return ensure_dir(str(Path(self.scratch_root) / "models" / "vllm"))

    @property
    def xdg_cache_home(self) -> str:
        return ensure_dir("/scratch/rl182/cache")

    @property
    def session_memory_path(self) -> str:
        if self.session_memory_override:
            return self.session_memory_override
        return str(Path(self.llm_root) / "session_memories.jsonl")

    @property
    def sample_intent_path(self) -> str:
        if self.sample_intent_override:
            return self.sample_intent_override
        return str(Path(self.llm_root) / "sample_intents.jsonl")

    @property
    def intent_cluster_path(self) -> str:
        if self.intent_cluster_override:
            return self.intent_cluster_override
        return str(Path(self.llm_root) / f"intent_clusters{self._artifact_suffix}.json")

    @property
    def manifest_path(self) -> str:
        if self.manifest_override:
            return self.manifest_override
        return str(Path(self.data_root) / f"usticker_manifest{self._artifact_suffix}.json")

    @property
    def model_path(self) -> str:
        return str(Path(self.model_root) / f"usticker_igsr{self._artifact_suffix}.pt")

    @property
    def result_path(self) -> str:
        return str(Path(self.run_root) / f"usticker_igsr_results{self._artifact_suffix}.json")


@dataclass
class UStickerDataConfig:
    max_files: int = 0
    session_gap_hours: float = 12.0
    max_context_turns: int = 12
    min_context_turns: int = 2
    history_session_limit: int = 8
    top_k_memories: int = 3
    max_summary_turns: int = 6
    max_text_chars: int = 180
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    max_stickers: int = 0
    min_sticker_frequency: int = 1
    max_train_samples: int = 100000
    max_val_samples: int = 0
    max_test_samples: int = 0
    supported_media: tuple[str, ...] = DEFAULT_SUPPORTED_MEDIA
    memory_strategy: str = "retrieved_topk"
    seed: int = 42


@dataclass
class ModelConfig:
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    memory_model_name: str = "intfloat/multilingual-e5-small"
    train_batch_size: int = 256
    infer_batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07
    hidden_dim: int = 512
    dropout: float = 0.1
    intent_clusters: int = 64
    intent_loss_weight: float = 0.2
    group_prior_alpha: float = 2.0
    rerank_top_groups: int = 2


@dataclass
class RuntimeConfig:
    device: str = "auto"
    num_workers: int = 0


@dataclass
class UStickerIGSRConfig:
    paths: UStickerPaths
    data: UStickerDataConfig = field(default_factory=UStickerDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class SessionRecord:
    session_id: str
    file_id: str
    file_index: int
    session_index: int
    domain: str
    turns: List[dict]
    summary_text: str
    session_text: str
    split: str


@dataclass
class SampleRecord:
    sample_id: str
    session_id: str
    file_id: str
    split: str
    domain: str
    context_text: str
    current_session_text: str
    memory_text: str
    retrieved_memory_text: str
    intent_text: str
    intent_label: str
    label_id: str
    label_index: int
    intent_cluster_id: int
    history_session_ids: List[str]
    turn_index: int


class IntentGuidedRetriever(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_intents: int, dropout: float, temperature: float):
        super().__init__()
        self.intent_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.intent_classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, num_intents),
        )
        self.temperature = temperature

    def forward(
        self,
        context_features: torch.Tensor,
        memory_features: torch.Tensor,
        intent_text_features: torch.Tensor,
        image_bank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        combined = torch.cat([context_features, memory_features, intent_text_features], dim=-1)
        intent_repr = F.normalize(self.intent_proj(combined), dim=-1)
        retrieval_logits = intent_repr @ image_bank.T / self.temperature
        intent_logits = self.intent_classifier(intent_repr)
        return intent_repr, retrieval_logits, intent_logits


def default_usticker_config(project_root: str) -> UStickerIGSRConfig:
    return UStickerIGSRConfig(paths=UStickerPaths(project_root=project_root))


def _set_cache_env(config: UStickerIGSRConfig) -> None:
    cache_root = config.paths.hf_home
    os.environ["HF_HOME"] = cache_root
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_root
    os.environ["HF_HUB_CACHE"] = cache_root
    os.environ["HF_XET_CACHE"] = str(Path(config.paths.xdg_cache_home) / "huggingface" / "xet")
    os.environ["TRANSFORMERS_CACHE"] = cache_root
    os.environ["TORCH_HOME"] = config.paths.torch_home
    os.environ["XDG_CACHE_HOME"] = config.paths.xdg_cache_home
    os.environ["VLLM_CACHE_ROOT"] = config.paths.vllm_home


def _resolve_device(config: UStickerIGSRConfig) -> torch.device:
    if config.runtime.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.runtime.device)


def _seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _load_domain_map(path: str) -> Dict[str, str]:
    mapping = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            file_name, domain = line.split(",", 1)
            mapping[Path(file_name.strip()).stem.zfill(2)] = domain.strip()
    return mapping


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _normalize_text(text: object, max_chars: int) -> str:
    normalized = " ".join(str(text or "").strip().split())
    return normalized[:max_chars]


def _normalize_supported_media(media: Sequence[str]) -> tuple[str, ...]:
    normalized = sorted({str(item).strip().lower() for item in media if str(item).strip()})
    invalid = [item for item in normalized if not item.startswith(".")]
    if invalid:
        raise ValueError(f"Media extensions must start with '.': {invalid}")
    unsupported = [item for item in normalized if item not in SUPPORTED_MEDIA]
    if unsupported:
        raise ValueError(f"Unsupported media extensions: {unsupported}; supported={sorted(SUPPORTED_MEDIA)}")
    return tuple(normalized)


def _is_usable_sticker(sticker: Optional[str], supported_media: Sequence[str]) -> bool:
    if not sticker:
        return False
    if str(sticker).startswith("U-Sticker detects"):
        return False
    return Path(str(sticker)).suffix.lower() in supported_media


def _format_turn(turn: dict, max_chars: int, supported_media: Sequence[str]) -> str:
    speaker = str(turn.get("from_id", "unknown"))[:8]
    text = _normalize_text(turn.get("text"), max_chars=max_chars)
    sticker = str(turn.get("sticker", "") or "").strip()
    fields = [speaker + ":"]
    if text:
        fields.append(text)
    if _is_usable_sticker(sticker, supported_media):
        fields.append("[sticker:" + sticker + "]")
    return " ".join(fields)


def _empty_media_stats(supported_media: Sequence[str]) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "json_files": 0,
        "total_rows": 0,
        "supported_sticker_rows": 0,
        "unsupported_sticker_rows": 0,
        "offensive_rows": 0,
    }
    for suffix in sorted(SUPPORTED_MEDIA):
        stats[f"{suffix[1:]}_rows"] = 0
    for suffix in _normalize_supported_media(supported_media):
        stats[f"{suffix[1:]}_supported_rows"] = 0
    return stats


def _build_session_text(turns: Sequence[dict], max_turns: int, max_chars: int, supported_media: Sequence[str]) -> str:
    selected_turns = list(turns[-max_turns:])
    return " || ".join(_format_turn(turn, max_chars=max_chars, supported_media=supported_media) for turn in selected_turns)


def _read_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return records
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_jsonl_map(path: str, key_field: str) -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    for item in _read_jsonl(path):
        key = str(item.get(key_field, "")).strip()
        if key:
            mapping[key] = item
    return mapping


def _build_session_summary(
    turns: Sequence[dict],
    domain: str,
    max_turns: int,
    max_chars: int,
    supported_media: Sequence[str],
) -> str:
    selected_turns = list(turns[-max_turns:])
    speakers = []
    sticker_ids = []
    snippets = []
    for turn in selected_turns:
        speaker = str(turn.get("from_id", "unknown"))[:8]
        if speaker not in speakers:
            speakers.append(speaker)
        text = _normalize_text(turn.get("text"), max_chars=max_chars)
        if text:
            snippets.append(speaker + ": " + text)
        sticker = str(turn.get("sticker", "") or "").strip()
        if _is_usable_sticker(sticker, supported_media):
            sticker_ids.append(sticker)
    parts = [
        "domain: " + domain,
        "speakers: " + " ".join(speakers[:6]),
        "recent_dialogue: " + " || ".join(snippets[-max_turns:]),
        "recent_stickers: " + " ".join(sticker_ids[-max_turns:]),
    ]
    return " | ".join(part for part in parts if part.strip())


def _split_sessions(rows: Sequence[dict], gap_hours: float) -> List[List[dict]]:
    sessions: List[List[dict]] = []
    current: List[dict] = []
    previous_time: Optional[datetime] = None
    for row in sorted(rows, key=lambda item: item["datetime"]):
        current_time = _parse_datetime(str(row["datetime"]))
        if previous_time is not None:
            delta_hours = (current_time - previous_time).total_seconds() / 3600.0
            if delta_hours > gap_hours and current:
                sessions.append(current)
                current = []
        current.append(row)
        previous_time = current_time
    if current:
        sessions.append(current)
    return sessions


def _split_name(index: int, total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> str:
    if total <= 1:
        return "train"
    if test_ratio <= 0.0:
        train_cut = max(1, int(total * train_ratio))
        if index < min(train_cut, total - 1):
            return "train"
        return "val"
    train_cut = max(1, int(total * train_ratio))
    val_cut = max(train_cut + 1, int(total * (train_ratio + val_ratio)))
    if index < train_cut:
        return "train"
    if index < min(val_cut, total - 1):
        return "val"
    return "test"


def _subsample(items: List[dict], limit: int, rng: np.random.Generator) -> List[dict]:
    if limit <= 0 or len(items) <= limit:
        return items
    indices = rng.permutation(len(items))[:limit]
    return [items[int(index)] for index in indices]


def _session_memory_text(session_id: str, session_memory_map: Dict[str, dict], session_lookup: Dict[str, SessionRecord]) -> str:
    return session_memory_map.get(session_id, {}).get("session_memory_text", session_lookup[session_id].summary_text)


def _memory_text_from_history(
    memory_strategy: str,
    history_ids: Sequence[str],
    top_k_memories: int,
    session_memory_map: Dict[str, dict],
    session_lookup: Dict[str, SessionRecord],
    query_embedding: Optional[np.ndarray] = None,
    summary_embedding_map: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    if memory_strategy == "disabled":
        return "memory disabled"
    if not history_ids:
        return "no prior session memory"

    if memory_strategy == "recent_topk":
        selected_history = list(reversed(list(history_ids)[-top_k_memories:]))
    elif memory_strategy == "retrieved_topk":
        if query_embedding is None or summary_embedding_map is None:
            raise ValueError("retrieved_topk memory strategy requires query and summary embeddings")
        history_matrix = np.asarray([summary_embedding_map[item] for item in history_ids], dtype=np.float32)
        scores = history_matrix @ query_embedding
        order = np.argsort(-scores)[:top_k_memories]
        selected_history = [history_ids[int(item)] for item in order]
    else:
        raise ValueError(f"Unsupported memory strategy: {memory_strategy}")

    return " || ".join(
        _session_memory_text(item, session_memory_map=session_memory_map, session_lookup=session_lookup)
        for item in selected_history
    )


class MeanPoolingEncoder:
    def __init__(self, model_name: str, device: torch.device):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def encode(self, texts: Sequence[str], batch_size: int, prefix: str = "") -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = [prefix + text for text in texts[start : start + batch_size]]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = F.normalize(pooled, dim=-1)
                outputs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)


class OpenClipEncoder:
    def __init__(self, model_name: str, pretrained: str, device: torch.device):
        from open_clip.factory import create_model_and_transforms, get_tokenizer

        model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(device)
        self.model.eval()
        self.preprocess = preprocess
        self.tokenizer = get_tokenizer(model_name)
        self.device = device

    @property
    def output_dim(self) -> int:
        return int(self.model.text_projection.shape[1])

    def encode_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                tokens = self.tokenizer(list(texts[start : start + batch_size])).to(self.device)
                encoded = self.model.encode_text(tokens)
                encoded = F.normalize(encoded, dim=-1)
                outputs.append(encoded.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, self.output_dim), dtype=np.float32)

    def _encode_frame_batch(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in images], dim=0).to(self.device)
        encoded = self.model.encode_image(batch)
        return F.normalize(encoded, dim=-1)

    def _encode_media_item(self, image_path: str, batch_size: int) -> np.ndarray:
        frame_images = _load_sticker_frames(image_path, all_frames=True)
        frame_batch_size = max(1, min(batch_size, 32))
        frame_features = []
        with torch.no_grad():
            for start in range(0, len(frame_images), frame_batch_size):
                encoded = self._encode_frame_batch(frame_images[start : start + frame_batch_size])
                frame_features.append(encoded)
        pooled = torch.cat(frame_features, dim=0).mean(dim=0)
        pooled = F.normalize(pooled.unsqueeze(0), dim=-1).squeeze(0)
        return pooled.cpu().numpy().astype(np.float32)

    def encode_images(self, image_paths: Sequence[str], batch_size: int) -> np.ndarray:
        outputs = np.zeros((len(image_paths), self.output_dim), dtype=np.float32)
        static_indices = [
            index for index, image_path in enumerate(image_paths) if Path(str(image_path)).suffix.lower() not in MULTI_FRAME_MEDIA
        ]
        animated_indices = [
            index for index, image_path in enumerate(image_paths) if Path(str(image_path)).suffix.lower() in MULTI_FRAME_MEDIA
        ]
        with torch.no_grad():
            for start in range(0, len(static_indices), batch_size):
                batch_indices = static_indices[start : start + batch_size]
                images = []
                for item_index in batch_indices:
                    image = _load_sticker_image(image_paths[item_index])
                    images.append(self.preprocess(image))
                # A still image is treated as a one-frame clip, so this batched path is
                # mathematically equivalent to mean-pooling over a single frame.
                batch = torch.stack(images, dim=0).to(self.device)
                encoded = self.model.encode_image(batch)
                encoded = F.normalize(encoded, dim=-1)
                outputs[np.asarray(batch_indices, dtype=np.int64)] = encoded.cpu().numpy().astype(np.float32)

            for item_index in animated_indices:
                outputs[item_index] = self._encode_media_item(image_paths[item_index], batch_size=batch_size)
        return outputs


def _resolve_ffmpeg_binary() -> str:
    ffmpeg_binary = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if ffmpeg_binary:
        return ffmpeg_binary
    raise RuntimeError(
        "ffmpeg was not found in PATH. For animated video stickers such as .webm, "
        "load FFmpeg first, e.g. `module load GCCcore/13.3.0 FFmpeg/7.0.2`."
    )


def _extract_video_frames_with_ffmpeg(path: str, all_frames: bool) -> List[Image.Image]:
    ffmpeg_binary = _resolve_ffmpeg_binary()
    with tempfile.TemporaryDirectory(prefix="usticker_frames_") as tmpdir:
        output_pattern = str(Path(tmpdir) / "frame_%06d.png")
        command = [ffmpeg_binary, "-v", "error", "-i", path]
        if all_frames:
            command += ["-vsync", "0"]
        else:
            command += ["-frames:v", "1"]
        command.append(output_pattern)
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to decode animated sticker via ffmpeg: {path}\n{completed.stderr.decode('utf-8', errors='ignore')}"
            )
        frame_paths = sorted(Path(tmpdir).glob("frame_*.png"))
        if not frame_paths:
            raise RuntimeError(f"ffmpeg produced no frames for animated sticker: {path}")
        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as frame_image:
                frames.append(frame_image.convert("RGB").copy())
        return frames


def _load_sticker_frames(path: str, all_frames: bool = True) -> List[Image.Image]:
    suffix = Path(path).suffix.lower()
    if suffix in VIDEO_MEDIA:
        return _extract_video_frames_with_ffmpeg(path, all_frames=all_frames)

    with Image.open(path) as image:
        if getattr(image, "is_animated", False):
            iterator = ImageSequence.Iterator(image)
            if all_frames:
                frames = [frame.convert("RGB").copy() for frame in iterator]
            else:
                first_frame = next(iterator)
                frames = [first_frame.convert("RGB").copy()]
        else:
            frames = [image.convert("RGB").copy()]
    if not frames:
        raise RuntimeError(f"No decodable frames found for sticker: {path}")
    return frames


def _load_sticker_image(path: str) -> Image.Image:
    return _load_sticker_frames(path, all_frames=False)[0]


def _extract_missing_stickers(zip_path: str, output_root: str, sticker_ids: Sequence[str]) -> Dict[str, str]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [sticker_id for sticker_id in sticker_ids if not (output_dir / sticker_id).exists()]
    if missing:
        with zipfile.ZipFile(zip_path, "r") as archive:
            available = set(archive.namelist())
            for sticker_id in missing:
                archive_name = "final_stickers/" + sticker_id
                if archive_name not in available:
                    continue
                archive.extract(archive_name, path=output_dir.parent)
    return {sticker_id: str(output_dir / sticker_id) for sticker_id in sticker_ids if (output_dir / sticker_id).exists()}


def _filter_decodable_stickers(sticker_paths: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, int]]:
    usable: Dict[str, str] = {}
    stats = {"checked": 0, "kept": 0, "failed": 0}
    for sticker_id, path in sticker_paths.items():
        stats["checked"] += 1
        try:
            _load_sticker_image(path)
        except Exception:
            stats["failed"] += 1
            continue
        usable[sticker_id] = path
        stats["kept"] += 1
    return usable, stats


def _kmeans_assignments(vectors: np.ndarray, clusters: int, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clusters = max(1, min(clusters, len(vectors)))
    centers = vectors[rng.choice(len(vectors), size=clusters, replace=False)].copy()
    assignments = np.zeros(len(vectors), dtype=np.int64)
    for _ in range(iterations):
        scores = vectors @ centers.T
        assignments = scores.argmax(axis=1)
        for cluster_index in range(clusters):
            mask = assignments == cluster_index
            if not mask.any():
                centers[cluster_index] = vectors[rng.integers(0, len(vectors))]
                continue
            center = vectors[mask].mean(axis=0)
            center /= np.linalg.norm(center) + 1e-8
            centers[cluster_index] = center.astype(np.float32)
    return assignments


def _metrics_from_scores(score_matrix: np.ndarray, gold_indices: np.ndarray) -> Dict[str, float]:
    if len(gold_indices) == 0:
        return {"p@1": 0.0, "p@3": 0.0, "p@5": 0.0, "p@10": 0.0, "p@30": 0.0, "map": 0.0, "mrr": 0.0}
    order = np.argsort(-score_matrix, axis=1)
    ranks = []
    for row_index, gold_index in enumerate(gold_indices):
        rank = int(np.where(order[row_index] == int(gold_index))[0][0]) + 1
        ranks.append(rank)
    ranks_np = np.asarray(ranks, dtype=np.float32)
    hit_rates = {
        "p@1": float((ranks_np <= 1).mean()),
        "p@3": float((ranks_np <= 3).mean()),
        "p@5": float((ranks_np <= 5).mean()),
        "p@10": float((ranks_np <= 10).mean()),
        "p@30": float((ranks_np <= 30).mean()),
    }
    return round_dict(
        hit_rates | {"map": float((1.0 / ranks_np).mean()), "mrr": float((1.0 / ranks_np).mean())}
    )


def _group_metrics_from_scores(
    score_matrix: np.ndarray,
    gold_indices: np.ndarray,
    sticker_group_ids: np.ndarray,
) -> Dict[str, float]:
    if len(gold_indices) == 0:
        return {"p@1": 0.0, "p@3": 0.0, "p@5": 0.0, "p@10": 0.0, "p@30": 0.0, "map": 0.0, "mrr": 0.0}
    order = np.argsort(-score_matrix, axis=1)
    ranks = []
    for row_index, gold_index in enumerate(gold_indices):
        gold_group = int(sticker_group_ids[int(gold_index)])
        group_rank = len(sticker_group_ids) + 1
        for candidate_rank, candidate_index in enumerate(order[row_index], start=1):
            if int(sticker_group_ids[int(candidate_index)]) == gold_group:
                group_rank = candidate_rank
                break
        ranks.append(group_rank)
    ranks_np = np.asarray(ranks, dtype=np.float32)
    hit_rates = {
        "p@1": float((ranks_np <= 1).mean()),
        "p@3": float((ranks_np <= 3).mean()),
        "p@5": float((ranks_np <= 5).mean()),
        "p@10": float((ranks_np <= 10).mean()),
        "p@30": float((ranks_np <= 30).mean()),
    }
    return round_dict(
        hit_rates | {"map": float((1.0 / ranks_np).mean()), "mrr": float((1.0 / ranks_np).mean())}
    )


def _fuse_group_prior_scores(
    score_matrix: np.ndarray,
    intent_logit_matrix: np.ndarray,
    sticker_group_ids: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if alpha <= 0.0:
        return score_matrix
    stabilized = intent_logit_matrix - intent_logit_matrix.max(axis=1, keepdims=True)
    group_log_probs = stabilized - np.log(np.exp(stabilized).sum(axis=1, keepdims=True) + 1e-8)
    sticker_group_log_probs = group_log_probs[:, sticker_group_ids]
    return score_matrix + alpha * sticker_group_log_probs.astype(np.float32)


def _two_stage_group_rerank_scores(
    score_matrix: np.ndarray,
    intent_logit_matrix: np.ndarray,
    sticker_group_ids: np.ndarray,
    top_groups: int,
    alpha: float,
) -> np.ndarray:
    fused_scores = _fuse_group_prior_scores(score_matrix, intent_logit_matrix, sticker_group_ids, alpha=alpha)
    if top_groups <= 0 or intent_logit_matrix.size == 0:
        return fused_scores
    top_groups = min(top_groups, intent_logit_matrix.shape[1])
    top_group_ids = np.argsort(-intent_logit_matrix, axis=1)[:, :top_groups]
    allowed_mask = (sticker_group_ids[None, :] == top_group_ids[:, :, None]).any(axis=1)
    masked_scores = fused_scores.copy()
    masked_scores[~allowed_mask] = -1e9
    return masked_scores


def _build_raw_sessions(config: UStickerIGSRConfig) -> tuple[List[SessionRecord], Dict[str, List[str]], Dict[str, int]]:
    root = Path(config.paths.usticker_root)
    domain_map = _load_domain_map(config.paths.domain_map_path)
    json_files = sorted(root.glob("*.json"))
    if config.data.max_files > 0:
        json_files = json_files[: config.data.max_files]
    supported_media = _normalize_supported_media(config.data.supported_media)

    all_sessions: List[SessionRecord] = []
    sessions_by_file: Dict[str, List[str]] = defaultdict(list)
    media_stats = _empty_media_stats(supported_media)
    media_stats["json_files"] = len(json_files)
    for file_index, json_path in enumerate(json_files):
        print(f"[prepare] loading {json_path.name} ({file_index + 1}/{len(json_files)})", flush=True)
        with open(json_path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        media_stats["total_rows"] += len(rows)
        for row in rows:
            sticker = row.get("sticker")
            if not sticker:
                continue
            sticker = str(sticker)
            if sticker.startswith("U-Sticker detects"):
                media_stats["offensive_rows"] += 1
                continue
            suffix = Path(sticker).suffix.lower()
            if suffix in SUPPORTED_MEDIA:
                media_stats[f"{suffix[1:]}_rows"] += 1
                if suffix in supported_media:
                    media_stats["supported_sticker_rows"] += 1
                    media_stats[f"{suffix[1:]}_supported_rows"] += 1
                else:
                    media_stats["unsupported_sticker_rows"] += 1
        file_id = json_path.stem.zfill(2)
        domain = domain_map.get(file_id, "Unknown")
        sessions = _split_sessions(rows, gap_hours=config.data.session_gap_hours)
        for session_index, turns in enumerate(sessions):
            session_id = f"{file_id}_session_{session_index:04d}"
            summary_text = _build_session_summary(
                turns=turns,
                domain=domain,
                max_turns=config.data.max_summary_turns,
                max_chars=config.data.max_text_chars,
                supported_media=supported_media,
            )
            session_text = _build_session_text(
                turns=turns,
                max_turns=max(config.data.max_summary_turns * 2, config.data.max_summary_turns),
                max_chars=config.data.max_text_chars,
                supported_media=supported_media,
            )
            record = SessionRecord(
                session_id=session_id,
                file_id=file_id,
                file_index=file_index,
                session_index=session_index,
                domain=domain,
                turns=turns,
                summary_text=summary_text,
                session_text=session_text,
                split="train",
            )
            all_sessions.append(record)
            sessions_by_file[file_id].append(session_id)
        print(
            f"[prepare] {json_path.name}: rows={len(rows)} sessions={len(sessions)} cumulative_sessions={len(all_sessions)}",
            flush=True,
        )
    print(
        "[prepare] media counts "
        + f"json_files={media_stats['json_files']} total_rows={media_stats['total_rows']} "
        + f"supported_media={list(supported_media)} "
        + f"supported_sticker_rows={media_stats['supported_sticker_rows']} "
        + f"unsupported_sticker_rows={media_stats['unsupported_sticker_rows']} "
        + f"png_rows={media_stats.get('png_rows', 0)} "
        + f"gif_rows={media_stats.get('gif_rows', 0)} "
        + f"webm_rows={media_stats.get('webm_rows', 0)} "
        + f"offensive_rows={media_stats['offensive_rows']}",
        flush=True,
    )
    ordered_sessions = sorted(all_sessions, key=lambda item: (item.file_index, item.session_index))
    rng = np.random.default_rng(config.data.seed)
    permutation = rng.permutation(len(ordered_sessions))
    shuffled_sessions = [ordered_sessions[int(index)] for index in permutation]
    for ordered_index, session in enumerate(ordered_sessions):
        session.split = "train"
    for ordered_index, session in enumerate(shuffled_sessions):
        session.split = _split_name(
            ordered_index,
            len(shuffled_sessions),
            config.data.train_ratio,
            config.data.val_ratio,
            config.data.test_ratio,
        )
    return all_sessions, sessions_by_file, media_stats


def _build_dataset_manifest(config: UStickerIGSRConfig, device: torch.device) -> dict:
    print("[prepare] building raw sessions", flush=True)
    sessions, sessions_by_file, media_stats = _build_raw_sessions(config)
    supported_media = _normalize_supported_media(config.data.supported_media)
    session_lookup = {session.session_id: session for session in sessions}
    session_memory_map = _load_jsonl_map(config.paths.session_memory_path, "session_id")
    sample_intent_map = _load_jsonl_map(config.paths.sample_intent_path, "sample_id")
    intent_cluster_data = {}
    intent_cluster_map: Dict[str, dict] = {}
    intent_cluster_path = Path(config.paths.intent_cluster_path)
    if intent_cluster_path.exists():
        with open(intent_cluster_path, "r", encoding="utf-8") as handle:
            intent_cluster_data = json.load(handle)
        intent_cluster_map = {
            str(sticker_id): dict(values)
            for sticker_id, values in intent_cluster_data.get("stickers", {}).items()
        }
    train_sticker_counts: Counter[str] = Counter()

    provisional_samples: List[dict] = []
    for session_index, session in enumerate(sessions):
        history_ids = sessions_by_file[session.file_id][: session.session_index][-config.data.history_session_limit :]
        formatted_turns = [
            _format_turn(turn, max_chars=config.data.max_text_chars, supported_media=supported_media)
            for turn in session.turns
        ]
        current_window = max(config.data.max_context_turns * 2, config.data.max_context_turns)
        for turn_index, turn in enumerate(session.turns):
            sticker_id = str(turn.get("sticker", "") or "").strip()
            if not _is_usable_sticker(sticker_id, config.data.supported_media):
                continue
            if turn_index < config.data.min_context_turns:
                continue
            context_text = " || ".join(formatted_turns[max(0, turn_index - config.data.max_context_turns) : turn_index])
            if not context_text.strip():
                continue
            current_session_text = " || ".join(formatted_turns[max(0, turn_index - current_window) : turn_index])
            row = {
                "sample_id": session.session_id + "#turn" + str(turn_index),
                "session_id": session.session_id,
                "file_id": session.file_id,
                "split": session.split,
                "domain": session.domain,
                "context_text": context_text,
                "current_session_text": current_session_text,
                "label_id": sticker_id,
                "history_session_ids": history_ids,
                "turn_index": turn_index,
            }
            provisional_samples.append(row)
            if session.split == "train":
                train_sticker_counts[sticker_id] += 1
        if (session_index + 1) % 100 == 0 or session_index + 1 == len(sessions):
            print(
                f"[prepare] scanned sessions={session_index + 1}/{len(sessions)} provisional_samples={len(provisional_samples)}",
                flush=True,
            )

    full_train_rows = [row for row in provisional_samples if row["split"] == "train"]
    full_val_rows = [row for row in provisional_samples if row["split"] == "val"]
    full_test_rows = [row for row in provisional_samples if row["split"] == "test"]
    print(
        f"[prepare] full split sizes before label filtering train={len(full_train_rows)} "
        + f"val={len(full_val_rows)} test={len(full_test_rows)}",
        flush=True,
    )

    eligible = [
        item
        for item, count in train_sticker_counts.items()
        if count >= config.data.min_sticker_frequency and Path(item).suffix.lower() in config.data.supported_media
    ]
    eligible = sorted(eligible, key=lambda item: (-train_sticker_counts[item], item))
    if config.data.max_stickers and config.data.max_stickers > 0:
        eligible = eligible[: config.data.max_stickers]
    sticker_to_index = {sticker_id: index for index, sticker_id in enumerate(eligible)}

    rng = _seed_everything(config.data.seed)
    train_rows = [row for row in full_train_rows if row["label_id"] in sticker_to_index]
    full_train_before_cap = len(train_rows)
    train_rows = _subsample(train_rows, config.data.max_train_samples, rng)
    train_label_set = {row["label_id"] for row in train_rows}
    valid_sticker_to_index = {sticker_id: index for index, sticker_id in enumerate(sorted(train_label_set))}
    train_rows = [row for row in train_rows if row["label_id"] in valid_sticker_to_index]
    val_rows = [row for row in full_val_rows if row["label_id"] in valid_sticker_to_index]
    val_rows = _subsample(val_rows, config.data.max_val_samples, rng)
    test_rows = [row for row in full_test_rows if row["label_id"] in valid_sticker_to_index]
    test_rows = _subsample(test_rows, config.data.max_test_samples, rng)
    sticker_to_index = valid_sticker_to_index
    eligible = sorted(train_label_set)
    selected_rows = train_rows + val_rows + test_rows
    print(
        f"[prepare] full_train_size_before_cap={full_train_before_cap} "
        + f"train_size_after_cap={len(train_rows)} "
        + f"valid_size_after_filtering={len(val_rows)} "
        + f"final_sticker_pool={len(eligible)}",
        flush=True,
    )

    memory_strategy = str(config.data.memory_strategy).strip().lower()
    summary_embedding_map: Dict[str, np.ndarray] | None = None
    query_embeddings: np.ndarray | None = None
    if memory_strategy == "retrieved_topk":
        print("[prepare] loading memory retriever", flush=True)
        memory_encoder = MeanPoolingEncoder(config.model.memory_model_name, device=device)
        session_ids = [session.session_id for session in sessions]
        print(f"[prepare] encoding {len(session_ids)} session summaries", flush=True)
        session_memory_texts = [
            _session_memory_text(session_id, session_memory_map=session_memory_map, session_lookup=session_lookup)
            for session_id in session_ids
        ]
        summary_embeddings = memory_encoder.encode(
            session_memory_texts,
            batch_size=config.model.infer_batch_size,
            prefix="passage: ",
        )
        summary_embedding_map = {session_id: summary_embeddings[index] for index, session_id in enumerate(session_ids)}
        query_embeddings = memory_encoder.encode(
            [row["current_session_text"] for row in selected_rows],
            batch_size=config.model.infer_batch_size,
            prefix="query: ",
        )
        print(f"[prepare] encoding {len(selected_rows)} sample queries", flush=True)
    elif memory_strategy in {"recent_topk", "disabled"}:
        print(f"[prepare] memory strategy={memory_strategy}; skipping semantic memory retrieval", flush=True)
    else:
        raise ValueError(f"Unsupported memory strategy: {config.data.memory_strategy}")

    samples_by_split = {"train": [], "val": [], "test": []}
    for row_index, row in enumerate(selected_rows):
        history_ids = row["history_session_ids"]
        memory_text = _memory_text_from_history(
            memory_strategy=memory_strategy,
            history_ids=history_ids,
            top_k_memories=config.data.top_k_memories,
            session_memory_map=session_memory_map,
            session_lookup=session_lookup,
            query_embedding=None if query_embeddings is None else query_embeddings[row_index],
            summary_embedding_map=summary_embedding_map,
        )
        sample_intent_payload = sample_intent_map.get(row["sample_id"], {})
        intent_text = sample_intent_payload.get("intent_text", row["current_session_text"])
        intent_label = sample_intent_payload.get("intent_label", "neutral_acknowledgment")
        sample = SampleRecord(
            sample_id=row["sample_id"],
            session_id=row["session_id"],
            file_id=row["file_id"],
            split=row["split"],
            domain=row["domain"],
            context_text=row["context_text"],
            current_session_text=row["current_session_text"],
            memory_text=memory_text,
            retrieved_memory_text=memory_text,
            intent_text=intent_text,
            intent_label=str(intent_label),
            label_id=row["label_id"],
            label_index=sticker_to_index[row["label_id"]],
            intent_cluster_id=int(intent_cluster_map.get(row["label_id"], {}).get("intent_cluster_id", -1)),
            history_session_ids=list(history_ids),
            turn_index=int(row["turn_index"]),
        )
        samples_by_split[row["split"]].append(asdict(sample))

    manifest = {
        "config": {
            "data": asdict(config.data),
            "model": asdict(config.model),
        },
        "sticker_ids": eligible,
        "sticker_to_index": sticker_to_index,
        "sessions": [
            {
                "session_id": session.session_id,
                "file_id": session.file_id,
                "file_index": session.file_index,
                "session_index": session.session_index,
                "domain": session.domain,
                "summary_text": session.summary_text,
                "session_text": session.session_text,
                "session_memory_text": session_memory_map.get(session.session_id, {}).get("session_memory_text", session.summary_text),
                "split": session.split,
            }
            for session in sessions
        ],
        "splits": samples_by_split,
        "dataset_summary": {
            "session_count": len(sessions),
            "supported_media_provisional_samples": len(provisional_samples),
            "full_train_size_before_cap": full_train_before_cap,
            "candidate_sticker_count": len(eligible),
            "memory_strategy": memory_strategy,
            "train_samples": len(samples_by_split["train"]),
            "val_samples": len(samples_by_split["val"]),
            "test_samples": len(samples_by_split["test"]),
        },
        "media_counts": media_stats,
    }
    manifest_path = Path(config.paths.manifest_path)
    save_json(manifest, str(manifest_path))
    print(f"[prepare] manifest saved to {manifest_path}", flush=True)
    return manifest


def prepare_manifest(config: UStickerIGSRConfig, force_rebuild: bool = False) -> dict:
    _set_cache_env(config)
    config.data.supported_media = _normalize_supported_media(config.data.supported_media)
    manifest_path = Path(config.paths.manifest_path)
    if manifest_path.exists() and not force_rebuild:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    device = _resolve_device(config)
    return _build_dataset_manifest(config, device=device)


def _gather_split_arrays(
    split_rows: Sequence[dict],
    context_embeddings: np.ndarray,
    memory_embeddings: np.ndarray,
    intent_text_embeddings: np.ndarray,
    intent_labels: Dict[str, int],
) -> dict:
    label_indices = np.asarray([int(row["label_index"]) for row in split_rows], dtype=np.int64)
    intent_indices = np.asarray([int(intent_labels[row["label_id"]]) for row in split_rows], dtype=np.int64)
    return {
        "rows": list(split_rows),
        "context_embeddings": context_embeddings.astype(np.float32),
        "memory_embeddings": memory_embeddings.astype(np.float32),
        "intent_text_embeddings": intent_text_embeddings.astype(np.float32),
        "label_indices": label_indices,
        "intent_indices": intent_indices,
    }


def train_usticker_igsr(config: UStickerIGSRConfig, force_rebuild: bool = False) -> dict:
    _set_cache_env(config)
    rng = _seed_everything(config.data.seed)
    device = _resolve_device(config)
    manifest = prepare_manifest(config=config, force_rebuild=force_rebuild)
    print("[train] extracting candidate stickers", flush=True)

    sticker_ids = manifest["sticker_ids"]
    sticker_paths = _extract_missing_stickers(
        zip_path=config.paths.zip_path,
        output_root=config.paths.sticker_root,
        sticker_ids=sticker_ids,
    )
    sticker_paths, decode_stats = _filter_decodable_stickers(sticker_paths)
    sticker_ids = [sticker_id for sticker_id in sticker_ids if sticker_id in sticker_paths]
    sticker_to_index = {sticker_id: index for index, sticker_id in enumerate(sticker_ids)}
    print(f"[train] decodable stickers kept={decode_stats['kept']} failed={decode_stats['failed']}", flush=True)

    for split_name in ["train", "val", "test"]:
        manifest["splits"][split_name] = [
            row for row in manifest["splits"][split_name] if row["label_id"] in sticker_to_index
        ]
        for row in manifest["splits"][split_name]:
            row["label_index"] = sticker_to_index[row["label_id"]]

    clip_encoder = OpenClipEncoder(
        model_name=config.model.clip_model_name,
        pretrained=config.model.clip_pretrained,
        device=device,
    )
    memory_encoder = MeanPoolingEncoder(config.model.memory_model_name, device=device)
    print("[train] encoding sticker images", flush=True)

    image_embeddings = clip_encoder.encode_images(
        [sticker_paths[sticker_id] for sticker_id in sticker_ids],
        batch_size=config.model.infer_batch_size,
    )

    all_rows = manifest["splits"]["train"] + manifest["splits"]["val"] + manifest["splits"]["test"]
    print(f"[train] encoding context texts for {len(all_rows)} samples", flush=True)
    all_context_embeddings = clip_encoder.encode_texts(
        [row["context_text"] for row in all_rows],
        batch_size=config.model.infer_batch_size,
    )
    print(f"[train] encoding memory texts for {len(all_rows)} samples", flush=True)
    all_memory_embeddings = clip_encoder.encode_texts(
        [row["memory_text"] for row in all_rows],
        batch_size=config.model.infer_batch_size,
    )
    print(f"[train] encoding intent texts for {len(all_rows)} samples", flush=True)
    all_intent_text_embeddings = clip_encoder.encode_texts(
        [row["intent_text"] for row in all_rows],
        batch_size=config.model.infer_batch_size,
    )
    train_rows = manifest["splits"]["train"]
    all_rows_by_sticker: Dict[str, List[dict]] = defaultdict(list)
    for row in manifest["splits"]["train"] + manifest["splits"]["val"] + manifest["splits"]["test"]:
        all_rows_by_sticker[str(row["label_id"])].append(row)

    sticker_group_name_map: Dict[str, str] = {}
    group_name_to_index: Dict[str, int] = {}
    sticker_group_ids_list: List[int] = []
    for sticker_id in sticker_ids:
        train_group_counts = Counter(
            str(row.get("intent_label", "neutral_acknowledgment"))
            for row in train_rows
            if row["label_id"] == sticker_id
        )
        if train_group_counts:
            group_name = sorted(train_group_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        else:
            fallback_counts = Counter(
                str(row.get("intent_label", "neutral_acknowledgment"))
                for row in all_rows_by_sticker.get(sticker_id, [])
            )
            group_name = (
                sorted(fallback_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
                if fallback_counts
                else "neutral_acknowledgment"
            )
        if group_name not in group_name_to_index:
            group_name_to_index[group_name] = len(group_name_to_index)
        sticker_group_name_map[sticker_id] = group_name
        sticker_group_ids_list.append(group_name_to_index[group_name])

    sticker_group_ids = np.asarray(sticker_group_ids_list, dtype=np.int64)
    intent_labels = {sticker_id: int(sticker_group_ids[index]) for index, sticker_id in enumerate(sticker_ids)}
    intent_cluster_count = max(1, len(group_name_to_index))
    save_json(
        {
            "semantic_cluster_count": int(intent_cluster_count),
            "intent_cluster_count": int(intent_cluster_count),
            "group_name_to_index": group_name_to_index,
            "stickers": {
                sticker_id: {
                    "semantic_cluster_id": int(sticker_group_ids[index]),
                    "intent_cluster_id": int(sticker_group_ids[index]),
                    "intent_group_name": sticker_group_name_map[sticker_id],
                }
                for index, sticker_id in enumerate(sticker_ids)
            },
        },
        config.paths.intent_cluster_path,
    )

    train_count = len(manifest["splits"]["train"])
    val_count = len(manifest["splits"]["val"])
    train_rows = manifest["splits"]["train"]
    val_rows = manifest["splits"]["val"]
    test_rows = manifest["splits"]["test"]

    train_context = all_context_embeddings[:train_count]
    val_context = all_context_embeddings[train_count : train_count + val_count]
    test_context = all_context_embeddings[train_count + val_count :]

    train_memory = all_memory_embeddings[:train_count]
    val_memory = all_memory_embeddings[train_count : train_count + val_count]
    test_memory = all_memory_embeddings[train_count + val_count :]

    train_intent_text = all_intent_text_embeddings[:train_count]
    val_intent_text = all_intent_text_embeddings[train_count : train_count + val_count]
    test_intent_text = all_intent_text_embeddings[train_count + val_count :]

    split_tensors = {
        "train": _gather_split_arrays(train_rows, train_context, train_memory, train_intent_text, intent_labels),
        "val": _gather_split_arrays(val_rows, val_context, val_memory, val_intent_text, intent_labels),
        "test": _gather_split_arrays(test_rows, test_context, test_memory, test_intent_text, intent_labels),
    }

    image_bank = torch.from_numpy(image_embeddings).to(device)
    model = IntentGuidedRetriever(
        input_dim=train_context.shape[1] + train_memory.shape[1] + train_intent_text.shape[1],
        output_dim=image_embeddings.shape[1],
        hidden_dim=config.model.hidden_dim,
        num_intents=int(intent_cluster_count),
        dropout=config.model.dropout,
        temperature=config.model.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )

    history = []
    best_state = None
    best_score = -1.0
    for epoch in range(1, config.model.epochs + 1):
        model.train()
        order = rng.permutation(len(split_tensors["train"]["rows"]))
        losses = []
        retrieval_losses = []
        intent_losses = []
        for start in range(0, len(order), config.model.train_batch_size):
            batch_indices = order[start : start + config.model.train_batch_size]
            context_batch = torch.from_numpy(split_tensors["train"]["context_embeddings"][batch_indices]).to(device)
            memory_batch = torch.from_numpy(split_tensors["train"]["memory_embeddings"][batch_indices]).to(device)
            intent_text_batch = torch.from_numpy(split_tensors["train"]["intent_text_embeddings"][batch_indices]).to(device)
            labels_batch = torch.from_numpy(split_tensors["train"]["label_indices"][batch_indices]).to(device)
            intent_batch = torch.from_numpy(split_tensors["train"]["intent_indices"][batch_indices]).to(device)

            _, retrieval_logits, intent_logits = model(context_batch, memory_batch, intent_text_batch, image_bank)
            retrieval_loss = F.cross_entropy(retrieval_logits, labels_batch)
            intent_loss = F.cross_entropy(intent_logits, intent_batch)
            loss = retrieval_loss + config.model.intent_loss_weight * intent_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            retrieval_losses.append(float(retrieval_loss.item()))
            intent_losses.append(float(intent_loss.item()))

        val_metrics = evaluate_split(
            model,
            split_tensors["val"],
            image_bank,
            sticker_group_ids,
            device,
            group_prior_alpha=config.model.group_prior_alpha,
            rerank_top_groups=config.model.rerank_top_groups,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(np.mean(losses)), 4),
                "retrieval_loss": round(float(np.mean(retrieval_losses)), 4),
                "intent_loss": round(float(np.mean(intent_losses)), 4),
                "val_p@1": val_metrics["metrics"]["p@1"],
                "val_map": val_metrics["semantic_metrics"]["map"],
                "val_group_p@30": val_metrics["semantic_metrics"]["p@30"],
                "val_fused_group_p@30": val_metrics["fused_semantic_metrics"]["p@30"],
                "val_two_stage_group_p@30": val_metrics["two_stage_semantic_metrics"]["p@30"],
            }
        )
        score = val_metrics["two_stage_semantic_metrics"]["p@30"]
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"[train] epoch={epoch} loss={history[-1]['train_loss']} "
            + f"exact_val_p@1={history[-1]['val_p@1']} "
            + f"group_val_p@30={history[-1]['val_group_p@30']} "
            + f"fused_group_val_p@30={history[-1]['val_fused_group_p@30']} "
            + f"two_stage_group_val_p@30={history[-1]['val_two_stage_group_p@30']}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = Path(config.paths.model_path)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "data": asdict(config.data),
                "model": asdict(config.model),
            },
            "sticker_ids": sticker_ids,
            "intent_cluster_count": intent_cluster_count,
            "semantic_cluster_count": intent_cluster_count,
        },
        str(model_path),
    )

    val_result = evaluate_split(
        model,
        split_tensors["val"],
        image_bank,
        sticker_group_ids,
        device,
        group_prior_alpha=config.model.group_prior_alpha,
        rerank_top_groups=config.model.rerank_top_groups,
    )
    test_result = evaluate_split(
        model,
        split_tensors["test"],
        image_bank,
        sticker_group_ids,
        device,
        group_prior_alpha=config.model.group_prior_alpha,
        rerank_top_groups=config.model.rerank_top_groups,
    )
    results = {
        "config": {
            "data": asdict(config.data),
            "model": asdict(config.model),
            "runtime": {
                "device": str(device),
                "cuda_available": bool(torch.cuda.is_available()),
                "torch_version": torch.__version__,
            },
        },
        "dataset_summary": manifest["dataset_summary"],
        "media_summary": {
            "sticker_bank_size": len(sticker_ids),
            "supported_media": list(config.data.supported_media),
            "extracted_stickers": len(sticker_paths),
        },
        "cluster_summary": {
            "semantic_cluster_count": int(intent_cluster_count),
            "intent_cluster_count": int(intent_cluster_count),
            "group_name_to_index": group_name_to_index,
        },
        "rerank_summary": {
            "group_prior_alpha": float(config.model.group_prior_alpha),
            "rerank_top_groups": int(config.model.rerank_top_groups),
        },
        "training_history": history,
        "val": val_result,
        "test": test_result,
        "artifacts": {
            "manifest_path": config.paths.manifest_path,
            "model_path": str(model_path),
            "intent_cluster_path": config.paths.intent_cluster_path,
        },
    }
    result_path = Path(config.paths.result_path)
    save_json(results, str(result_path))
    return results


def evaluate_split(
    model: IntentGuidedRetriever,
    split_arrays: dict,
    image_bank: torch.Tensor,
    sticker_group_ids: np.ndarray,
    device: torch.device,
    group_prior_alpha: float = 2.0,
    rerank_top_groups: int = 2,
) -> dict:
    model.eval()
    score_chunks = []
    intent_logit_chunks = []
    with torch.no_grad():
        for start in range(0, len(split_arrays["rows"]), 512):
            context_batch = torch.from_numpy(split_arrays["context_embeddings"][start : start + 512]).to(device)
            memory_batch = torch.from_numpy(split_arrays["memory_embeddings"][start : start + 512]).to(device)
            intent_text_batch = torch.from_numpy(split_arrays["intent_text_embeddings"][start : start + 512]).to(device)
            _, retrieval_logits, intent_logits = model(context_batch, memory_batch, intent_text_batch, image_bank)
            score_chunks.append(retrieval_logits.cpu().numpy().astype(np.float32))
            intent_logit_chunks.append(intent_logits.cpu().numpy().astype(np.float32))
    score_matrix = np.concatenate(score_chunks, axis=0) if score_chunks else np.zeros((0, image_bank.shape[0]), dtype=np.float32)
    intent_logit_matrix = (
        np.concatenate(intent_logit_chunks, axis=0)
        if intent_logit_chunks
        else np.zeros((0, int(sticker_group_ids.max()) + 1 if len(sticker_group_ids) else 0), dtype=np.float32)
    )
    metrics = _metrics_from_scores(score_matrix, split_arrays["label_indices"])
    semantic_metrics = _group_metrics_from_scores(score_matrix, split_arrays["label_indices"], sticker_group_ids)
    fused_score_matrix = _fuse_group_prior_scores(score_matrix, intent_logit_matrix, sticker_group_ids, alpha=group_prior_alpha)
    fused_metrics = _metrics_from_scores(fused_score_matrix, split_arrays["label_indices"])
    fused_semantic_metrics = _group_metrics_from_scores(fused_score_matrix, split_arrays["label_indices"], sticker_group_ids)
    two_stage_score_matrix = _two_stage_group_rerank_scores(
        score_matrix,
        intent_logit_matrix,
        sticker_group_ids,
        top_groups=rerank_top_groups,
        alpha=group_prior_alpha,
    )
    two_stage_metrics = _metrics_from_scores(two_stage_score_matrix, split_arrays["label_indices"])
    two_stage_semantic_metrics = _group_metrics_from_scores(two_stage_score_matrix, split_arrays["label_indices"], sticker_group_ids)
    return {
        "metrics": metrics,
        "semantic_metrics": semantic_metrics,
        "fused_metrics": fused_metrics,
        "fused_semantic_metrics": fused_semantic_metrics,
        "two_stage_metrics": two_stage_metrics,
        "two_stage_semantic_metrics": two_stage_semantic_metrics,
        "group_prior_alpha": float(group_prior_alpha),
        "rerank_top_groups": int(rerank_top_groups),
        "sample_count": len(split_arrays["rows"]),
    }
