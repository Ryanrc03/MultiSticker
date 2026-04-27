"""Collect logs and result files for one MultiSticker experiment run.

The runner writes logs and model/results into separate top-level directories.
This helper copies the files for a single artifact prefix into a parameterized
archive folder and writes a manifest so later analysis can find the run inputs.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-prefix", required=True)
    p.add_argument("--log-file", required=True)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--archive-root", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--run-scale", required=True)
    p.add_argument("--experiment-kind", required=True)
    p.add_argument("--mode", default="")
    p.add_argument("--memory-strategy", default="")
    p.add_argument("--query-mode", default="")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-stickers", type=int, default=0)
    p.add_argument("--min-sticker-frequency", type=int, default=1)
    p.add_argument("--supported-media", default="")
    p.add_argument("--train-batch-size", type=int, default=0)
    p.add_argument("--infer-batch-size", type=int, default=0)
    p.add_argument("--intent-clusters", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--log-samples", type=int, default=0)
    p.add_argument("--session-memories-file", default="")
    p.add_argument("--sample-intents-file", default="")
    p.add_argument("--lora-r", type=int, default=0)
    p.add_argument("--lora-alpha", type=int, default=0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-lr", type=float, default=0.0)
    p.add_argument("--head-lr", type=float, default=0.0)
    p.add_argument("--copy", action="store_true", help="Copy files instead of the default copy2 metadata-preserving behavior.")
    return p.parse_args()


def slug(value: str) -> str:
    value = value.strip() or "none"
    value = re.sub(r"[^A-Za-z0-9._=-]+", "-", value)
    return value.strip("-") or "none"


def archive_bucket(args: argparse.Namespace) -> Path:
    data_label = (
        f"scale-{args.run_scale}"
        f"__epochs-{args.epochs}"
        f"__train-{args.max_train_samples}"
        f"__val-{args.max_val_samples}"
        f"__stickers-{args.max_stickers}"
        f"__minfreq-{args.min_sticker_frequency}"
    )
    media_label = args.supported_media.replace(",", "-").replace(".", "")
    if media_label:
        data_label += f"__media-{media_label}"
    return Path(args.archive_root) / slug(args.run_name) / slug(data_label) / slug(args.artifact_prefix)


def collect_sources(args: argparse.Namespace) -> list[Path]:
    sources: list[Path] = []
    log_file = Path(args.log_file)
    if log_file.exists():
        sources.append(log_file)

    result_dir = Path(args.result_dir)
    if result_dir.exists():
        for item in sorted(result_dir.glob(f"{args.artifact_prefix}*")):
            if item.is_file():
                sources.append(item)

    log_dir = log_file.parent
    if log_dir.exists():
        for item in sorted(log_dir.glob(f"{args.artifact_prefix}*")):
            if item.is_file() and item not in sources:
                sources.append(item)

    return sources


def main() -> None:
    args = parse_args()
    destination = archive_bucket(args)
    destination.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in collect_sources(args):
        dst = destination / src.name
        if args.copy:
            shutil.copy(src, dst)
        else:
            shutil.copy2(src, dst)
        copied.append({"source": str(src), "archive_path": str(dst), "bytes": dst.stat().st_size})

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_prefix": args.artifact_prefix,
        "run": {
            "name": args.run_name,
            "scale": args.run_scale,
            "experiment_kind": args.experiment_kind,
            "mode": args.mode,
            "memory_strategy": args.memory_strategy,
            "query_mode": args.query_mode,
        },
        "data": {
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_stickers": args.max_stickers,
            "min_sticker_frequency": args.min_sticker_frequency,
            "supported_media": [item for item in args.supported_media.split(",") if item],
            "session_memories_file": args.session_memories_file,
            "sample_intents_file": args.sample_intents_file,
        },
        "training": {
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "infer_batch_size": args.infer_batch_size,
            "intent_clusters": args.intent_clusters,
            "num_workers": args.num_workers,
            "log_every": args.log_every,
            "log_samples": args.log_samples,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "lora_lr": args.lora_lr,
            "head_lr": args.head_lr,
        },
        "files": copied,
    }
    manifest_path = destination / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"[archive] {args.artifact_prefix}: {len(copied)} files -> {destination}", flush=True)


if __name__ == "__main__":
    main()
