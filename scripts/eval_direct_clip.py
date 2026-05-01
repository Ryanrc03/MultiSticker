"""Zero-shot CLIP baseline for U-Sticker retrieval.

Scores val queries directly against the frozen sticker image bank using
dot-product similarity. No training, no trainable head.

Query modes:
  clip_context               : context_text only (recommended clean baseline)
  clip_context_intent        : context_text + intent_text
  clip_context_memory_intent : context_text + memory_text + intent_text
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MULTISTICKER_ROOT = Path("/home/rl182/dl/V2L/Project-meme/MultiSticker")
SCRATCH_ARTIFACT_ROOT = Path("/scratch/rl182/mutlisticker")
VENDOR_ROOT = Path("/home/rl182/dl/V2L/Project-meme") / ".vendor"
if VENDOR_ROOT.exists() and str(VENDOR_ROOT) not in sys.path:
    sys.path.append(str(VENDOR_ROOT))
if str(MULTISTICKER_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTISTICKER_ROOT))

from src.multisticker import (  # noqa: E402
    OpenClipEncoder,
    _extract_missing_stickers,
    _filter_decodable_stickers,
    _group_metrics_from_scores,
    _metrics_from_scores,
    _normalize_supported_media,
    _resolve_device,
    _seed_everything,
    _set_cache_env,
    default_multisticker_config,
    prepare_manifest,
)
from src.utils import save_json  # noqa: E402
from embedding_cache import (
    cache_path,
    describe_cache,
    digest_rows,
    digest_stickers,
    load_npz,
    save_npz,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--max-stickers", type=int, default=0)
    p.add_argument("--min-sticker-frequency", type=int, default=1)
    p.add_argument("--max-train-samples", type=int, default=100000)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--supported-media", type=str, default=".png,.gif,.webm")
    p.add_argument("--session-memories-file", type=str, default="")
    p.add_argument("--sample-intents-file", type=str, default="")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument(
        "--query-mode",
        choices=["clip_context", "clip_context_intent", "clip_context_memory_intent"],
        default="clip_context",
    )
    p.add_argument("--infer-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--results-dir", type=str, default=str(SCRATCH_ARTIFACT_ROOT / "results")
    )
    p.add_argument(
        "--embedding-cache-dir",
        type=str,
        default=str(SCRATCH_ARTIFACT_ROOT / "embedding_cache"),
        help="Directory for reusable frozen CLIP embeddings.",
    )
    p.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help="Disable disk cache for frozen direct CLIP embeddings.",
    )
    return p.parse_args()


def build_config(args):
    config = default_multisticker_config(str(MULTISTICKER_ROOT))
    config.runtime.device = args.device
    config.data.max_files = args.max_files
    config.data.max_stickers = args.max_stickers
    config.data.min_sticker_frequency = args.min_sticker_frequency
    config.data.max_train_samples = args.max_train_samples
    config.data.max_val_samples = args.max_val_samples
    config.data.supported_media = _normalize_supported_media(
        args.supported_media.split(",")
    )
    config.data.seed = args.seed
    config.model.infer_batch_size = args.infer_batch_size
    # direct_clip never needs memory retrieval; use disabled to skip expensive encoding
    # when rebuilding the manifest. If reusing an existing manifest (built with retrieved_topk),
    # the stored memory_text values are ignored for clip_context query mode anyway.
    config.data.memory_strategy = "disabled"
    if args.session_memories_file:
        config.paths.session_memory_override = args.session_memories_file
    if args.sample_intents_file:
        config.paths.sample_intent_override = args.sample_intents_file
    # Share manifest with train_am.py runs under the same run_name.
    config.paths.run_name = args.run_name
    return config


def _build_sticker_groups(sticker_ids, train_rows, all_rows_by_sticker):
    group_name_to_index = {}
    sticker_group_ids_list = []
    for sid in sticker_ids:
        c = Counter(
            str(r.get("intent_label", "neutral_acknowledgment"))
            for r in train_rows
            if r["label_id"] == sid
        )
        if c:
            name = sorted(c.items(), key=lambda x: (-x[1], x[0]))[0][0]
        else:
            c2 = Counter(
                str(r.get("intent_label", "neutral_acknowledgment"))
                for r in all_rows_by_sticker.get(sid, [])
            )
            name = (
                sorted(c2.items(), key=lambda x: (-x[1], x[0]))[0][0]
                if c2
                else "neutral_acknowledgment"
            )
        if name not in group_name_to_index:
            group_name_to_index[name] = len(group_name_to_index)
        sticker_group_ids_list.append(group_name_to_index[name])
    return np.asarray(sticker_group_ids_list, dtype=np.int64)


def main():
    args = parse_args()
    config = build_config(args)
    _set_cache_env(config)
    _seed_everything(args.seed)
    device = _resolve_device(config)
    embedding_cache_dir = (
        Path(args.embedding_cache_dir)
        if args.embedding_cache_dir
        else SCRATCH_ARTIFACT_ROOT / "embedding_cache"
    )
    use_embedding_cache = not args.no_embedding_cache
    if use_embedding_cache:
        embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[direct_clip] embedding_cache={embedding_cache_dir}", flush=True)
    print(
        f"[direct_clip] query_mode={args.query_mode} run={args.run_name} device={device}",
        flush=True,
    )

    manifest = prepare_manifest(config=config, force_rebuild=args.force_rebuild)

    sticker_ids = manifest["sticker_ids"]
    sticker_paths = _extract_missing_stickers(
        config.paths.zip_path, config.paths.sticker_root, sticker_ids
    )
    sticker_paths, _ = _filter_decodable_stickers(sticker_paths)
    sticker_ids = [s for s in sticker_ids if s in sticker_paths]
    sticker_to_index = {s: i for i, s in enumerate(sticker_ids)}
    for split in ["train", "val", "test"]:
        manifest["splits"][split] = [
            r for r in manifest["splits"][split] if r["label_id"] in sticker_to_index
        ]
        for r in manifest["splits"][split]:
            r["label_index"] = sticker_to_index[r["label_id"]]
    train_rows = manifest["splits"]["train"]
    val_rows = manifest["splits"]["val"]
    print(
        f"[direct_clip] sticker_bank={len(sticker_ids)} val={len(val_rows)}", flush=True
    )

    clip_encoder = OpenClipEncoder(
        config.model.clip_model_name, config.model.clip_pretrained, device=device
    )

    all_rows_by_sticker: dict = defaultdict(list)
    for r in train_rows + val_rows + manifest["splits"]["test"]:
        all_rows_by_sticker[r["label_id"]].append(r)
    sticker_group_ids = _build_sticker_groups(
        sticker_ids, train_rows, all_rows_by_sticker
    )

    image_paths_list = [sticker_paths[s] for s in sticker_ids]
    image_cache = None
    if use_embedding_cache:
        image_cache = cache_path(
            embedding_cache_dir,
            "frozen_image_bank",
            {
                "clip_model_name": config.model.clip_model_name,
                "clip_pretrained": config.model.clip_pretrained,
                "stickers_digest": digest_stickers(sticker_ids, image_paths_list),
            },
        )
        cached = load_npz(image_cache)
    else:
        cached = None
    if cached is not None:
        print(
            f"[direct_clip] loaded sticker image bank from {describe_cache(image_cache)}",
            flush=True,
        )
        img_np = cached["image"]
    else:
        print("[direct_clip] encoding sticker image bank", flush=True)
        img_np = clip_encoder.encode_images(
            image_paths_list, batch_size=config.model.infer_batch_size
        )
        if image_cache is not None:
            save_npz(image_cache, image=img_np)
            print(
                f"[direct_clip] saved sticker image bank to {describe_cache(image_cache)}",
                flush=True,
            )

    def query_text(row: dict) -> str:
        if args.query_mode == "clip_context":
            return row["context_text"]
        elif args.query_mode == "clip_context_intent":
            return row["context_text"] + " " + row.get("intent_text", "")
        else:
            return (
                row["context_text"]
                + " "
                + row.get("memory_text", "")
                + " "
                + row.get("intent_text", "")
            )

    query_texts = [query_text(r) for r in val_rows]
    query_cache = None
    if use_embedding_cache:
        query_cache = cache_path(
            embedding_cache_dir,
            "direct_clip_query",
            {
                "clip_model_name": config.model.clip_model_name,
                "clip_pretrained": config.model.clip_pretrained,
                "query_mode": args.query_mode,
                "rows_digest": digest_rows(
                    val_rows, ["context_text", "memory_text", "intent_text"]
                ),
            },
        )
        cached = load_npz(query_cache)
    else:
        cached = None
    if cached is not None:
        print(
            f"[direct_clip] loaded {len(val_rows)} val query embeddings from {describe_cache(query_cache)}",
            flush=True,
        )
        query_np = cached["query"]
    else:
        print(f"[direct_clip] encoding {len(val_rows)} val queries", flush=True)
        query_np = clip_encoder.encode_texts(
            query_texts, batch_size=config.model.infer_batch_size
        )
        if query_cache is not None:
            save_npz(query_cache, query=query_np)
            print(
                f"[direct_clip] saved val query embeddings to {describe_cache(query_cache)}",
                flush=True,
            )
    val_label_idx = np.asarray([r["label_index"] for r in val_rows], dtype=np.int64)

    # Pure cosine similarity (both sides already L2-normalised by encoders)
    score_matrix = query_np @ img_np.T
    metrics = _metrics_from_scores(score_matrix, val_label_idx)
    semantic = _group_metrics_from_scores(
        score_matrix, val_label_idx, sticker_group_ids
    )

    results = {
        "mode": "direct_clip",
        "query_mode": args.query_mode,
        "run_name": args.run_name,
        "config": {
            "data": {
                "max_stickers": args.max_stickers,
                "min_sticker_frequency": args.min_sticker_frequency,
                "max_val_samples": args.max_val_samples,
                "supported_media": list(config.data.supported_media),
            },
            "model": {
                "clip_model_name": config.model.clip_model_name,
                "clip_pretrained": config.model.clip_pretrained,
            },
        },
        "dataset_summary": manifest["dataset_summary"],
        "media_summary": {
            "sticker_bank_size": len(sticker_ids),
            "supported_media": list(config.data.supported_media),
        },
        "val": {
            "metrics": metrics,
            "semantic_metrics": semantic,
            "sample_count": len(val_rows),
        },
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"direct_clip_{args.query_mode}_{args.run_name}"
    save_json(results, str(results_dir / f"{out_name}.json"))
    print(f"[direct_clip] DONE {out_name}", flush=True)
    print(json.dumps(metrics, indent=2))
    print("[direct_clip] group metrics:", json.dumps(semantic, indent=2))


if __name__ == "__main__":
    main()
