"""Export top PNG validation demos from the medium dual-LoRA checkpoint."""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageSequence

ROOT = Path("/home/rl182/dl/V2L/Project-meme/MultiSticker")
VENDOR_ROOT = Path("/home/rl182/dl/V2L/Project-meme") / ".vendor"
if VENDOR_ROOT.exists() and str(VENDOR_ROOT) not in sys.path:
    sys.path.append(str(VENDOR_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.multisticker import (  # noqa: E402
    IntentGuidedRetriever,
    OpenClipEncoder,
    _extract_missing_stickers,
    _filter_decodable_stickers,
    _normalize_supported_media,
    _resolve_device,
    _seed_everything,
    _set_cache_env,
    default_multisticker_config,
    prepare_manifest,
)
from train_am import apply_lora, encode_image_bank, encode_texts_grad  # noqa: E402


OUT_DIR = ROOT / "Latex_report" / "demo_assets" / "dual_lora_png_demo"
CHECKPOINT = ROOT / "results" / "am_dual_lora_retrieved_topk_rebuild_medium.pt"


def short_text(text: str, limit: int = 220) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


_STICKER_TOKEN_RE = re.compile(r"\[sticker:([^\]]+)\]")


def extract_context_sticker_ids(text: str) -> list[str]:
    seen: list[str] = []
    for match in _STICKER_TOKEN_RE.finditer(str(text)):
        sid = match.group(1).strip()
        if sid and sid not in seen:
            seen.append(sid)
    return seen


def build_config():
    config = default_multisticker_config(str(ROOT))
    config.runtime.device = "cuda"
    config.paths.run_name = "rebuild_medium"
    config.data.max_train_samples = 30000
    config.data.max_val_samples = 5000
    config.data.max_stickers = 8000
    config.data.min_sticker_frequency = 2
    config.data.supported_media = _normalize_supported_media([".png", ".gif", ".webm"])
    config.paths.session_memory_override = (
        "/scratch/rl182/meme/usticker_igsr/llm/"
        "session_memories_qwen32_gptq_v10_png_gif_webm_merged.jsonl"
    )
    config.paths.sample_intent_override = (
        "/scratch/rl182/meme/usticker_igsr/llm/"
        "sample_intents_qwen32_gptq_v10_png_gif_webm_merged.jsonl"
    )
    config.model.epochs = 5
    config.model.train_batch_size = 96
    config.model.infer_batch_size = 256
    config.model.intent_clusters = 64
    return config


def build_sticker_groups(sticker_ids, train_rows, all_rows_by_sticker):
    group_name_to_index = {}
    sticker_group_ids = []
    sticker_group_names = []
    for sid in sticker_ids:
        counts = Counter(str(r.get("intent_label", "neutral_acknowledgment")) for r in train_rows if r["label_id"] == sid)
        if not counts:
            counts = Counter(
                str(r.get("intent_label", "neutral_acknowledgment"))
                for r in all_rows_by_sticker.get(sid, [])
            )
        name = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0] if counts else "neutral_acknowledgment"
        if name not in group_name_to_index:
            group_name_to_index[name] = len(group_name_to_index)
        sticker_group_ids.append(group_name_to_index[name])
        sticker_group_names.append(name)
    return np.asarray(sticker_group_ids, dtype=np.int64), sticker_group_names, group_name_to_index


def copy_sticker(sticker_id: str, source: str, dest_dir: Path, prefix: str) -> str:
    src = Path(source)
    dst = dest_dir / f"{prefix}_{sticker_id}"
    shutil.copy2(src, dst)
    return str(dst)


def preview_image(path: Path, size: int = 128) -> Image.Image:
    suffix = path.suffix.lower()
    try:
        if suffix == ".gif":
            with Image.open(path) as im:
                frame = next(ImageSequence.Iterator(im)).convert("RGB")
        elif suffix == ".png":
            frame = Image.open(path).convert("RGB")
        else:
            frame = Image.new("RGB", (size, size), "white")
            draw = ImageDraw.Draw(frame)
            draw.text((12, 48), suffix.upper().lstrip("."), fill=(60, 60, 60))
            return frame
        frame.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(frame, ((size - frame.width) // 2, (size - frame.height) // 2))
        return canvas
    except Exception:
        canvas = Image.new("RGB", (size, size), "white")
        ImageDraw.Draw(canvas).text((10, 48), "preview\nfailed", fill=(120, 0, 0))
        return canvas


def make_contact_sheet(cases, sticker_paths):
    cell_w, cell_h = 150, 186
    cols = 6
    rows = len(cases)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for r, case in enumerate(cases):
        items = [("Gold", case["gold"])] + [(f"Top {p['rank']}", p["sticker_id"]) for p in case["top_predictions"][:5]]
        for c, (label, sid) in enumerate(items):
            x, y = c * cell_w, r * cell_h
            img = preview_image(Path(sticker_paths[sid]))
            sheet.paste(img, (x + 11, y + 22))
            draw.text((x + 8, y + 6), label, fill=(0, 0, 0), font=font)
            draw.text((x + 8, y + 154), sid[:20], fill=(40, 40, 40), font=font)
        draw.text((8, r * cell_h + cell_h - 16), f"case {r + 1}: rank={case['gold_rank']} intent={case['intent_label']}", fill=(0, 0, 0), font=font)
    sheet.save(OUT_DIR / "dual_lora_top5_png_contact_sheet.png")


def write_case_csv(path: Path, cases):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "sample_id", "domain", "intent_label", "gold", "gold_rank", "gold_score", "top1", "top1_same_group", "context_preview"])
        writer.writeheader()
        for i, case in enumerate(cases, start=1):
            writer.writerow({
                "case": i,
                "sample_id": case["sample_id"],
                "domain": case["domain"],
                "intent_label": case["intent_label"],
                "gold": case["gold"],
                "gold_rank": case["gold_rank"],
                "gold_score": case["gold_score"],
                "top1": case["top_predictions"][0]["sticker_id"],
                "top1_same_group": case["top_predictions"][0]["same_group"],
                "context_preview": case["context_preview"],
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-cases", type=int, default=5, help="Number of validation cases to export.")
    parser.add_argument("--max-per-gold", type=int, default=1, help="Max cases sharing the same gold sticker.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = build_config()
    _set_cache_env(config)
    _seed_everything(config.data.seed)
    device = _resolve_device(config)
    print(f"[export] device={device}")
    manifest = prepare_manifest(config=config, force_rebuild=False)

    sticker_ids = manifest["sticker_ids"]
    sticker_paths = _extract_missing_stickers(config.paths.zip_path, config.paths.sticker_root, sticker_ids)
    sticker_paths, _ = _filter_decodable_stickers(sticker_paths)
    sticker_ids = [s for s in sticker_ids if s in sticker_paths]
    sticker_to_index = {s: i for i, s in enumerate(sticker_ids)}
    for split in ["train", "val", "test"]:
        manifest["splits"][split] = [r for r in manifest["splits"][split] if r["label_id"] in sticker_to_index]
        for row in manifest["splits"][split]:
            row["label_index"] = sticker_to_index[row["label_id"]]

    train_rows = manifest["splits"]["train"]
    val_rows = manifest["splits"]["val"]
    all_rows_by_sticker = defaultdict(list)
    for row in train_rows + val_rows + manifest["splits"]["test"]:
        all_rows_by_sticker[row["label_id"]].append(row)
    sticker_group_ids, sticker_group_names, group_name_to_index = build_sticker_groups(sticker_ids, train_rows, all_rows_by_sticker)
    print(f"[export] sticker_bank={len(sticker_ids)} val={len(val_rows)} groups={len(group_name_to_index)}")

    clip_encoder = OpenClipEncoder(config.model.clip_model_name, config.model.clip_pretrained, device=device)
    apply_lora(clip_encoder.model, "dual_lora", 8, 16, 0.05)
    retriever = IntentGuidedRetriever(
        input_dim=3 * clip_encoder.output_dim,
        output_dim=clip_encoder.output_dim,
        hidden_dim=config.model.hidden_dim,
        num_intents=len(group_name_to_index),
        dropout=config.model.dropout,
        temperature=config.model.temperature,
    ).to(device)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    retriever.load_state_dict(ckpt["retriever"])
    clip_encoder.model.load_state_dict(ckpt["clip_lora"], strict=False)
    retriever.eval()
    clip_encoder.model.eval()
    print(f"[export] loaded checkpoint={CHECKPOINT}")

    image_paths_list = [sticker_paths[sid] for sid in sticker_ids]
    with torch.no_grad():
        bank = encode_image_bank(clip_encoder, image_paths_list, config.model.infer_batch_size, device, with_grad=False)
        ctx = encode_texts_grad(clip_encoder.model, clip_encoder.tokenizer, [r["context_text"] for r in val_rows], device, batch_size=config.model.infer_batch_size)
        mem = encode_texts_grad(clip_encoder.model, clip_encoder.tokenizer, [r["memory_text"] for r in val_rows], device, batch_size=config.model.infer_batch_size)
        intent = encode_texts_grad(clip_encoder.model, clip_encoder.tokenizer, [r["intent_text"] for r in val_rows], device, batch_size=config.model.infer_batch_size)
        chunks = []
        for start in range(0, len(val_rows), 512):
            _, logits, _ = retriever(ctx[start:start + 512], mem[start:start + 512], intent[start:start + 512], bank)
            chunks.append(logits.detach().cpu().numpy().astype(np.float32))
    scores = np.concatenate(chunks, axis=0)
    val_label_idx = np.asarray([r["label_index"] for r in val_rows], dtype=np.int64)

    candidates = []
    for row_idx, row in enumerate(val_rows):
        if not row["label_id"].lower().endswith(".png"):
            continue
        row_scores = scores[row_idx]
        gold_idx = int(val_label_idx[row_idx])
        gold_score = float(row_scores[gold_idx])
        gold_rank = int(np.sum(row_scores > gold_score) + 1)
        top_idx = np.argsort(-row_scores)[:5]
        top_predictions = []
        gold_group = int(sticker_group_ids[gold_idx])
        for rank, pred_idx in enumerate(top_idx, start=1):
            pred_sid = sticker_ids[int(pred_idx)]
            top_predictions.append({
                "rank": rank,
                "sticker_id": pred_sid,
                "score": round(float(row_scores[int(pred_idx)]), 4),
                "same_group": bool(int(sticker_group_ids[int(pred_idx)]) == gold_group),
                "group": sticker_group_names[int(pred_idx)],
            })
        margin = float(row_scores[top_idx[0]] - row_scores[top_idx[1]]) if len(top_idx) > 1 else 0.0
        context_text = str(row.get("context_text", ""))
        context_sticker_ids = extract_context_sticker_ids(context_text)
        candidates.append({
            "row_idx": row_idx,
            "sample_id": row["sample_id"],
            "domain": row.get("domain", ""),
            "intent_label": row.get("intent_label", ""),
            "intent_text": row.get("intent_text", ""),
            "context_text": context_text,
            "context_preview": short_text(context_text),
            "memory_text": str(row.get("memory_text", "")),
            "memory_preview": short_text(row.get("memory_text", "")),
            "context_sticker_ids": context_sticker_ids,
            "gold": row["label_id"],
            "gold_rank": gold_rank,
            "gold_score": round(gold_score, 4),
            "top_margin": round(margin, 4),
            "top_predictions": top_predictions,
        })

    ranked = sorted(candidates, key=lambda x: (x["gold_rank"], -x["gold_score"], -x["top_margin"]))
    max_per_gold = max(1, int(args.max_per_gold))
    gold_counts: Counter = Counter()
    selected = []
    for case in ranked:
        if gold_counts[case["gold"]] >= max_per_gold:
            continue
        selected.append(case)
        gold_counts[case["gold"]] += 1
        if len(selected) >= args.num_cases:
            break

    for i, case in enumerate(selected, start=1):
        case_dir = OUT_DIR / f"case_{i:02d}_rank_{case['gold_rank']}"
        case_dir.mkdir(parents=True, exist_ok=True)
        copy_sticker(case["gold"], sticker_paths[case["gold"]], case_dir, "gold")
        for pred in case["top_predictions"]:
            copy_sticker(pred["sticker_id"], sticker_paths[pred["sticker_id"]], case_dir, f"top{pred['rank']}")
        ctx_dir = case_dir / "context_stickers"
        ctx_dir.mkdir(exist_ok=True)
        copied_ctx = []
        for sid in case["context_sticker_ids"]:
            src = sticker_paths.get(sid)
            if not src:
                continue
            dst = ctx_dir / sid
            try:
                shutil.copy2(src, dst)
                copied_ctx.append(sid)
            except FileNotFoundError:
                continue
        case["context_sticker_files"] = copied_ctx
        with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(case, f, indent=2, ensure_ascii=False)

    json_path = OUT_DIR / f"top{args.num_cases}_png_cases.json"
    csv_path = OUT_DIR / f"top{args.num_cases}_png_cases.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    write_case_csv(csv_path, selected)
    if args.num_cases != 5:
        with open(OUT_DIR / "top5_png_cases.json", "w", encoding="utf-8") as f:
            json.dump(selected[:5], f, indent=2, ensure_ascii=False)
        write_case_csv(OUT_DIR / "top5_png_cases.csv", selected[:5])
    make_contact_sheet(selected, sticker_paths)
    print(f"[export] wrote {OUT_DIR}")
    for i, case in enumerate(selected, start=1):
        print(f"[export] case={i} rank={case['gold_rank']} gold={case['gold']} sample={case['sample_id']}")


if __name__ == "__main__":
    main()
