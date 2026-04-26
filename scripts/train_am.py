"""LoRA-capable retrieval training for U-Sticker IGSR.

Modes: head_only | image_lora | text_lora | dual_lora.
Memory strategies: retrieved_topk | recent_topk | disabled.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MULTISTICKER_ROOT = Path("/home/rl182/dl/V2L/Project-meme/MultiSticker")
VENDOR_ROOT = Path("/home/rl182/dl/V2L/Project-meme") / ".vendor"
if VENDOR_ROOT.exists() and str(VENDOR_ROOT) not in sys.path:
    sys.path.append(str(VENDOR_ROOT))
if str(MULTISTICKER_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTISTICKER_ROOT))

from src.usticker_igsr import (  # noqa: E402
    IntentGuidedRetriever,
    MeanPoolingEncoder,
    MULTI_FRAME_MEDIA,
    OpenClipEncoder,
    _extract_missing_stickers,
    _filter_decodable_stickers,
    _fuse_group_prior_scores,
    _group_metrics_from_scores,
    _load_sticker_frames,
    _load_sticker_image,
    _metrics_from_scores,
    _normalize_supported_media,
    _resolve_device,
    _seed_everything,
    _set_cache_env,
    _two_stage_group_rerank_scores,
    default_usticker_config,
    prepare_manifest,
)
from src.utils import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tuning-mode", choices=["head_only", "image_lora", "text_lora", "dual_lora"], required=True)
    p.add_argument("--memory-strategy", choices=["retrieved_topk", "recent_topk", "disabled"], default="retrieved_topk")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--max-stickers", type=int, default=0)
    p.add_argument("--min-sticker-frequency", type=int, default=1)
    p.add_argument("--max-train-samples", type=int, default=100000)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--supported-media", type=str, default=".png,.gif,.webm")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--infer-batch-size", type=int, default=64)
    p.add_argument("--intent-clusters", type=int, default=128)
    p.add_argument("--session-memories-file", type=str, default="")
    p.add_argument("--sample-intents-file", type=str, default="")
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-reencode-every-epoch", action="store_true", default=True)
    return p.parse_args()


def build_config(args):
    config = default_usticker_config(str(MULTISTICKER_ROOT))
    config.runtime.device = args.device
    config.data.max_files = args.max_files
    config.data.max_stickers = args.max_stickers
    config.data.min_sticker_frequency = args.min_sticker_frequency
    config.data.max_train_samples = args.max_train_samples
    config.data.max_val_samples = args.max_val_samples
    config.data.max_test_samples = args.max_test_samples
    config.data.supported_media = _normalize_supported_media(args.supported_media.split(","))
    config.data.seed = args.seed
    config.data.memory_strategy = args.memory_strategy
    config.model.epochs = args.epochs
    config.model.train_batch_size = args.train_batch_size
    config.model.infer_batch_size = args.infer_batch_size
    config.model.intent_clusters = args.intent_clusters
    if args.session_memories_file:
        config.paths.session_memory_override = args.session_memories_file
    if args.sample_intents_file:
        config.paths.sample_intent_override = args.sample_intents_file
    # Use base run_name for manifest path so all modes share the same manifest.
    config.paths.run_name = args.run_name
    return config


def apply_lora(model: nn.Module, mode: str, r: int, alpha: int, dropout: float) -> nn.Module:
    from peft import LoraConfig, inject_adapter_in_model

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["out_proj", "c_fc", "c_proj"],
        bias="none",
    )
    inject_adapter_in_model(lora_cfg, model)
    for name, p in model.named_parameters():
        if "lora_" not in name:
            p.requires_grad = False
            continue
        if mode == "image_lora" and not name.startswith("visual."):
            p.requires_grad = False
        elif mode == "text_lora" and name.startswith("visual."):
            p.requires_grad = False
    return model


def encode_texts_grad(clip_model, tokenizer, texts, device, batch_size=64):
    outs = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(list(texts[start:start + batch_size])).to(device)
        enc = clip_model.encode_text(tokens)
        enc = F.normalize(enc, dim=-1)
        outs.append(enc)
    return torch.cat(outs, dim=0) if outs else torch.zeros((0, clip_model.text_projection.shape[1]), device=device)


def encode_image_bank(clip_encoder: OpenClipEncoder, image_paths, batch_size: int, device, with_grad: bool, max_frames: int = 0):
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    outputs = torch.zeros((len(image_paths), clip_encoder.output_dim), dtype=torch.float32, device=device)
    static_idx = [i for i, p in enumerate(image_paths) if Path(p).suffix.lower() not in MULTI_FRAME_MEDIA]
    animated_idx = [i for i, p in enumerate(image_paths) if Path(p).suffix.lower() in MULTI_FRAME_MEDIA]
    with ctx:
        for start in range(0, len(static_idx), batch_size):
            idxs = static_idx[start:start + batch_size]
            imgs = [clip_encoder.preprocess(_load_sticker_image(image_paths[i])) for i in idxs]
            batch = torch.stack(imgs, dim=0).to(device)
            enc = clip_encoder.model.encode_image(batch)
            enc = F.normalize(enc, dim=-1)
            for j, idx in enumerate(idxs):
                outputs[idx] = enc[j]
        for idx in animated_idx:
            frames = _load_sticker_frames(image_paths[idx], all_frames=True)
            if max_frames and len(frames) > max_frames:
                step = max(1, len(frames) // max_frames)
                frames = frames[::step][:max_frames]
            frame_feats = []
            fb = max(1, min(batch_size, 8 if with_grad else 32))
            for s in range(0, len(frames), fb):
                batch = torch.stack([clip_encoder.preprocess(fr) for fr in frames[s:s + fb]], dim=0).to(device)
                enc = clip_encoder.model.encode_image(batch)
                enc = F.normalize(enc, dim=-1)
                frame_feats.append(enc)
            pooled = torch.cat(frame_feats, dim=0).mean(dim=0)
            pooled = F.normalize(pooled.unsqueeze(0), dim=-1).squeeze(0)
            outputs[idx] = pooled
    return outputs


def per_media_metrics(score_matrix, label_indices, rows, sticker_ids, sticker_group_ids):
    out = {}
    for suffix in (".png", ".gif", ".webm"):
        mask = np.array([Path(r["label_id"]).suffix.lower() == suffix for r in rows], dtype=bool)
        if not mask.any():
            out[suffix[1:]] = {"count": 0}
            continue
        sub_scores = score_matrix[mask]
        sub_labels = label_indices[mask]
        exact = _metrics_from_scores(sub_scores, sub_labels)
        group = _group_metrics_from_scores(sub_scores, sub_labels, sticker_group_ids)
        out[suffix[1:]] = {"count": int(mask.sum()), "exact": exact, "group": group}
    return out


def main():
    args = parse_args()
    config = build_config(args)
    _set_cache_env(config)
    _seed_everything(config.data.seed)
    device = _resolve_device(config)
    print(f"[am] mode={args.tuning_mode} memory={args.memory_strategy} run={args.run_name} device={device}", flush=True)

    manifest = prepare_manifest(config=config, force_rebuild=args.force_rebuild)

    sticker_ids = manifest["sticker_ids"]
    sticker_paths = _extract_missing_stickers(config.paths.zip_path, config.paths.sticker_root, sticker_ids)
    sticker_paths, _ = _filter_decodable_stickers(sticker_paths)
    sticker_ids = [s for s in sticker_ids if s in sticker_paths]
    sticker_to_index = {s: i for i, s in enumerate(sticker_ids)}
    for split in ["train", "val", "test"]:
        manifest["splits"][split] = [r for r in manifest["splits"][split] if r["label_id"] in sticker_to_index]
        for r in manifest["splits"][split]:
            r["label_index"] = sticker_to_index[r["label_id"]]
    print(f"[am] sticker_bank={len(sticker_ids)} train={len(manifest['splits']['train'])} val={len(manifest['splits']['val'])}", flush=True)

    clip_encoder = OpenClipEncoder(config.model.clip_model_name, config.model.clip_pretrained, device=device)
    use_image_lora = args.tuning_mode in {"image_lora", "dual_lora"}
    use_text_lora = args.tuning_mode in {"text_lora", "dual_lora"}
    if use_image_lora or use_text_lora:
        apply_lora(clip_encoder.model, args.tuning_mode, args.lora_r, args.lora_alpha, args.lora_dropout)
        trainable = [n for n, p in clip_encoder.model.named_parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in clip_encoder.model.parameters() if p.requires_grad)
        print(f"[am] lora injected; trainable clip params={n_trainable} sample_names={trainable[:3]}", flush=True)

    from collections import Counter
    all_rows_by_sticker = defaultdict(list)
    for r in manifest["splits"]["train"] + manifest["splits"]["val"] + manifest["splits"]["test"]:
        all_rows_by_sticker[r["label_id"]].append(r)
    train_rows = manifest["splits"]["train"]
    group_name_to_index = {}
    sticker_group_ids_list = []
    for sid in sticker_ids:
        c = Counter(str(r.get("intent_label", "neutral_acknowledgment")) for r in train_rows if r["label_id"] == sid)
        if c:
            name = sorted(c.items(), key=lambda x: (-x[1], x[0]))[0][0]
        else:
            c2 = Counter(str(r.get("intent_label", "neutral_acknowledgment")) for r in all_rows_by_sticker.get(sid, []))
            name = sorted(c2.items(), key=lambda x: (-x[1], x[0]))[0][0] if c2 else "neutral_acknowledgment"
        if name not in group_name_to_index:
            group_name_to_index[name] = len(group_name_to_index)
        sticker_group_ids_list.append(group_name_to_index[name])
    sticker_group_ids = np.asarray(sticker_group_ids_list, dtype=np.int64)
    intent_labels = {sid: int(sticker_group_ids[i]) for i, sid in enumerate(sticker_ids)}
    n_clusters = max(1, len(group_name_to_index))

    val_rows = manifest["splits"]["val"]
    all_rows = train_rows + val_rows
    train_n = len(train_rows)

    def encode_texts_frozen(texts):
        return clip_encoder.encode_texts(texts, batch_size=config.model.infer_batch_size)

    if not use_text_lora:
        print("[am] precomputing frozen text embeddings", flush=True)
        ctx_np = encode_texts_frozen([r["context_text"] for r in all_rows])
        mem_np = encode_texts_frozen([r["memory_text"] for r in all_rows])
        int_np = encode_texts_frozen([r["intent_text"] for r in all_rows])
        text_dim = ctx_np.shape[1]
    else:
        text_dim = clip_encoder.output_dim
        ctx_np = mem_np = int_np = None

    image_paths_list = [sticker_paths[s] for s in sticker_ids]
    if not use_image_lora:
        print("[am] precomputing frozen image bank", flush=True)
        img_np = clip_encoder.encode_images(image_paths_list, batch_size=config.model.infer_batch_size)
        image_bank = torch.from_numpy(img_np).to(device)
    else:
        image_bank = None

    retriever = IntentGuidedRetriever(
        input_dim=3 * clip_encoder.output_dim,
        output_dim=clip_encoder.output_dim,
        hidden_dim=config.model.hidden_dim,
        num_intents=n_clusters,
        dropout=config.model.dropout,
        temperature=config.model.temperature,
    ).to(device)

    head_params = list(retriever.parameters())
    lora_params = [p for p in clip_encoder.model.parameters() if p.requires_grad]
    param_groups = [{"params": head_params, "lr": args.head_lr, "weight_decay": config.model.weight_decay}]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups)

    train_label_idx = np.asarray([r["label_index"] for r in train_rows], dtype=np.int64)
    train_intent_idx = np.asarray([intent_labels[r["label_id"]] for r in train_rows], dtype=np.int64)
    val_label_idx = np.asarray([r["label_index"] for r in val_rows], dtype=np.int64)

    rng = np.random.default_rng(config.data.seed)
    history = []
    best_state = None
    best_score = -1.0

    def evaluate(precomputed_bank=None):
        retriever.eval()
        clip_encoder.model.eval()
        with torch.no_grad():
            if use_image_lora:
                if precomputed_bank is not None:
                    bank = precomputed_bank
                else:
                    bank = encode_image_bank(clip_encoder, image_paths_list, config.model.infer_batch_size, device, with_grad=False)
            else:
                bank = image_bank
            if use_text_lora:
                ctx = encode_texts_grad(clip_encoder.model, clip_encoder.tokenizer, [r["context_text"] for r in val_rows], device, batch_size=config.model.infer_batch_size)
                mem = encode_texts_grad(clip_encoder.model, clip_encoder.tokenizer, [r["memory_text"] for r in val_rows], device, batch_size=config.model.infer_batch_size)
                ints = encode_texts_grad(clip_encoder.model, clip_encoder.tokenizer, [r["intent_text"] for r in val_rows], device, batch_size=config.model.infer_batch_size)
            else:
                ctx = torch.from_numpy(ctx_np[train_n:train_n + len(val_rows)]).to(device)
                mem = torch.from_numpy(mem_np[train_n:train_n + len(val_rows)]).to(device)
                ints = torch.from_numpy(int_np[train_n:train_n + len(val_rows)]).to(device)
            score_chunks = []
            logit_chunks = []
            for s in range(0, len(val_rows), 512):
                _, rl, il = retriever(ctx[s:s + 512], mem[s:s + 512], ints[s:s + 512], bank)
                score_chunks.append(rl.cpu().numpy().astype(np.float32))
                logit_chunks.append(il.cpu().numpy().astype(np.float32))
        score_matrix = np.concatenate(score_chunks, axis=0)
        logit_matrix = np.concatenate(logit_chunks, axis=0)
        metrics = _metrics_from_scores(score_matrix, val_label_idx)
        semantic = _group_metrics_from_scores(score_matrix, val_label_idx, sticker_group_ids)
        fused = _fuse_group_prior_scores(score_matrix, logit_matrix, sticker_group_ids, alpha=config.model.group_prior_alpha)
        fused_metrics = _metrics_from_scores(fused, val_label_idx)
        fused_sem = _group_metrics_from_scores(fused, val_label_idx, sticker_group_ids)
        ts = _two_stage_group_rerank_scores(score_matrix, logit_matrix, sticker_group_ids, top_groups=config.model.rerank_top_groups, alpha=config.model.group_prior_alpha)
        ts_metrics = _metrics_from_scores(ts, val_label_idx)
        ts_sem = _group_metrics_from_scores(ts, val_label_idx, sticker_group_ids)
        media_bd = per_media_metrics(score_matrix, val_label_idx, val_rows, sticker_ids, sticker_group_ids)
        return {
            "metrics": metrics,
            "semantic_metrics": semantic,
            "fused_metrics": fused_metrics,
            "fused_semantic_metrics": fused_sem,
            "two_stage_metrics": ts_metrics,
            "two_stage_semantic_metrics": ts_sem,
            "per_media": media_bd,
            "sample_count": len(val_rows),
        }, bank

    results_dir = MULTISTICKER_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"am_{args.tuning_mode}_{args.memory_strategy}_{args.run_name}"

    for epoch in range(1, args.epochs + 1):
        if use_image_lora:
            print(f"[am] epoch {epoch}: re-encoding image bank (no-grad bank; positives re-encoded per batch)", flush=True)
            with torch.no_grad():
                bank_nog = encode_image_bank(clip_encoder, image_paths_list, config.model.infer_batch_size, device, with_grad=False).detach()
            image_bank = bank_nog
            torch.cuda.empty_cache()

        retriever.train()
        if use_image_lora or use_text_lora:
            clip_encoder.model.train()

        order = rng.permutation(train_n)
        losses = []
        B = config.model.train_batch_size
        for step_start in range(0, train_n, B):
            bi = order[step_start:step_start + B]
            rows_batch = [train_rows[int(i)] for i in bi]
            labels = torch.from_numpy(train_label_idx[bi]).to(device)
            intents = torch.from_numpy(train_intent_idx[bi]).to(device)

            if use_text_lora:
                ctx = encode_texts_grad(clip_encoder.model, clip_encoder.tokenizer, [r["context_text"] for r in rows_batch], device, batch_size=len(rows_batch))
                mem = encode_texts_grad(clip_encoder.model, clip_encoder.tokenizer, [r["memory_text"] for r in rows_batch], device, batch_size=len(rows_batch))
                ints = encode_texts_grad(clip_encoder.model, clip_encoder.tokenizer, [r["intent_text"] for r in rows_batch], device, batch_size=len(rows_batch))
            else:
                ctx = torch.from_numpy(ctx_np[bi]).to(device)
                mem = torch.from_numpy(mem_np[bi]).to(device)
                ints = torch.from_numpy(int_np[bi]).to(device)

            bank = image_bank
            if use_image_lora:
                uniq_label_list = list(dict.fromkeys(int(x) for x in labels.tolist()))
                pos_sticker_paths = [image_paths_list[i] for i in uniq_label_list]
                pos_feats = encode_image_bank(clip_encoder, pos_sticker_paths, max(1, min(16, len(pos_sticker_paths))), device, with_grad=True, max_frames=4)
                bank_live = image_bank.clone()
                uniq_label_tensor = torch.tensor(uniq_label_list, dtype=torch.long, device=device)
                bank_live[uniq_label_tensor] = pos_feats
                bank = bank_live

            _, rl, il = retriever(ctx, mem, ints, bank)
            retrieval_loss = F.cross_entropy(rl, labels)
            intent_loss = F.cross_entropy(il, intents)
            loss = retrieval_loss + config.model.intent_loss_weight * intent_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            if step_start // B % 50 == 0:
                print(f"[am] epoch {epoch} step {step_start//B} loss={loss.item():.4f}", flush=True)

        val_result, _ = evaluate(precomputed_bank=image_bank if use_image_lora else None)
        history.append({
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 4),
            "val_p@1": val_result["metrics"]["p@1"],
            "val_p@5": val_result["metrics"]["p@5"],
            "val_p@30": val_result["metrics"]["p@30"],
            "val_group_p@30": val_result["semantic_metrics"]["p@30"],
            "val_two_stage_group_p@30": val_result["two_stage_semantic_metrics"]["p@30"],
        })
        print(f"[am] epoch {epoch} summary {history[-1]}", flush=True)
        score = val_result["two_stage_semantic_metrics"]["p@30"]
        if score > best_score:
            best_score = score
            best_state = {
                "retriever": {k: v.detach().cpu().clone() for k, v in retriever.state_dict().items()},
                "clip_lora": {k: v.detach().cpu().clone() for k, v in clip_encoder.model.state_dict().items() if "lora_" in k},
            }
        ckpt_results = {
            "mode": args.tuning_mode,
            "memory_strategy": args.memory_strategy,
            "run_name": args.run_name,
            "config": {
                "data": asdict(config.data),
                "model": asdict(config.model),
                "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                         "lora_lr": args.lora_lr, "head_lr": args.head_lr},
            },
            "dataset_summary": manifest["dataset_summary"],
            "media_summary": {"sticker_bank_size": len(sticker_ids), "supported_media": list(config.data.supported_media)},
            "training_history": history,
            "val": val_result,
            "best_val_two_stage_group_p@30": best_score,
            "partial": epoch < args.epochs,
        }
        save_json(ckpt_results, str(results_dir / f"{out_name}.json"))
        if best_state is not None:
            torch.save(best_state, str(results_dir / f"{out_name}.pt"))
        print(f"[am] epoch {epoch} checkpoint saved (best_score={best_score:.4f})", flush=True)

    final_val, _ = evaluate()

    results = {
        "mode": args.tuning_mode,
        "memory_strategy": args.memory_strategy,
        "run_name": args.run_name,
        "config": {
            "data": asdict(config.data),
            "model": asdict(config.model),
            "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                     "lora_lr": args.lora_lr, "head_lr": args.head_lr},
        },
        "dataset_summary": manifest["dataset_summary"],
        "media_summary": {"sticker_bank_size": len(sticker_ids), "supported_media": list(config.data.supported_media)},
        "training_history": history,
        "val": final_val,
        "best_val_two_stage_group_p@30": best_score,
    }
    save_json(results, str(results_dir / f"{out_name}.json"))
    if best_state is not None:
        torch.save(best_state, str(results_dir / f"{out_name}.pt"))
    print(f"[am] DONE {out_name} best_score={best_score:.4f}", flush=True)
    print(json.dumps(results["val"]["metrics"], indent=2))


if __name__ == "__main__":
    main()
