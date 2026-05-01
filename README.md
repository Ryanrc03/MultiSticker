# MultiSticker: Intent-Guided Multimodal Sticker Retrieval with Dialogue Memory

MultiSticker is a COMP 646 final-project codebase for retrieving chat stickers from a large U-Sticker candidate bank. The system combines three text signals for each sticker-reply turn:

1. recent dialogue context,
2. retrieved long-term dialogue memory from previous sessions, and
3. an LLM-generated reply intent.

These signals are encoded with OpenCLIP and projected into the sticker image space by a trainable intent-guided retriever. The code supports static `.png` stickers and animated `.gif` / `.webm` stickers, with frame-wise OpenCLIP encoding and mean pooling for animated media.

## What This Repository Contains

| path | purpose |
| --- | --- |
| `src/multisticker.py` | Core library: U-Sticker session/sample construction, memory retrieval, OpenCLIP wrappers, animated sticker decoding, `IntentGuidedRetriever`, metrics, and the older head-only helper. |
| `src/utils.py` | Shared JSON, directory, and seed helpers. |
| `scripts/train_am.py` | Main train/eval entry for `head_only`, `image_lora`, `text_lora`, and `dual_lora` experiments. |
| `scripts/eval_direct_clip.py` | Zero-shot frozen OpenCLIP baseline with no trainable retrieval head. |
| `scripts/run_rebuild_pilot.sh` | SLURM/local runner for the pilot or medium experiment matrix. |
| `scripts/archive_run_artifacts.py` | Copies matched logs/results/checkpoints into organized run folders. |
| `organized_runs/rebuild_medium/` | Completed medium-scale result JSONs, logs, histories, plots, and checkpoints used in the report. |
| `Latex_report/report.tex` | Final 4-page CVPR-style report source. |
| `Latex_report/report.pdf` | Compiled final report. |
| `final_report.md` | Longer internal method/result notes with implementation links. |
| `check.md` | Submission self-check against the course report advice. |

## Method Summary

The project is a retrieval system, not a sticker generator. For every validation sample, the model ranks all candidate stickers and reports whether the observed gold sticker appears in the top-K list.

```text
U-Sticker dialogue JSON
    -> 12-hour session split
    -> sticker-reply samples
    -> LLM session memories and reply intents
    -> E5 top-3 memory retrieval from prior sessions
    -> OpenCLIP encodes context, memory, intent, and sticker media
    -> IntentGuidedRetriever ranks the sticker bank
```

The main model uses `OpenCLIP ViT-B-32 / laion2b_s34b_b79k`. Context, memory, and intent strings are encoded separately by the CLIP text encoder, concatenated as vectors, and passed through `IntentGuidedRetriever`. Sticker images are encoded by the CLIP image encoder. For `.gif` and `.webm`, each decoded frame is encoded and the frame features are mean-pooled.

The implemented experiment modes are:

| report name | command/script mode | trainable parameters |
| --- | --- | --- |
| `direct_clip` | `scripts/eval_direct_clip.py` | none; frozen OpenCLIP baseline |
| `head_only` | `train_am.py --tuning-mode head_only` | retriever head and intent classifier |
| `image_lora` | `train_am.py --tuning-mode image_lora` | visual-side LoRA plus retriever head |
| `text_lora` | `train_am.py --tuning-mode text_lora` | text-side LoRA plus retriever head |
| `dual_lora` | `train_am.py --tuning-mode dual_lora` | text-side LoRA, visual-side LoRA, and retriever head |

Memory ablations use `--memory-strategy disabled`; the main model uses `--memory-strategy retrieved_topk`.

## Main Results

The completed medium run uses 30,000 train samples, 4,998 validation samples after filtering, and a 6,758-sticker bank across `.png`, `.gif`, and `.webm`.

Exact R@K means the observed sticker id is in the top-K retrieved list. Group R@K means at least one retrieved sticker has the same majority intent group as the observed sticker. Because each sample has one observed positive, these are single-positive Recall/Hit@K metrics.

| method | Exact R@1 | Exact R@5 | Exact R@30 | MAP/MRR | Group R@1 | Group R@5 | Group R@10 | Group R@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_clip` | 0.0002 | 0.0012 | 0.0028 | 0.0013 | 0.6335 | 0.8778 | **0.9084** | **0.9588** |
| `head_only` | **0.0110** | 0.0296 | 0.1263 | 0.0282 | **0.7933** | 0.8619 | 0.8780 | 0.9406 |
| `image_lora` | 0.0108 | **0.0376** | 0.1188 | **0.0302** | 0.7063 | 0.8812 | 0.9044 | 0.9550 |
| `text_lora` | 0.0102 | 0.0312 | 0.1190 | 0.0280 | 0.7623 | 0.8661 | 0.8880 | 0.9420 |
| `dual_lora` | 0.0100 | 0.0326 | **0.1315** | 0.0286 | 0.6275 | **0.8816** | 0.9064 | 0.9540 |

The most useful interpretation is:

- frozen CLIP already retrieves semantically related stickers, but almost never recovers the exact sticker id;
- retrieved dialogue memory substantially improves the learned head's exact retrieval;
- `dual_lora` is the most balanced trained model for a chat sticker panel, while `image_lora` gives the strongest exact top-5 ranking.

Detailed per-media tables and the no-memory ablation are in `organized_runs/rebuild_medium/mediumReport.md` and `Latex_report/report.pdf`.

## Reproducing the Medium Experiment

The code assumes the U-Sticker files and LLM annotation artifacts are available under the scratch paths used in the Rice cluster environment:

```text
/scratch/rl182/meme/u-sticker/u-sticker-combined.zip
/scratch/rl182/meme/u-sticker/idx_to_domain.txt
/scratch/rl182/meme/usticker_igsr/llm/session_memories_qwen32_gptq_v10_png_gif_webm_merged.jsonl
/scratch/rl182/meme/usticker_igsr/llm/sample_intents_qwen32_gptq_v10_png_gif_webm_merged.jsonl
```

Run the medium experiment matrix:

```bash
cd /home/rl182/dl/V2L/Project-meme/MultiSticker
bash scripts/run_rebuild_pilot.sh
```

Run a smaller smoke test:

```bash
cd /home/rl182/dl/V2L/Project-meme/MultiSticker
RUN_SCALE=pilot bash scripts/run_rebuild_pilot.sh
```

Useful environment overrides:

```bash
RUN_NAME=my_run MAX_TRAIN=5000 MAX_VAL=2000 MAX_STICKERS=3000 bash scripts/run_rebuild_pilot.sh
LORA_BS=32 INFER_BS=64 bash scripts/run_rebuild_pilot.sh
```

The runner writes logs and result artifacts to:

```text
/scratch/rl182/mutlisticker/logs/
/scratch/rl182/mutlisticker/results/
/scratch/rl182/mutlisticker/archives/<run_name>/
/scratch/rl182/mutlisticker/embedding_cache/
```

This repository also contains copied medium-run artifacts under `organized_runs/rebuild_medium/` for report verification.

## Running Individual Commands

Zero-shot frozen CLIP baseline:

```bash
/scratch/rl182/envs/dl/bin/python scripts/eval_direct_clip.py \
  --device cuda \
  --query-mode clip_context \
  --supported-media ".png,.gif,.webm" \
  --max-train-samples 30000 \
  --max-val-samples 5000 \
  --max-stickers 8000 \
  --min-sticker-frequency 2 \
  --run-name rebuild_medium
```

Memory-enabled learned head:

```bash
/scratch/rl182/envs/dl/bin/python scripts/train_am.py \
  --device cuda \
  --tuning-mode head_only \
  --memory-strategy retrieved_topk \
  --supported-media ".png,.gif,.webm" \
  --max-train-samples 30000 \
  --max-val-samples 5000 \
  --max-stickers 8000 \
  --min-sticker-frequency 2 \
  --epochs 5 \
  --train-batch-size 256 \
  --infer-batch-size 256 \
  --run-name rebuild_medium
```

No-memory learned-head ablation:

```bash
/scratch/rl182/envs/dl/bin/python scripts/train_am.py \
  --device cuda \
  --tuning-mode head_only \
  --memory-strategy disabled \
  --supported-media ".png,.gif,.webm" \
  --max-train-samples 30000 \
  --max-val-samples 5000 \
  --max-stickers 8000 \
  --min-sticker-frequency 2 \
  --epochs 5 \
  --train-batch-size 256 \
  --infer-batch-size 256 \
  --run-name rebuild_medium_no_memory_head
```

Change `--tuning-mode` to `image_lora`, `text_lora`, or `dual_lora` for the LoRA experiments.

## Dependencies

The Rice cluster runs used `/scratch/rl182/envs/dl/bin/python`. The code imports:

- PyTorch
- NumPy
- Pillow
- OpenCLIP
- Hugging Face Transformers
- Hugging Face PEFT
- FFmpeg for `.webm` decoding
- Matplotlib for optional history plots

On the cluster, `scripts/run_rebuild_pilot.sh` sets the cache directories and loads FFmpeg when available.

## Authorship and External Components

Code under `src/` and `scripts/` is project code written for MultiSticker, including the manifest builder, memory-injection logic, animated media handling, training/evaluation scripts, metrics, artifact archiving, and the `IntentGuidedRetriever` module.

The project uses external libraries and pretrained components as off-the-shelf building blocks: OpenCLIP for the CLIP text/image encoders, `intfloat/multilingual-e5-small` for memory retrieval embeddings, PEFT for LoRA injection, PyTorch for optimization, and FFmpeg/Pillow for media decoding. The report cites these components and distinguishes them from the project-specific code.

AI assistance was used for coding and writing support and is acknowledged in the final report.

## Report and Submission Checklist

The final report follows the course guidance in the provided CVPR-style template:

- real input/output and pipeline figures are exported as PDF;
- exact model names are used for OpenCLIP, E5, and LoRA settings;
- the report includes a frozen CLIP baseline, memory ablation, LoRA ablations, qualitative examples, and quantitative tables;
- figures, logs, JSON metrics, checkpoints, and demo assets are included under `Latex_report/`, `results/`, `logs/`, and `organized_runs/`.

