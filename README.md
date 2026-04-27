# MultiSticker

Multimodal sticker retrieval with dialogue memory.

## Repository Structure

| path | purpose |
| --- | --- |
| `src/` | Core Python package for dataset construction, memory retrieval, OpenCLIP encoding, sticker-bank encoding, metrics, and utility helpers. This directory was renamed from `meme_project/` so imports now use `from src...`. |
| `src/multisticker.py` | Main library module. It defines config dataclasses, U-Sticker session/sample construction, E5 memory retrieval, OpenCLIP wrappers, animated sticker frame loading, `IntentGuidedRetriever`, metrics, and the older head-only training helper. |
| `src/utils.py` | Small shared utilities such as JSON save/load helpers. |
| `scripts/train_am.py` | Main train/eval entry for the rebuild experiments. It supports `head_only`, `image_lora`, `text_lora`, and `dual_lora`, plus `--memory-strategy retrieved_topk/recent_topk/disabled`. |
| `scripts/eval_direct_clip.py` | Zero-shot direct OpenCLIP retrieval baseline. It ranks stickers without training a retriever head. |
| `scripts/run_rebuild_pilot.sh` | Pilot runner for the rebuild experiment matrix: direct CLIP, memory/head-only, image LoRA, text LoRA, and dual LoRA. |
| `scripts/rebuild_pilot.sbatch` | SLURM wrapper for launching the rebuild pilot on the cluster. |
| `final_report.md` | Current method and result summary, including memory retrieval, context/memory/intent inputs, metric interpretation, and current result tables. |
| `rebuildPlan.md` | Proposed rebuild plan: smaller dataset ladder, experiment matrix, no-memory ablation, direct CLIP baseline, and Recall/Hit@K reporting. |
| `/scratch/rl182/mutlisticker/logs/` | Runtime logs. |
| `/scratch/rl182/mutlisticker/results/` | JSON/PT outputs from experiments. |
| `/scratch/rl182/mutlisticker/archives/` | Per-experiment copies of matching logs/results plus a manifest of the run parameters. |
| `/scratch/rl182/mutlisticker/embedding_cache/` | Reusable frozen CLIP embedding cache. |

## Main Experiment Modes

The planned rebuild compares:

| report name | command/script mode | meaning |
| --- | --- | --- |
| `direct_clip` | `scripts/eval_direct_clip.py` | Frozen OpenCLIP text/image retrieval, no trainable head. |
| `memory` | `train_am.py --tuning-mode head_only --memory-strategy retrieved_topk` | Memory-enabled retriever head. |
| `lora_image` | `train_am.py --tuning-mode image_lora` | Visual-side LoRA plus retriever head. |
| `lora_text` | `train_am.py --tuning-mode text_lora` | Text-side LoRA plus retriever head. |
| `dual_lora` | `train_am.py --tuning-mode dual_lora` | Text-side and visual-side LoRA plus retriever head. |

## Medium Run

```bash
cd /home/rl182/dl/V2L/Project-meme/MultiSticker
bash scripts/run_rebuild_pilot.sh
```

The default run now uses the medium profile for a longer training pass:

```text
max train samples: 30,000
max validation samples: 5,000
max stickers: 8,000
min sticker frequency: 2
epochs: 5
SBATCH CPUs: 16
```

For a smaller smoke run, use:

```bash
RUN_SCALE=pilot bash scripts/run_rebuild_pilot.sh
```

The pilot profile uses:

```text
max train samples: 5,000
max validation samples: 2,000
max stickers: 3,000
min sticker frequency: 2
epochs: 2
```

Each train run writes the normal stdout log plus extra training artifacts:

| artifact | contents |
| --- | --- |
| `/scratch/rl182/mutlisticker/logs/am_<mode>_<memory>_<run>_epochs.jsonl` | One JSON record per epoch with metrics, per-media breakdowns, and validation examples. |
| `/scratch/rl182/mutlisticker/results/am_<mode>_<memory>_<run>_history.csv` | Compact epoch history for plotting/comparison. |
| `/scratch/rl182/mutlisticker/results/am_<mode>_<memory>_<run>_history.png` | Loss and validation Recall@30 curves when `matplotlib` is available. |

Successful runner steps are also copied into grouped archive folders:

```text
/scratch/rl182/mutlisticker/archives/<run_name>/scale-<scale>__epochs-<n>__train-<n>__val-<n>__stickers-<n>__minfreq-<n>__media-png-gif-webm/<artifact_prefix>/
```

Each archive folder contains the matched log/result files and `manifest.json`, which records the iteration count, dataset limits, media inputs, memory/intent files, batch sizes, LoRA settings, and file paths.

## Metrics

The current train/eval JSON stores exact metrics under names like `recall@1`, `recall@5`, and `recall@30`. In this project, each validation sample has one observed gold sticker, so these fields are single-positive `Recall@K` or `Hit@K`: whether the gold sticker appears in the top-K retrieved list.

The report also tracks semantic-group Recall/Hit@K, where a retrieval is counted as correct if any top-K sticker has the same majority intent group as the observed gold sticker.
