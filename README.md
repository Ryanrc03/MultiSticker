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
| `logs/` | Runtime logs. Ignored by git. |
| `results/` | JSON/PT outputs from experiments. Ignored by git. |

## Main Experiment Modes

The planned rebuild compares:

| report name | command/script mode | meaning |
| --- | --- | --- |
| `direct_clip` | `scripts/eval_direct_clip.py` | Frozen OpenCLIP text/image retrieval, no trainable head. |
| `memory` | `train_am.py --tuning-mode head_only --memory-strategy retrieved_topk` | Memory-enabled retriever head. |
| `lora_image` | `train_am.py --tuning-mode image_lora` | Visual-side LoRA plus retriever head. |
| `lora_text` | `train_am.py --tuning-mode text_lora` | Text-side LoRA plus retriever head. |
| `dual_lora` | `train_am.py --tuning-mode dual_lora` | Text-side and visual-side LoRA plus retriever head. |

## Pilot Run

```bash
cd /home/rl182/dl/V2L/Project-meme/MultiSticker
bash scripts/run_rebuild_pilot.sh
```

The default pilot uses a smaller subset for iteration:

```text
max train samples: 5,000
max validation samples: 2,000
max stickers: 3,000
min sticker frequency: 2
epochs: 2
```

## Metrics

The code still stores exact metrics under names like `p@1`, `p@5`, and `p@30`. In this project, each validation sample has one observed gold sticker, so these fields should be read as single-positive `Recall@K` or `Hit@K`: whether the gold sticker appears in the top-K retrieved list.

The report also tracks semantic-group Recall/Hit@K, where a retrieval is counted as correct if any top-K sticker has the same majority intent group as the observed gold sticker.
