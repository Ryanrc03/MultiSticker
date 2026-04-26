# MultiSticker Rebuild Plan

## 0. Current Understanding

The current `MultiSticker` pipeline is an IGSR-style sticker retrieval system:

```text
dialogue turn
  -> context_text
  -> optional retrieved memory_text
  -> LLM intent_text
  -> OpenCLIP text encoder
  -> IntentGuidedRetriever
  -> score against OpenCLIP sticker image bank
```

The current implementation already supports:

- all-media stickers: `.png`, `.gif`, `.webm`
- memory retrieval with `intfloat/multilingual-e5-small`
- frozen OpenCLIP head-only training
- text-side LoRA
- image-side LoRA
- dual text+image LoRA
- exact sticker and semantic-group retrieval metrics

Important code paths:

- Manifest/data construction: `/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py`
- AM training entry: `/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py`
- Pilot script: `/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/run_pilot.sh`
- Full script: `/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/run_full.sh`
- Current report: `/home/rl182/dl/V2L/Project-meme/MultiSticker/final_report.md`

## 1. Dataset Size and Why Rebuild

Current full all-media run size:

| item | value |
| --- | ---: |
| raw JSON files | 67 |
| raw dialogue rows | 8,951,733 |
| sessions after 12h split | 8,129 |
| provisional supported-media sticker samples | 368,503 |
| full train size before cap | 328,953 |
| current train cap | 100,000 |
| validation samples | 18,403 |
| candidate stickers before decodable filtering | 44,510 |
| decodable sticker bank | 42,319 |
| supported media | `.png`, `.gif`, `.webm` |

This is too large for fast iteration because the expensive parts scale with the number of samples and the size of the sticker bank:

- memory manifest build encodes all session summaries and sample queries;
- frozen text mode precomputes `context_text`, `memory_text`, and `intent_text` features;
- image mode encodes the whole sticker bank;
- image LoRA repeatedly re-encodes the sticker bank and batch positives;
- evaluation scores every validation query against every sticker.

The current pilot setup is much more manageable:

| item | value |
| --- | ---: |
| train samples | 5,000 |
| validation samples | 2,000 |
| max candidate stickers | 3,000 |
| decodable sticker bank | 1,873 |
| min sticker frequency | 2 |
| epochs | 2 |

This should become the default rebuild scale until the experimental design is stable.

## 2. Smaller Dataset Plan

Use a staged dataset ladder instead of jumping directly to the full 42K-bank setting.

| stage | purpose | max train | max val | max stickers | min freq | epochs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| smoke | check code paths only | 512 | 256 | 500 | 2 | 1 |
| pilot | compare methods quickly | 5,000 | 2,000 | 3,000 | 2 | 2 |
| medium | more stable ranking | 20,000 | 5,000 | 8,000 | 2 | 3 |
| full | final table only | 100,000 | 18,403 | 0 | 1 | 3 |

Recommended first target:

```bash
--max-train-samples 5000
--max-val-samples 2000
--max-stickers 3000
--min-sticker-frequency 2
--epochs 2
```

If pilot runtime is still too long, first reduce `--max-stickers`, then reduce `--max-val-samples`. Evaluation cost is roughly `val_samples x sticker_bank_size`, so the sticker bank is the first lever to pull.

## 3. Experiment Matrix

The new comparison should have five rows:

| experiment | memory? | trainable params | text input | image input | purpose |
| --- | --- | --- | --- | --- | --- |
| `direct_clip` | no | none | raw context or intent text | frozen OpenCLIP image bank | pure zero-shot CLIP baseline |
| `memory` | yes | retriever head only | context + memory + intent | frozen OpenCLIP image bank | current IGSR-style memory model |
| `lora_image` | yes | image-side LoRA + head | context + memory + intent | image-side adapted OpenCLIP | test sticker visual-domain adaptation |
| `lora_text` | yes | text-side LoRA + head | context + memory + intent | frozen OpenCLIP image bank | test dialogue/query adaptation |
| `dual_lora` | yes | text-side LoRA + image-side LoRA + head | context + memory + intent | adapted OpenCLIP | test joint adaptation |

Mapping to current code:

| desired experiment | current support | code change needed |
| --- | --- | --- |
| `direct_clip` | not directly supported | add a no-training evaluation mode |
| `memory` | `head_only` with `memory_strategy=retrieved_topk` | rename/report as `memory` |
| `lora_image` | `image_lora` | mostly already supported |
| `lora_text` | `text_lora` | mostly already supported |
| `dual_lora` | `dual_lora` | already supported |

## 4. No-Memory Baseline

We should add an explicit no-memory switch because the current AM script does not expose `config.data.memory_strategy` from the CLI.

Required change in `scripts/train_am.py`:

```text
add arg:
  --memory-strategy {retrieved_topk,recent_topk,disabled}

in build_config:
  config.data.memory_strategy = args.memory_strategy
```

Then we can run:

```bash
--memory-strategy disabled
```

This sets `memory_text` to `"memory disabled"` in `prepare_manifest`. It keeps the architecture identical but removes useful retrieved memory. This is the cleanest ablation for whether memory helps.

Recommended ablations:

| experiment | tuning mode | memory strategy |
| --- | --- | --- |
| `no_memory_head` | `head_only` | `disabled` |
| `memory` | `head_only` | `retrieved_topk` |

The five-row main table requested above can use `direct_clip` as the strict baseline and `memory` as the head-only learned baseline. The no-memory head ablation is still useful as an extra diagnostic row.

## 5. Direct CLIP Baseline

`direct_clip` should not train `IntentGuidedRetriever`. It should encode a text query and rank the frozen sticker image bank directly:

```text
query_text -> OpenCLIP text encoder -> query_feature
sticker image -> OpenCLIP image encoder -> image_feature
score = query_feature dot image_feature
```

Candidate query variants:

| variant | query text |
| --- | --- |
| `clip_context` | `context_text` |
| `clip_context_intent` | `context_text + intent_text` |
| `clip_context_memory_intent` | `context_text + memory_text + intent_text` |

For a clean baseline, start with `clip_context`, because it tests what frozen OpenCLIP can do without memory, intent, or trainable heads. Then optionally report the other two as diagnostic variants.

Implementation options:

1. Add `--eval-only-mode direct_clip` to `train_am.py`.
2. Or create a small separate script: `scripts/eval_direct_clip.py`.

I recommend a separate script first because it is simpler and avoids mixing no-training baselines with LoRA training logic.

## 6. Metrics: Use Recall/Hit@K

The report should consistently use retrieval terminology:

```text
Exact Recall@K / Exact Hit@K:
  whether the observed gold sticker id appears in top K

Group Recall@K / Group Hit@K:
  whether any top-K sticker has the same intent group as the gold sticker
```

The code currently names these fields `p@1`, `p@3`, `p@5`, `p@10`, and `p@30`. Because each validation sample has only one observed gold sticker, these values are better interpreted as single-positive Recall@K/Hit@K rather than classic multi-positive Precision@K.

Main metrics to report:

| metric | why |
| --- | --- |
| Exact Recall@1 | strict auto-send quality |
| Exact Recall@5 | small candidate panel |
| Exact Recall@10 | medium candidate panel |
| Exact Recall@30 | broad retrieval quality |
| Group Recall@1 | strict semantic quality |
| Group Recall@5 | semantic candidate panel quality |
| Group Recall@30 | whether top candidates contain the right intent group |
| MRR | ranking quality of the observed gold sticker |

For the main conclusion, prioritize:

```text
Exact Recall@30 + Group Recall@30
```

These best match the retrieve use case where the system returns a candidate list. Use Recall@1 only when discussing automatic single-sticker reply.

## 7. Result Table Template

Use this table in the next report update:

| experiment | memory strategy | trainable part | sticker bank | train | val | Exact R@1 | Exact R@5 | Exact R@10 | Exact R@30 | Group R@1 | Group R@5 | Group R@30 | MRR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_clip` | disabled | none | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `memory` | retrieved_topk | head | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `lora_image` | retrieved_topk | visual LoRA + head | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `lora_text` | retrieved_topk | text LoRA + head | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `dual_lora` | retrieved_topk | text+visual LoRA + head | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Optional ablation table:

| experiment | memory strategy | trainable part | Exact R@30 | Group R@30 | purpose |
| --- | --- | --- | ---: | ---: | --- |
| `no_memory_head` | disabled | head | TBD | TBD | isolate memory effect |
| `memory` | retrieved_topk | head | TBD | TBD | memory-enabled counterpart |

## 8. Implementation Checklist

1. Add CLI support for `--memory-strategy`.
   - File: `scripts/train_am.py`
   - Update `parse_args()`.
   - Update `build_config()`.
   - Save strategy into result JSON.

2. Add direct CLIP eval.
   - Preferred new file: `scripts/eval_direct_clip.py`
   - Reuse `prepare_manifest`, `OpenClipEncoder`, sticker extraction/filtering, `_metrics_from_scores`, and `_group_metrics_from_scores`.
   - Save results under `results/direct_clip_<run_name>.json`.

3. Add a new run script for the rebuild experiments.
   - Suggested file: `scripts/run_rebuild_pilot.sh`
   - Default scale: 5K train, 2K val, 3K max stickers, 2 epochs.
   - Run modes: `direct_clip memory image_lora text_lora dual_lora`.

4. Rename report labels without necessarily renaming result files.
   - `head_only + retrieved_topk` -> `memory`
   - `image_lora` -> `lora_image`
   - `text_lora` -> `lora_text`
   - `dual_lora` -> `dual_lora`

5. Update result extraction helper.
   - Convert JSON `p@K` fields to table labels `Recall@K`.
   - Keep a note that raw code names remain `p@K`.

## 9. First Rebuild Run

Recommended pilot command pattern after code changes:

```bash
RUN_NAME=rebuild_pilot \
MAX_TRAIN=5000 \
MAX_VAL=2000 \
MAX_STICKERS=3000 \
MIN_FREQ=2 \
EPOCHS=2 \
bash /home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/run_rebuild_pilot.sh
```

Expected outputs:

```text
results/direct_clip_rebuild_pilot.json
results/am_head_only_rebuild_pilot.json
results/am_image_lora_rebuild_pilot.json
results/am_text_lora_rebuild_pilot.json
results/am_dual_lora_rebuild_pilot.json
```

Report these as:

```text
direct_clip
memory
lora_image
lora_text
dual_lora
```

## 10. Decision Rules

Use the pilot results to decide the next step:

- If `direct_clip` is close to `memory`, the trainable head may not be adding enough value.
- If `memory` beats `no_memory_head`, keep memory retrieval.
- If `lora_text` improves Exact Recall@K, the query/text side needs adaptation.
- If `lora_image` improves Group Recall@K or Exact Recall@K, visual sticker-domain adaptation is useful.
- If `dual_lora` does not beat single-side LoRA, avoid its extra cost.
- If all methods have high Group Recall@30 but low Exact Recall@30, focus on reranking, better sticker grouping, or multi-positive evaluation rather than only training larger encoders.
