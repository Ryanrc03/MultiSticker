# Retrieve Method Summary

## 1. Method Overview

### 1.1 Pipeline sketch

```text
U-Sticker dialogue json
        |
        | split by 12h session gap
        v
session records + sticker-turn samples
        |
        | LLM annotation
        | - session_memory_text for each session
        | - intent_label / intent_text for each sticker-reply sample
        v
manifest
        |
        | memory retrieval: multilingual-e5-small
        | query = current session text
        | passages = previous session memories from the same file/user stream
        | select top-3 historical sessions
        v
sample text fields
        |---------------- context_text ----------------|
        |---------------- memory_text -----------------|
        |---------------- intent_text -----------------|
        v
OpenCLIP text encoder: ViT-B-32 / laion2b_s34b_b79k
        |
        | concatenate [context, memory, intent]
        v
IntentGuidedRetriever head
        |--> intent_repr
        |--> intent classifier over intent groups
        |
        v
cosine / dot-product similarity against OpenCLIP image sticker bank
        |
        v
rank all candidate stickers; report exact Hit@K and semantic-group Hit@K
```

The method is an intent-guided CLIP retrieval model. OpenCLIP provides a text encoder and an image encoder. The text side encodes three views of the query: recent dialogue context, retrieved long-term memory, and LLM-generated reply intent. These three vectors are concatenated and projected by `IntentGuidedRetriever` into the same dimension as the sticker image embeddings. Retrieval logits are `intent_repr @ image_bank.T / temperature`, and a small auxiliary classifier predicts the sticker's intent group.

Implementation links: [IntentGuidedRetriever](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:217), [forward scoring](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:233), [AM training entry](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py:196).

### 1.2 Memory

Sketch:

```text
previous sessions
   | -> LLM session_memory_text -> E5 passage embeddings
current session
   | -> current_session_text  -> E5 query embedding
                                  |
                                  v
                         top-3 similar old sessions
                                  |
                                  v
                        memory_text for CLIP query
```

Memory is added before CLIP encoding, at manifest construction time. Each raw dialogue file is split into sessions using a 12-hour gap. For a sample in the current session, the code collects up to 8 previous sessions from the same file, embeds their `session_memory_text` as passages, embeds the current session text as a query, and retrieves the top-3 most similar memories. The selected memory texts are concatenated with ` || ` and stored as `memory_text`.

The memory encoder is `intfloat/multilingual-e5-small`, implemented as mean pooling over the transformer hidden states with L2 normalization. It uses E5-style prefixes: `passage: ` for session memories and `query: ` for current-session queries.

Worked example from the full AM manifest:

```text
sample_id: 16_session_0093#turn431192
session_id: 16_session_0093
domain: Catizen / play-to-earn game discussion
gold sticker reply intent: gratitude
```

The current session query is the text field `current_session_text`. It is a multilingual dialogue about Catizen/wCATI/xZEN, airdrop timing, game progress, and help from other users. A shortened view is:

```text
04757c16: Hello
16d12550: Let's play and wait for announcements
9d94fbd1: Are you from catizen team? I mean team not mod
16d12550: [sticker:5271961661647884354.gif]
6d04a9ac: Well i already reached level 219 ...
16d12550: there hasn't been a drop yet, increase your cat level ...
ff6f56c8: Thanks to you I collected a huge amount of wCATI and xZEN ...
04757c16: Please is there some one explain me how to use this app?
ff6f56c8: After you receive the Airdrob, I can gift you as much as you want.
```

For this sample, the available history window contains the previous 8 sessions from the same file/user stream:

| history session | LLM `session_memory_text` |
| --- | --- |
| `16_session_0085` | The session was brief and only contained a single digit `1`, possibly indicating a stock price, quantity, or a rating. |
| `16_session_0086` | `3cc0363e` inquired about Cp, likely referring to a financial metric or company. |
| `16_session_0087` | `4e086323` expressed approval with a thumbs up sticker. |
| `16_session_0088` | Gm |
| `16_session_0089` | `f4e9795f` sent a sticker indicating excitement or happiness, likely related to a positive financial update or achievement. |
| `16_session_0090` | `c53db665` inquired about the current status of a project, seeking immediate updates. |
| `16_session_0091` | Earn tons in my binary link |
| `16_session_0092` | `f4e9795f` requested financial statistics, `bd80a41d` provided the stats, and `3d2cf429` responded with a cute emoji. |

The calculation is:

```text
p_i = E5("passage: " + session_memory_text_i)
q   = E5("query: " + current_session_text)

p_i = normalize(mean_pool(last_hidden_state_i))
q   = normalize(mean_pool(last_hidden_state_q))

score_i = p_i dot q
```

Because both vectors are L2-normalized, the dot product is cosine similarity. The code sorts all 8 history sessions by `score_i` and keeps the largest `top_k_memories = 3`. In the saved manifest, the selected top-3 memories for this real sample are:

```text
1. 16_session_0092:
   f4e9795f requested financial statistics, bd80a41d provided the stats,
   and 3d2cf429 responded with a cute emoji.

2. 16_session_0089:
   f4e9795f sent a sticker indicating excitement or happiness,
   likely related to a positive financial update or achievement.

3. 16_session_0087:
   4e086323 expressed approval with a thumbs up sticker.
```

The output written into the sample is the concatenated `memory_text`:

```text
f4e9795f requested financial statistics, bd80a41d provided the stats,
and 3d2cf429 responded with a cute emoji. ||
f4e9795f sent a sticker indicating excitement or happiness,
likely related to a positive financial update or achievement. ||
4e086323 expressed approval with a thumbs up sticker.
```

This `memory_text` is not used as an E5 vector in the final sticker model. It is stored as text, then later encoded by the OpenCLIP text encoder together with `context_text` and `intent_text`. In this example, the retrieved memory gives the model a weak long-term prior that this chat often involves financial/game progress, positive updates, and appreciative or approving sticker reactions.

The memory retrieval score is only a construction-time ranking score; there is no separate labeled memory-retrieval metric in the current code. The reported metrics are end-to-end sticker retrieval metrics after memory has been injected: exact sticker Hit@K/Recall@K, semantic-group Hit@K, and reranked/fused group metrics. In the current result JSONs these fields are still named `p@1`, `p@5`, `p@10`, and `p@30`, but because each sample has only one observed gold sticker, they are best interpreted as whether the gold item appears in the top-K list rather than as multi-positive precision. Therefore memory is evaluated indirectly: if better memory helps choose better stickers, it should improve those downstream retrieval metrics.

Implementation links: [memory config](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:129), [memory selection](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:449), [MeanPoolingEncoder](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:481), [manifest memory retrieval](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:973).

### 1.3 Context, memory, and intent text

For every sticker-reply training sample, the text-side input has three parts:

```text
context_text = recent local dialogue before the target sticker
memory_text  = retrieved long-term session memory from previous sessions
intent_text  = LLM description of what the sticker reply should express
```

`context_text` is the direct local evidence. When a turn contains the target sticker, the code takes up to `max_context_turns = 12` previous formatted turns from the same session. This field answers: what did people just say immediately before the sticker was sent?

Example from `16_session_0093#turn431192`:

```text
d12ccf08: Здравствуйте! Написала 3 августа в поддержку. Ничего не решили. ||
16d12550: and I have zero coins wCati😅 ||
ff6f56c8: @#USER Thanks to you I collected a huge amount of wCATI and xZEN from the event. ||
04757c16: Please is there some one explain me how to use this app? I join today the group. ||
ff6f56c8: After you receive the Airdrob, I can gift you as much as you want. ☺️
```

`memory_text` is the long-range prior. It does not come from the immediate local context; it is retrieved from previous sessions in the same file/user stream. This field answers: what similar historical conversation patterns or sticker reactions have appeared before?

For the same example:

```text
f4e9795f requested financial statistics, bd80a41d provided the stats,
and 3d2cf429 responded with a cute emoji. ||
f4e9795f sent a sticker indicating excitement or happiness,
likely related to a positive financial update or achievement. ||
4e086323 expressed approval with a thumbs up sticker.
```

`intent_text` is the LLM-generated semantic target for the sticker reply. It abstracts away from the exact sticker id and describes the communicative function of the reply. This field answers: what should the sticker mean in this moment?

For the same example:

```text
Expressing sincere thanks for answering questions, showing appreciation for the help and support.
```

Conceptually, the model's final text-side information is:

```text
final_text_information = context_text + memory_text + intent_text
```

In the actual implementation, these strings are not concatenated into one long prompt before CLIP. They are encoded separately by the OpenCLIP text encoder:

```text
context_text -> OpenCLIP text encoder -> context_features
memory_text  -> OpenCLIP text encoder -> memory_features
intent_text  -> OpenCLIP text encoder -> intent_text_features
```

Then `IntentGuidedRetriever` concatenates the three vectors, not the raw strings:

```text
combined = concat(context_features, memory_features, intent_text_features)
intent_repr = normalize(MLP(combined))
retrieval_logits = intent_repr @ image_bank.T / temperature
```

This design keeps the three sources separate until the trainable retrieval head. The head can learn how much to rely on immediate context, retrieved memory, and explicit intent for each sticker retrieval decision.

Implementation links: [context construction](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:893), [intent text load](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:1014), [three-vector concat](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:240), [AM text encoding](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py:253).

### 1.4 Multi-frame stickers

Sketch:

```text
.png
  -> first/only frame -> OpenCLIP image encoder -> normalize

.gif / .webm
  -> decode all frames -> OpenCLIP each frame -> mean pool -> normalize

image-LoRA training
  -> frozen full bank + grad-enabled positive stickers patched into bank
```

The supported sticker formats are `.png`, `.gif`, and `.webm`. Static images are treated as one-frame clips. For animated `.gif` and `.webm`, the code decodes all frames, runs every frame through the OpenCLIP image encoder, mean-pools frame features, and normalizes the pooled vector. `.webm` decoding uses `ffmpeg`; `.gif` decoding uses PIL `ImageSequence`.

During image-LoRA training, the full image bank is re-encoded without gradients for memory efficiency. Then, for each batch, only the unique positive stickers in that batch are re-encoded with gradients and patched back into a cloned bank. Animated positives are capped to at most 4 uniformly sampled frames when gradients are enabled.

Implementation links: [media constants](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:26), [OpenCLIP image encoding](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:537), [ffmpeg webm decoding](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:591), [gif/webm frame loader](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:621), [LoRA image-bank encoding](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py:139), [positive re-encoding patch](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py:352).

### 1.5 Fine-tuning: head, adapter, LoRA

Sketch:

```text
head_only:
  frozen OpenCLIP text/image encoders
  train IntentGuidedRetriever only

text_lora:
  train text-side LoRA + retriever head

image_lora:
  train visual-side LoRA + retriever head

dual_lora:
  train text-side LoRA + visual-side LoRA + retriever head
```

The current implemented tuning modes are `head_only`, `image_lora`, `text_lora`, and `dual_lora`. There is no separate classic adapter module in the current `MultiSticker` code; the implemented parameter-efficient adaptation is LoRA through PEFT.

LoRA is injected into OpenCLIP linear modules named `out_proj`, `c_fc`, and `c_proj`, with default `r=8`, `alpha=16`, and dropout `0.05`. All non-LoRA OpenCLIP parameters are frozen. In `image_lora`, only LoRA parameters whose names start with `visual.` remain trainable. In `text_lora`, visual LoRA parameters are frozen. In `dual_lora`, both text-side and visual-side LoRA parameters are trainable. The retrieval head is always trainable.

Implementation links: [mode arguments](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py:52), [LoRA injection](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py:104), [OpenCLIP LoRA mode switch](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py:210), [optimizer parameter groups](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py:282), [training loss](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py:394).

### 1.6 OpenCLIP architecture used here

Sketch:

```text
Text branch:
  context / memory / intent text -> tokenizer -> CLIP text Transformer -> 512-d embedding

Image branch:
  sticker frame/image -> preprocess -> ViT-B/32 visual Transformer -> 512-d embedding

Retrieval:
  projected query embedding <dot product> sticker image bank
```

The code uses `OpenCLIP ViT-B-32` pretrained on `laion2b_s34b_b79k`. Architecturally, this is a dual-encoder CLIP model:

```text
text tokens -> CLIP text Transformer -> text embedding -> L2 normalize
image/frame -> ViT-B/32 visual encoder -> image embedding -> L2 normalize

retriever intent_repr and image embeddings live in the same CLIP embedding space.
```

The project does not manually reimplement OpenCLIP internals; it calls `create_model_and_transforms()` and `get_tokenizer()` from `open_clip`. Text is encoded by `model.encode_text()`, images by `model.encode_image()`, then both are normalized.

Implementation links: [OpenClipEncoder init](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:512), [text encoding](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:527), [image encoding](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:554), [model config](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:151).

## 2. Dataset and Annotation

Sketch:

```text
raw U-Sticker json files
        |
        v
12h session split + domain map
        |
        v
sticker-turn samples
        |
        v
filter supported media / eligible train stickers
        |
        v
train / val manifest
```

The dataset is U-Sticker from `/scratch/rl182/meme/u-sticker`, with stickers extracted from `u-sticker-combined.zip`. The current all-media setup supports `.png`, `.gif`, and `.webm`. Dialogues are loaded from JSON files, mapped to domains with `idx_to_domain.txt`, and split into sessions by a 12-hour inactivity gap.

Each training sample is a turn where the current message contains a usable sticker and has at least 2 previous context turns. The model input is the previous dialogue context, retrieved memory, and intent annotation. The positive label is the actual sticker used in that turn. The full run summary is: 8,129 sessions, 368,503 provisional supported-media sticker samples, 100,000 training samples, 18,403 validation samples, and 42,319 decodable stickers in the retrieval bank.

Implementation links: [raw sessions](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:874), [sample construction](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:893), [label filtering and split caps](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:941), [full run config](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/run_full.sh:21).

### 2.1 What was annotated

Sketch:

```text
session-level annotation:
  session_text -> LLM -> session_memory_text

sample-level annotation:
  context_text + retrieved memory -> LLM -> intent_label + intent_text + confidence
```

Two LLM artifacts are generated:

1. Session memory: one `session_memory_text` for each session. It summarizes facts, tone, speaker style, emotion, and sticker cues for future retrieval.
2. Sample intent: one `intent_label`, `intent_text`, and confidence for each sticker-reply sample. The label is from a compact 10-class taxonomy: `celebration_approval`, `humor_teasing`, `encouragement_support`, `gratitude`, `affection_care`, `confusion_curiosity`, `frustration_disapproval`, `disappointment_sadness`, `surprise_reaction`, `neutral_acknowledgment`.

The all-media runs use merged annotation files:
`/scratch/rl182/meme/usticker_igsr/llm/session_memories_qwen32_gptq_v10_png_gif_webm_merged.jsonl` and
`/scratch/rl182/meme/usticker_igsr/llm/sample_intents_qwen32_gptq_v10_png_gif_webm_merged.jsonl`.

Implementation links: [taxonomy](/home/rl182/dl/V2L/Project-meme/scripts/run_generate_usticker_llm_artifacts.py:29), [annotation output paths](/home/rl182/dl/V2L/Project-meme/scripts/run_generate_usticker_llm_artifacts.py:354), [manifest reads annotations](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:879), [sample intent fallback](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:1014).

### 2.2 Annotation prompts

Sketch:

```text
session prompt asks:
  summarize long-term useful memory

intent prompt asks:
  choose 1 label from 10 classes
  describe what sticker reply should express
  return strict JSON
```

Session-memory prompt:

```text
You summarize a chat session for future sticker retrieval.
Write 2-4 short sentences that preserve salient facts, tone, speaker style, emotion, and any sticker cues.
Do not think aloud. Do not explain your reasoning. Be concise and concrete.

Domain: {domain}
Session text:
{session_text}

Return JSON only in the form {"session_memory_text":"..."}
```

Intent-label prompt:

```text
You label sticker-reply intent for retrieval.
Classify the reply into exactly one intent label from this closed set:
{10 intent labels}

Guidelines:
- Choose the label that best matches the communicative goal of the reply, not just surface sentiment.
- Use the conversation context first, then the retrieved memory as a weak prior.
- If multiple labels seem possible, choose the most specific one.
- If the reply is mainly a simple acknowledgment or reaction without strong affect, use neutral_acknowledgment.

Return valid JSON only with this schema:
{"intent_label":"one label from the closed set","intent_text":"1-2 short sentences describing the likely sticker intent, tone, and what kind of sticker would fit","confidence":"high|medium|low"}

Context:
{context_text}

Retrieved memory:
{memory_text}

Return JSON only.
```

Implementation links: [session prompt](/home/rl182/dl/V2L/Project-meme/scripts/run_generate_usticker_llm_artifacts.py:194), [intent prompt](/home/rl182/dl/V2L/Project-meme/scripts/run_generate_usticker_llm_artifacts.py:205), [batched generation](/home/rl182/dl/V2L/Project-meme/scripts/run_generate_usticker_llm_artifacts.py:387), [intent generation](/home/rl182/dl/V2L/Project-meme/scripts/run_generate_usticker_llm_artifacts.py:422).

## 3. Metrics and Positive/Negative Samples

Sketch:

```text
for each validation sample:
  query representation -> scores over all stickers
                         |
                         v
                 sorted ranking list
                         |
        exact hit: gold sticker in top-K?
        group hit: any same-intent-group sticker in top-K?

training:
  positive = real sticker used in the dialogue turn
  negatives = all other stickers in the bank
```

The main exact metrics in the result files are named `p@1`, `p@3`, `p@5`, `p@10`, `p@30`, `MAP`, and `MRR`. For retrieval analysis, the `p@K` fields should be read as exact `Hit@K` or single-positive `Recall@K`: for each validation sample, the system ranks every sticker in the candidate bank by retrieval score, and the metric is 1 if the observed gold sticker appears in the top K, otherwise 0. The final number is averaged over all validation samples. Since each sample has one observed gold sticker, `MAP` and `MRR` are both implemented as the mean reciprocal rank of the gold sticker.

This distinction matters because sticker retrieval is not a normal multi-class classification problem. The data records only the sticker that the user actually sent, but many other stickers can be valid replies with the same emotion, intent, or visual style. Exact `Hit@K` measures whether the system can recover the observed sticker; semantic-group Hit@K measures whether the retrieved list contains a sticker with the same intent group. For a retrieval product that shows users a candidate panel, `Hit@10` or `Hit@30` is usually more informative than only `Hit@1`. If the system must automatically send exactly one sticker, then `Hit@1` is the strictest metric.

Semantic-group metrics use the same ranked list, but a hit is counted when any retrieved sticker has the same intent group as the gold sticker. A sticker's group is assigned by majority `intent_label` among training samples using that sticker, with fallback to all splits and then `neutral_acknowledgment`. The code also computes fused and two-stage group rerank scores, where intent-classifier logits act as a group prior over sticker scores.

Positive samples are the observed sticker labels from real dialogue turns. Negative samples are implicit: because `retrieval_logits` has shape `[batch_size, full_sticker_bank_size]`, cross-entropy treats the gold sticker as the only positive class and all other stickers in the bank as negatives for that sample. There is no separate hard-negative mining in the AM script. For image-LoRA training, gradients flow through the positive sticker image features in the batch, while the rest of the bank acts as frozen negatives for efficiency.

Implementation links: [exact metrics](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:694), [semantic group metrics](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:715), [group prior and two-stage rerank](/home/rl182/dl/V2L/Project-meme/MultiSticker/src/usticker_igsr.py:744), [group assignment](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py:220), [evaluation metrics](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py:299), [training positive/all-bank loss](/home/rl182/dl/V2L/Project-meme/MultiSticker/scripts/train_am.py:394).

## 4. Current Results

The tables below keep the raw JSON metric names in parentheses but use retrieval-oriented labels. `Exact Hit@K` means the observed gold sticker appears in the top-K retrieved list. `Group Hit@K` means at least one top-K sticker has the same majority intent group as the gold sticker.

Full all-media runs:

| mode | sticker bank | train | val | Exact Hit@1 (`p@1`) | Exact Hit@5 (`p@5`) | Exact Hit@30 (`p@30`) | Group Hit@1 | Group Hit@5 | Group Hit@30 | Two-stage Group Hit@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `head_only` | 42,319 | 100,000 | 18,403 | 0.0064 | 0.0188 | 0.0589 | 0.6816 | 0.8424 | 0.9578 | 0.7122 |
| `text_lora` | 42,319 | 100,000 | 18,403 | 0.0092 | 0.0208 | 0.0565 | 0.7138 | 0.8413 | 0.9501 | 0.6884 |
| `image_lora` | 42,319 | 100,000 | 18,403 | 0.0008 | 0.0029 | 0.0111 | 0.7979 | 0.8078 | 0.9075 | 0.7903 |

Pilot runs:

| mode | sticker bank | train | val | Exact Hit@1 (`p@1`) | Exact Hit@5 (`p@5`) | Exact Hit@30 (`p@30`) | Group Hit@1 | Group Hit@5 | Group Hit@30 | Two-stage Group Hit@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `head_only` | 1,873 | 5,000 | 2,000 | 0.0160 | 0.0641 | 0.1913 | 0.7722 | 0.8398 | 0.9710 | 0.7121 |
| `text_lora` | 1,873 | 5,000 | 2,000 | 0.0255 | 0.0791 | 0.2359 | 0.7822 | 0.8553 | 0.9690 | 0.7121 |
| `image_lora` | 1,873 | 5,000 | 2,000 | 0.0095 | 0.0461 | 0.1758 | 0.7496 | 0.8923 | 0.9790 | 0.7501 |
| `dual_lora` | 1,873 | 5,000 | 2,000 | 0.0220 | 0.0736 | 0.2389 | 0.6745 | 0.8758 | 0.9720 | 0.6970 |

Interpretation: exact sticker retrieval remains very strict, especially in the full run with a 42K sticker bank. Semantic-group retrieval is much stronger, which suggests the model often retrieves stickers with the right communicative intent even when it does not recover the exact sticker id. For future revisions, the most important retrieval-facing numbers to track are Exact Hit@30 and Group Hit@30 for candidate-panel quality, plus Exact Hit@1 and Group Hit@1 if the system is expected to auto-select a single sticker.
