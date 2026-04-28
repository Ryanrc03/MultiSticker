# Medium Run Report

## 1. 实验设计

本轮 `rebuild_medium` 目标是在同一份 medium 数据切分上比较 5 个检索方案：

| 方法 | 说明 |
| --- | --- |
| `direct_clip` | 不训练，直接用 CLIP 文本查询匹配 sticker 图像 embedding，作为 zero-shot baseline。 |
| `head_only` | 冻结 CLIP，只训练检索 head / intent head。 |
| `image_lora` | 训练图像侧 CLIP LoRA + 检索 head，文本侧冻结。 |
| `text_lora` | 训练文本侧 CLIP LoRA + 检索 head，图像侧冻结。 |
| `dual_lora` | 同时训练图像侧和文本侧 CLIP LoRA + 检索 head。 |

训练时，`head_only` 只更新检索 head / intent head；LoRA 方案在对应的 CLIP 分支注入低秩适配器，并和检索 head 一起优化。训练目标由 sticker 检索交叉熵和 intent 分类辅助损失组成。主要评价两个层面：

- **Exact sticker retrieval**: gold sticker 是否出现在 top-k 中，是最终推荐贴纸的核心指标。
- **Semantic/group retrieval**: top-k 是否命中同一语义组，衡量语义方向是否对，但不等价于精确贴纸命中。

## 2. 参数

| 项目 | 值 |
| --- | --- |
| run name | `rebuild_medium` |
| train samples | 30,000 |
| val samples | 5,000，实际评估 `sample_count=4,998` |
| sticker bank | 6,758 |
| candidate sticker count before filtering | 7,011 |
| supported media | `.png`, `.gif`, `.webm` |
| min sticker frequency | 2 |
| seed | 42 |
| CLIP backbone | `ViT-B-32` |
| CLIP pretrained | `laion2b_s34b_b79k` |
| epochs | 5 |
| infer batch size | 256 |
| head batch size | 256 |
| LoRA batch size | 96 |
| intent clusters | 64 |
| hidden dim | 512 |
| dropout | 0.1 |
| temperature | 0.07 |
| weight decay | 0.0001 |
| intent loss weight | 0.2 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA lr | 0.0001 |
| head lr | 0.001 |

## 3. 结果

### Overall

| 方法 | Exact R@1 | Exact R@5 | Exact R@30 | MAP/MRR | Group R@1 | Group R@5 | Group R@10 | Group R@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_clip` | 0.0002 | 0.0012 | 0.0028 | 0.0013 | 0.6335 | 0.8778 | **0.9084** | **0.9588** |
| `head_only` | **0.0110** | 0.0296 | 0.1263 | 0.0282 | **0.7933** | 0.8619 | 0.8780 | 0.9406 |
| `image_lora` | 0.0108 | **0.0376** | 0.1188 | **0.0302** | 0.7063 | 0.8812 | 0.9044 | 0.9550 |
| `text_lora` | 0.0102 | 0.0312 | 0.1190 | 0.0280 | 0.7623 | 0.8661 | 0.8880 | 0.9420 |
| `dual_lora` | 0.0100 | 0.0326 | **0.1315** | 0.0286 | 0.6275 | **0.8816** | 0.9064 | 0.9540 |

### PNG 分项

| 方法 | Count | Exact R@1 | Exact R@5 | Exact R@30 | MAP/MRR | Group R@1 | Group R@5 | Group R@10 | Group R@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `head_only` | 1,676 | 0.0048 | 0.0197 | **0.1062** | 0.0204 | **0.8896** | 0.9141 | 0.9242 | 0.9594 |
| `image_lora` | 1,676 | 0.0060 | 0.0185 | 0.0752 | 0.0188 | 0.8371 | 0.9254 | 0.9391 | **0.9636** |
| `text_lora` | 1,676 | **0.0072** | 0.0286 | 0.0990 | 0.0238 | 0.8532 | 0.9206 | 0.9314 | 0.9582 |
| `dual_lora` | 1,676 | **0.0072** | **0.0304** | 0.0973 | **0.0247** | 0.5823 | **0.9320** | **0.9403** | 0.9624 |

### GIF 分项

| 方法 | Count | Exact R@1 | Exact R@5 | Exact R@30 | MAP/MRR | Group R@1 | Group R@5 | Group R@10 | Group R@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `head_only` | 2,284 | **0.0201** | 0.0499 | 0.1935 | 0.0456 | **0.7583** | 0.8695 | 0.8870 | 0.9422 |
| `image_lora` | 2,284 | 0.0188 | **0.0670** | 0.1948 | **0.0498** | 0.6335 | **0.8927** | 0.9168 | **0.9645** |
| `text_lora` | 2,284 | 0.0166 | 0.0468 | 0.1843 | 0.0426 | 0.7369 | 0.8717 | 0.8971 | 0.9461 |
| `dual_lora` | 2,284 | 0.0158 | 0.0464 | **0.2062** | 0.0418 | 0.6681 | 0.8892 | **0.9177** | 0.9549 |

### WEBM 分项

| 方法 | Count | Exact R@1 | Exact R@5 | Exact R@30 | MAP/MRR | Group R@1 | Group R@5 | Group R@10 | Group R@30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `head_only` | 1,038 | 0.0010 | 0.0010 | 0.0106 | 0.0028 | **0.7148** | 0.7611 | 0.7832 | 0.9066 |
| `image_lora` | 1,038 | 0.0010 | 0.0039 | **0.0222** | 0.0054 | 0.6551 | **0.7842** | 0.8208 | 0.9200 |
| `text_lora` | 1,038 | 0.0010 | 0.0010 | 0.0077 | 0.0027 | 0.6715 | 0.7659 | 0.7977 | 0.9066 |
| `dual_lora` | 1,038 | **0.0019** | **0.0058** | **0.0222** | **0.0060** | 0.6108 | 0.7832 | **0.8266** | **0.9383** |

## 4. 分析

### 哪个最好

本任务更接近 chat 场景里的 sticker recommendation：屏幕上通常推荐 5-10 个候选，用户不一定要求命中某一个 exact sticker ID，但希望候选整体语义、情绪和互动意图是对的。因此这里把 **Group R@5 / Group R@10** 作为北极星指标，exact retrieval 作为副指标，用来观察模型是否能进一步区分同一语义组里的具体 sticker。

从这个口径看：

- Group R@5 最好的是 `dual_lora`，整体 **0.8816**，但只比 `image_lora` 的 0.8812 高 0.0004，差距很小。
- Group R@10 最高的是 `direct_clip`，整体 **0.9084**；训练模型里 `dual_lora` 最高，为 **0.9064**，接近 `direct_clip`。
- 分媒体看，PNG 的 Group R@5/R@10 是 `dual_lora` 最好；GIF 的 Group R@5 是 `image_lora` 最好、Group R@10 是 `dual_lora` 最好；WEBM 的 Group R@5 是 `image_lora` 最好、Group R@10/R@30 是 `dual_lora` 最好。

因此，**`dual_lora` 是最均衡的训练方案**：它在 Group R@5 上整体第一，在训练模型中 Group R@10 也第一，并且分媒体结果也比较稳定。`direct_clip` 的 Group R@10/R@30 最高，说明 zero-shot CLIP 本身已经很擅长抓语义组；但它的 Exact R@5/R@30 极低，只适合做语义 baseline，不适合作为最终推荐模型。

Exact 指标作为副指标看，`image_lora` 的 Exact R@5 = **0.0376**、MAP/MRR = **0.0302** 最高，说明它更擅长把具体 sticker 排到前面；`dual_lora` 的 Exact R@30 = **0.1315** 最高，说明它更擅长扩大候选池覆盖。对于只做 retrieval 的阶段，这两个现象都可以接受：先保证 5-10 个推荐的语义组正确，再在后续 rerank 或 UI 展示中优化具体 sticker 排序。

`head_only` 的 Exact R@1 最高但优势很小，Group R@5/R@10 又不如 `dual_lora` 和 `image_lora`，因此更像一个强 baseline，而不是最终推荐方案。

### 可能原因

`direct_clip` 的 exact retrieval 很弱，但 group R@10/R@30 很高。这说明原始 CLIP 能大致找到语义相近的 sticker 类别；这对 chat 推荐是有价值的，因为用户看到的是一组可选 sticker。但它无法在 6,758 个具体 sticker 中区分非常相似的表情包变体，所以 exact 指标很低。

LoRA 的作用主要体现在让模型更贴近当前 sticker 语料。`dual_lora` 同时调整文本和图像空间，因此 group@5-10 更均衡；`image_lora` 对 GIF 特别有效，可能因为图像侧适配能更好地区分 sticker 的视觉风格、角色和动图帧特征，所以它的 Exact R@5 和 MAP/MRR 更强。

PNG 上 `head_only` 的 R@30 最高，而 LoRA 没有明显全面胜出，可能说明静态图像在 frozen CLIP 空间里已经比较稳定，额外微调收益有限，甚至会牺牲部分原有语义结构。WEBM 是最难的媒体类型，所有模型 exact 指标都很低；它可能受帧采样、视频动态信息压缩和样本稀疏影响更大。

综合来看，如果目标是 chat UI 里的 5-10 个推荐候选，推荐把 **Group R@5/R@10 作为主指标**，选择 `dual_lora` 作为主模型；exact 指标作为副指标继续跟踪即可。后续可以用 `dual_lora` 产出 group-aware top-k candidate，再用更细粒度的视觉/文本 reranker 或多样性规则优化具体 sticker 排序。
