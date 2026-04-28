# Rebuild Pilot 运行结果报告 (job 8725445)

运行脚本: `scripts/run_rebuild_pilot.sh` (经 `scripts/rebuild_pilot.sbatch` 提交).
驱动日志: `logs/rebuild_pilot_srun_8725445.log` — START/DONE 三个子任务都已写出.
完成时间(按子日志 mtime): head_only ≈ 19:32, direct_clip 紧随其后, image_lora 21:22.

数据切片(三个 run 共享):
- `session_count = 8129`,`provisional_samples = 368503`,`full_train_size_before_cap = 133751`.
- 经 cap/过滤后:`train = 4774`, `val = 1997`, `sticker_bank = 1873`(`candidate_sticker_count = 1954`,差额来自素材落盘 / 频次过滤).
- val 媒体分布:png 635 / gif 1111 / webm 251.

---

## 1. 关于 “没有 recall 这一指标” 的说明

源码 `src/multisticker.py:694-742` 中,`_metrics_from_scores` 与 `_group_metrics_from_scores` 都是按
`(rank_of_gold ≤ K).mean()` 计算的.这个量在 IR 文献里叫 **HitRate@K**;而在「每条 query 只有 1 个 gold」的设置下,
它**恰好等于 Recall@K**(因为 Recall = 命中数/相关总数 = {0,1}/1 = HitRate).

所以现有 JSON 里的所有 `p@K` 实质上就是 Recall@K,只是命名沿用了 "precision@K" 的旧叫法.下面表格我直接以
`Recall@K (= HitRate@K)` 重新解读.

唯一**没有**直接等价的,是 group/semantic 评测下的「真·多正样本 Recall@K」
(即 `#group_stickers_in_topK / #group_stickers_in_bank`).当前代码只看
"gold 所在 group 的任一成员是否进 top-K",这是一种 *any-hit* 召回,不是 micro/macro 多正样本召回.
要做后者必须保留 score matrix 重算 — score matrix 当前没有落盘,需要重跑 eval 才能算出.
本报告末尾给出实施建议;如果你只关心「能不能命中正确的一张/一组贴纸」,现有数字已经够用.

---

## 2. 三种方法在 val (1997 样本) 上的结果

### 2.1 Exact-sticker Recall@K (单正样本,gold = 那张特定贴纸,1873 候选)

| 方法                                  | R@1    | R@3    | R@5    | R@10   | R@30   | MRR    |
|---------------------------------------|--------|--------|--------|--------|--------|--------|
| direct_clip (clip_context, zero-shot) | 0.0000 | 0.0010 | 0.0010 | 0.0010 | 0.0035 | 0.0024 |
| head_only + retrieved_topk            | 0.0150 | 0.0376 | 0.0566 | 0.0761 | 0.1397 | 0.0401 |
| image_lora + retrieved_topk           | 0.0050 | 0.0090 | 0.0125 | 0.0175 | 0.0481 | 0.0145 |

### 2.2 Group/Semantic Recall@K (any-hit:gold group 的任一贴纸进 top-K)

| 方法                                            | R@1    | R@3    | R@5    | R@10   | R@30   | MRR    |
|-------------------------------------------------|--------|--------|--------|--------|--------|--------|
| direct_clip (clip_context)                      | 0.5689 | 0.8202 | 0.8748 | 0.9264 | 0.9780 | 0.7035 |
| head_only — semantic                            | 0.7636 | 0.8257 | 0.8338 | 0.8618 | 0.9539 | 0.8059 |
| head_only — fused (group prior, α=2)            | 0.7126 | 0.7241 | 0.7316 | 0.7376 | 0.7581 | 0.7230 |
| head_only — two-stage rerank                    | 0.7066 | 0.7166 | 0.7216 | 0.7271 | 0.7391 | 0.7150 |
| image_lora — semantic                           | 0.7176 | 0.8363 | 0.8643 | 0.8888 | 0.9730 | 0.7865 |
| image_lora — fused                              | 0.6630 | 0.6800 | 0.6930 | 0.7066 | 0.7341 | 0.6780 |
| image_lora — two-stage rerank                   | 0.6610 | 0.6760 | 0.6865 | 0.6960 | 0.7126 | 0.6732 |

### 2.3 媒体分桶 (group recall@30,纯 retrieval 分支)

| 方法        | png (n=635) | gif (n=1111) | webm (n=251) |
|-------------|-------------|--------------|--------------|
| head_only   | 0.9795      | 0.9433       | 0.9363       |
| image_lora  | 0.9701      | 0.9775       | 0.9602       |

---

## 3. 训练曲线

### head_only (`logs/head_only_retrieved_topk_rebuild_pilot.log`)

| epoch | train_loss | val_p@1 | val_p@5 | val_p@30 | val_group_p@30 | val_two_stage_group_p@30 |
|-------|-----------|---------|---------|----------|----------------|--------------------------|
| 1     | 7.1244    | 0.0035  | 0.0330  | 0.1407   | 0.9574         | 0.8127 ★                  |
| 2     | 6.4462    | 0.0150  | 0.0566  | 0.1397   | 0.9539         | 0.7391                    |

best_val_two_stage_group_p@30 = **0.8127** (epoch 1).Exact 指标在 epoch 2 还在升,但 two-stage rerank 反而下降,
说明 group prior / rerank 这一支跟着主分支变化的鲁棒性较弱(只 2 个 epoch,信号有限).

### image_lora (`logs/image_lora_retrieved_topk_rebuild_pilot.log`)

| epoch | train_loss | val_p@1 | val_p@5 | val_p@30 | val_group_p@30 | val_two_stage_group_p@30 |
|-------|-----------|---------|---------|----------|----------------|--------------------------|
| 1     | 4.3512    | 0.0085  | 0.0361  | 0.1412   | 0.9680         | 0.8222 ★                  |
| 2     | 3.9138    | 0.0050  | 0.0125  | 0.0481   | 0.9730         | 0.7126                    |

best = **0.8222** (epoch 1).image_lora 第二个 epoch exact recall 明显回退(R@30 从 0.1412 → 0.0481),
group recall 反而升,典型「LoRA 把 image 表达拉得更贴 group 形态、损失个体辨识度」的迹象.

---

## 4. 结果解读

1. **direct CLIP zero-shot 不能做 exact 检索** (R@30 只有 0.0035);但 group recall 已经很高
   (R@30 = 0.978),说明 CLIP image-text 对齐足以把 query 拉到「相关一类贴纸」,但区分同类内的具体成员需要训练.
2. **head_only > image_lora 在 exact** (R@30: 0.1397 vs 0.0481);**image_lora 略优在 group**
   (best two-stage 0.8222 vs 0.8127).只跑 2 epoch、5k train,LoRA 的拟合优势没有体现;
   而 head_only 由于参数少 + 冻结 backbone,在小数据上反而稳.
3. **fused / two-stage rerank 比 semantic-only 差**:semantic_metrics 的 group R@30 ≈ 0.95,
   而 fused / two-stage 都掉到 0.71–0.76.group prior α=2 在当前训练量下过强,把 retrieval 分布拍扁了.
   建议:扫一下 α ∈ {0.0, 0.5, 1.0, 2.0} 或干脆只用 semantic 分支报数.
4. **Epoch 间不稳定**:两个训练 run 的 best 都是 epoch 1.要么早停,要么把 epoch 提到 ≥5 看真曲线.

---

## 5. 关于「再算一下 recall」的可执行建议

按上文,exact 那部分已经就是 Recall@K,不需要重算;只重命名展示即可(本报告 §2.1 已经做了).

如果你要的是 **多正样本 group Recall@K**(分母 = bank 中该 group 的所有贴纸),需要做:

1. 在 `src/multisticker.py` 加一个 `_group_recall_at_k(score_matrix, gold_indices, sticker_group_ids)`,
   对每条 query 算 `|topK ∩ gold_group| / |gold_group_in_bank|`,再求均值.
2. 把它加进 `evaluate_split` 返回字典,并在 `train_am.py` 输出端落盘.
3. 重跑 eval(无需重训)— 但当前没有保存 score matrix,所以需要从 checkpoint 重新前向一遍 val,
   流程跟 evaluate_split 内部一样.建议加一个 `scripts/eval_only.py`,加载
   `results/am_*_rebuild_pilot.pt` + manifest,跑一次 val 即可.

要我直接动手把这两步实现并跑一次 eval-only 吗?
