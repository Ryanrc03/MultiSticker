# rebuild_medium

Copied from `logs/` and `results/`; refreshed after the 2026-04-27/2026-04-28 LoRA补跑完成.

| experiment | status | recall@1 | recall@5 | recall@30 | group recall@30 |
| --- | --- | ---: | ---: | ---: | ---: |
| head_only + retrieved_topk | done | 0.0110 | 0.0296 | 0.1263 | 0.9414 |
| direct_clip + clip_context | done | 0.0002 | 0.0012 | 0.0028 | 0.9588 |
| image_lora + retrieved_topk | done | 0.0108 | 0.0376 | 0.1188 | 0.9610 |
| text_lora + retrieved_topk | done | 0.0102 | 0.0312 | 0.1190 | 0.9444 |
| dual_lora + retrieved_topk | done | 0.0100 | 0.0326 | 0.1315 | 0.9602 |

`rebuild_medium_8727908.log` stopped after `image_lora` with a shell unbound-variable error. The missing `text_lora` and `dual_lora` rows were completed later as separate Slurm jobs (`8743469` and `8743470`).
