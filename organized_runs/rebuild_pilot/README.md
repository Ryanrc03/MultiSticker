# rebuild_pilot

Copied from `logs/` and `results/` on 2026-04-27.

| experiment | status | p@1 | p@5 | p@30 | group/two-stage @30 |
| --- | --- | ---: | ---: | ---: | ---: |
| head_only + retrieved_topk | done | 0.0150 | 0.0566 | 0.1397 | 0.8127 |
| direct_clip + clip_context | done | 0.0000 | 0.0010 | 0.0035 | 0.9780 |
| image_lora + retrieved_topk | done | 0.0090 | 0.0336 | 0.1372 | 0.8222 |
| text_lora + retrieved_topk | done | 0.0185 | 0.0726 | 0.2118 | 0.7732 |
| dual_lora + retrieved_topk | cancelled | | | | |

`dual_lora_retrieved_topk_rebuild_pilot.log` stopped at epoch 1 image-bank re-encoding, and `rebuild_pilot_srun_8725445.log` shows the Slurm step was cancelled at 2026-04-26 22:17:05.
