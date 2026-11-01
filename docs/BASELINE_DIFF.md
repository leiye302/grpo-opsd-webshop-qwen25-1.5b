# Exact differences from public GRPO+OPSD

Pinned source: ZJU-REAL/SDAR `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`.
Reference: `examples/grpo_opsd_trainer/run_webshop_3b.sh`.

## Named changes

1. **Gate beta0 ->5**, with unchanged native compute_sdar_loss, detach, mask,
   aggregation, coefficient0.01 and Actor backward path. beta0 is uniform0.5;
   beta5 depends on the Teacher-Student sampled-token log-probability gap.
   This is a requested gated variant, not the paper's exact zero-gate baseline.
2. **Qwen2.5-1.5B-Instruct** in place of the public launcher's3B backbone.
   The model/revision and complete weight receipt are pinned before training.
3. **The same shared140 evaluation** as all other local-method releases:
   ordered content IDs, worker seeds1000..1139, temperature0.4, top-p1,
   unrestricted top-k, one rollout;15 steps/history2/prompt4096/response512.
   All matching held-out goal variants are excluded from every RL training worker.

## Execution-only changes

Four FSDP ranks, two TP2 replicas, preserving global mini64 and16x8 rollouts.
Portable paths/public assets, console logging,150-update stop, evaluation every5,
complete checkpoints every20/final150, transactional verification and newest2
retention. No private paths, API, external Teacher, Analyzer, RETHINK, Route-SFT,
action-only mask, new reward, new optimizer or new distillation loss is added.

Only two native files are patched: the validation/checkpoint telemetry Trainer
import and the actual content-disjoint WebShop worker factory/index range.
The native GRPO, sdar_utils, Actor, current-Actor Teacher builder, static skill
selector, prompt, parser, reward and environment engine remain SHA-locked.

At beta5 this uses the same confidence-gated code branch as SDAR. For publication
label it GRPO+OPSD (beta=5), and do not claim its training objective is distinct
from an otherwise identically configured SDAR run. A paper-default beta0 control
would be a separate, explicitly configured experiment, not this entry point.
