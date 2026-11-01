# Exact native self-distillation implementation, with beta=5

Source: public ZJU-REAL/SDAR, pinned in configs/upstream.json at
29dcb12a55d8fc17ba6ae03fd0a69048f43051f3.
Launcher: examples/grpo_opsd_trainer/run_webshop_3b.sh.
The loss is the released SDAR implementation. The public GRPO+OPSD launcher defaults to beta0; this baseline explicitly uses beta5.

## Observation -> rollout -> Teacher scoring -> update

1. The Student Actor sees only the native WebShop instruction and recent
   observation/action history. It freely generates the original think/action
   response, which the native parser executes. There is no skill, hint,
   Analyzer, RETHINK token, forced prefix or replacement action at rollout.
2. Original environment rewards and native GRPO group-normalized advantages
   are computed from the same collected episodes. The native SDAR training
   loop does not replace them with RLSD token-weighted advantages.
3. The original SkillProvider loads general_skills.md and the mapped static
   WebShop skills. For text data it can use native prompt keyword selection;
   multiple keyword matches include multiple skill texts, general text once.
   Native gamefile/data_source fallback behavior is retained exactly.
   There are eight skill markdown files, not a trainable Analyzer or API.
4. build_teacher_batch in rlsd_ray_trainer.py decodes the Student prompt,
   prepends the literal text `[Privileged Skill Information]`, the selected
   skill and blank separators, then re-tokenizes that Teacher-only prompt.
   It appends the **identical original response token IDs and response mask**.
   This is the original text-prefix implementation, not a new chat-message schema.
5. The same actor_rollout_wg computes Teacher log probabilities with current
   Actor weights before the Actor update. The Teacher is not a fixed base
   model, EMA copy, external model or a generator of corrected trajectories.
   At the next update it follows the updated Actor. The base RefPolicy is
   separate and fixed only for the original reference-KL regularizer.
6. The original dp_actor.update_policy adds gated SDAR to GRPO and keeps the
   reference KL, entropy term, mask, token-mean reduction, minibatching,
   gradient accumulation, optimizer and scheduler unchanged.

## Exact sampled-token loss and gradient

For already sampled response token y_t, let s_t be the normal Student prefix,
r the static skill, and sg denote stop-gradient:

\[
\ell_{S,t}=\log\pi_\theta(y_t\mid s_t),\qquad
\ell_{T,t}=\log\pi_\theta(y_t\mid r,s_t).
\]

The public compute_sdar_loss computes

\[
\Delta_t=\operatorname{sg}(\ell_{T,t}-\ell_{S,t}),\qquad
g_t=\operatorname{sg}\!\left[\sigma(5\Delta_t)\right],
\]

\[
\mathcal L_{\mathrm{SDAR}}=
\frac{\sum_t m_t g_t[\operatorname{sg}(\ell_{T,t})-\ell_{S,t}]}{\sum_t m_t}.
\]

Thus the Student log-probability gradient is -m_t*g_t/sum(m); neither Teacher
probabilities nor the gate receive a gradient. The mask covers original valid
response tokens, not a newly introduced route/action-only region. A gate
above0.5 is telemetry, not a hard routing threshold. This is the upstream
sampled-token surrogate, not a newly substituted full-vocabulary KL objective.

For the released WebShop settings the total objective is

\[
\mathcal L=\mathcal L_{\mathrm{GRPO}}+0.01\mathcal L_{\mathrm{SDAR}}
+0.01\mathcal L_{\mathrm{reference\text{-}KL}}-0.001\mathcal H.
\]

The original invalid-action penalty0.1 is retained in reward processing.
The default native skillsd/sdl_lambda diagnostic is not SDAR's loss coefficient:
main_sdar sets use_sdar_loss and sdar_loss_coef=.01, and the Actor SDAR branch
uses that coefficient directly. No SDL warmdown is added to SDAR.

## Native Teacher truncation is not silently repaired

Student data processing retains data.truncation=error and prompt4096/response512.
However, the **original Teacher builder always left-truncates its augmented
prompt to4096**, even when its truncation argument says error. That can remove
part of a skill or older prompt context. This behavior is left unchanged to
preserve the released SDAR method; the CPU tests explicitly cover overflow,
response/EOS/padding alignment and non-mutation of Student inputs.

## Shared fixed140 evaluation and exclusions

The two SDAR sizes and the corresponding GRPO releases use byte-identical
configs/task_split.json and configs/fixed_validation140.json. Each category
has28 tasks; target ASINs are distinct. Training filters the content IDs before
Actor interaction, including automatic resets, retaining6765 other native goal
entries. Randomized price variants cannot bypass the same-goal exclusion.
This is a custom objective-held-out split, not canonical WebShop human500 or
unseen-product zero-shot evaluation. No skill or Teacher scoring is used at eval.

Training remains150 updates,16 groups x8 episodes/update, LR1e-6, globalmini64,
micro8/GPU, history2 and max15 environment steps. The original two-GPU SDAR
launcher is bound to four FSDP ranks/two TP2 replicas without changing the
global episode/minibatch budget. Validate all140 tasks at0/every5/final150;
save20/final150, keeping the latest two SHA-verified full checkpoints.

## What has and has not been verified

Public CPU CI exercises original GRPO and SDAR gradients, the actual Actor
SDAR-addition branch, current-Actor Teacher calls, static skill selection,
response alignment, fixed task identity/exclusion, checkpoint failures/resume,
clean public dependency/data installation and real WebShop search/buy/reward.
It is not proof of CUDA memory fit, NCCL, vLLM/FSDP startup or150-update results.
No model scores are prefilled and no GPU training is launched by setup or tests.
