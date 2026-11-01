# WebShop GRPO+OPSD (beta=5) — Qwen2.5-1.5B

直接使用 [SDAR 公开 GRPO+OPSD](https://github.com/ZJU-REAL/SDAR) 实现的独立基线。
从 **Qwen2.5-1.5B-Instruct 基模**开始，不经过 Route-SFT，不注册 RETHINK，
不使用 Analyzer、局部 hint 或付费 API。默认单机 **4×A100 80GB**。

## 只改指定的门控和公共评测接线

上游固定 `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`，原入口为
`examples/grpo_opsd_trainer/run_webshop_3b.sh`。其默认 gate_beta=0；
**本版明确改为 beta=5**，保留 sdar_coef=0.01、原生 GRPO、自蒸馏、
静态 WebShop skill、reference KL 和熵正则。只将模型换为1.5B并作四卡资源适配。

这不是原论文零门控配置的逐项复现：beta=5 使用 SDAR 式置信度门控，
在其他设置相同时与其相同代码分支一致。内部 `algorithm.sdar`、`run_sdar`
和 `sdar_eval.py` 名称保留上游接口，不代表额外混入另一套训练目标。

| 项目 | 设置 |
|---|---|
| 起点 | 固定 revision 的 Qwen/Qwen2.5-1.5B-Instruct |
| Teacher | 当前 Actor 更新前的无梯度评分视角；不是固定基模或外部模型 |
| 特权信息 | 仅上游 `skills/webshop` 静态 skill；Student 与评测不读取 |
| 门控 | `sigmoid(5 * (teacher_logp - student_logp))`；Teacher/gate 均 detach |
| 蒸馏系数 | 0.01；原生有效 response token mask，不新增 action-only loss |
| 训练预算 | 150 updates；每次16任务组×8轨迹=128 episodes |
| PPO与学习率 | 全局 mini-batch64；micro8/GPU；LR1e-6 |
| 正则与惩罚 | reference KL0.01、entropy0.001、非法动作惩罚0.1 |
| 环境 | 官方1000商品＋合成目标；最多15步、history2 |
| 生成预算 | prompt4096、response512；训练温度1.0 |
| 评测 | 初始及每5步；同一固定140任务，温度0.4、top-p1、top-k不限、每题一次 |
| 保存 | 每20步及最终150；只保留最近两份完整可续训断点 |

## 与我们其他方法相同的固定140评测

`configs/fixed_validation140.json`、`configs/task_split.json`、
`configs/evaluation_protocol.json` 与现有其他 WebShop 方法**字节一致**，
不是本仓库另外抽140题。实际任务内容、顺序、worker seeds1000..1139均固定。
所有 RL 训练 worker 在 Actor 观察前排除这140个目标及价格变体；训练保留6765条原生目标。
固定任务不保证随机采样输出逐次相同。

任务清单 SHA256：
`706adb889929acc45bafd52fc144cd160b85ee7f752adf7b6dad8699f8c5ce4a`

成功仅指原生终局 score=1；平均连续 Score 独立报告，不把部分得分当成功。
这套是固定合成目标划分，不是官方 human500。独立评测也走同一原生 vLLM 入口。

## 干净服务器安装

需要 Linux x86-64、Python3.10、Java11、CUDA12.8兼容驱动与四张获配 A100 80GB。
建议至少32逻辑CPU、256–512GiB主机内存及250GiB空闲盘；断点原子写入期间
需要同时放下旧两份与新一份。这是容量预算，不是已实测的显存/内存最低值。
代码不依赖维护者电脑、F盘、私有训练数据或任何私人服务器。

```bash
git clone https://github.com/leiye302/grpo-opsd-webshop-qwen25-1.5b.git
cd grpo-opsd-webshop-qwen25-1.5b
conda create -n webshop-grpo-opsd python=3.10 openjdk=11 -c conda-forge -y
conda activate webshop-grpo-opsd
export JAVA_HOME="$CONDA_PREFIX"
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil
python src/setup.py --framework
python -m pip install -r configs/requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e .deps/sdar -c configs/requirements.txt
python -m pip check

cp .env.example .env
# 仅填写自己获配的GPU/CPU，不需要API key。
set -a
source .env
set +a
python src/assets.py --download --nlp --index --verify
python src/env_smoke.py
python src/setup.py --download-model 1.5b
python src/setup.py --verify
python scripts/check_release.py
python -m pytest tests -q
bash scripts/train.sh --dry-run
```

所有模型、数据、NLP和上游代码来自 README/配置中指定的公开固定版本，
并有 SHA/source-lock 校验。FlashAttention需要兼容wheel或本地nvcc编译。
不执行原版脚本的私人代理、wandb示例、用户路径或全机清理命令。

## 训练、续跑、评测和导出

```bash
# 从基模正式训练150次更新；使用tmux/调度器保持会话。
bash scripts/train.sh
# 确认旧作业退出后，从最近一份完整模型/优化器/RNG/数据状态恢复。
bash scripts/train.sh --resume
# 完成后导出HF权重并汇总真实指标。
python src/run.py export --size 1.5b
python src/report.py
# 用新输出目录，在完全相同的140题/采样设置上独立评测。
bash scripts/eval.sh --model "$PWD/runs/grpo_opsd_1.5b/hf150" \
  --run-dir "$PWD/runs/grpo_opsd_1.5b_final_eval"
```

`scripts/run_both.sh` 是兼容名称，只执行本仓库1.5B的训练→导出→报告。
输出位于 `runs/grpo_opsd_1.5b/`：metrics.jsonl、evaluation/update_000及每5步
的成功率/Score/实际任务身份、完整续训断点；`results/webshop.csv`与png由实际结果生成。
没有预填模型成绩，也不把最佳验证点冒充最终150步。

旧目录不能被静默覆盖。保存先写暂存目录，校验四rank模型/优化器/extra、
data.pt、tokenizer与SHA后原子发布，再轮换最近两份。缺完整断点会报错，
不会偷偷从基模重训。保留原生环境RNG机制，续跑不承诺与未中断训练逐位相同。

## 验证边界与来源

CPU测试覆盖真实上游GRPO与门控梯度、Teacher停止梯度、同Actor随更新评分、
静态skill加载、140任务身份/隔离、保存失败和恢复；GitHub Actions还会在干净
Linux/Python3.10/Java11中下载公开资产，执行原生Lucene搜索/选规格/购买与奖励检查。
这些不等于四卡CUDA/NCCL/vLLM/FSDP或完整150步训练已经跑通。

首次实际GPU运行仍需核实四rank与两个TP2副本、有限loss/grad、非零参数更新、
首次完整保存与恢复。仓库准备或CPU检查不会自动占卡或启动训练。

- [相对公开基线的精确改动](docs/BASELINE_DIFF.md)
- [原生Teacher/门控/梯度实现](docs/SDAR_METHOD.md)
- [完整数据、评测和续训合同](docs/REPRODUCTION.md)
- [所有方法共用的评测协议](docs/SHARED_EVALUATION140.md)
- [第三方来源与许可](THIRD_PARTY.md)
