<p align="center">
  <font size="6"><b>Capability Self-Assessment:<br>Teaching LLMs to Know Their Limits</b></font>
</p>
<p align="center">
<a href="#"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg"></a>
<a href="#"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
<a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white"></a>
</p>

<!-- =============================================================== -->
<!-- TEASER / HERO FIGURE                                            -->
<!-- Replace `assets/teaser.png` with your promotional figure.       -->
<!-- Suggested: a single eye-catching image that conveys the core    -->
<!-- idea of CSA at a glance (e.g. the SELF_SOLVE / DELEGATE         -->
<!-- routing intuition, or headline results).                        -->
<!-- =============================================================== -->
<p align="center">
  <img src="figures/teaser_figure.png" alt="CSA teaser figure" width="100%">
</p>

> **TL;DR.** We define **Capability Self-Assessment (CSA)** as a model's ability to judge whether a query falls within its own solvable set, formulated as a binary policy choice between **`SELF_SOLVE`** (attempt the query) and **`DELEGATE`** (defer to a stronger system). Across model families and scales, current LLMs systematically overestimate themselves. We show that CSA is *teachable*, that **RLVR** (reinforcement learning with verifiable rewards) injects it more effectively than supervised fine-tuning, and that the learned behavior preserves the model's underlying problem-solving ability and transfers across domains.

---

## 📁 Repository Structure


```
csa-llm/
├── src/                         # All Python source (invoked as `python -m src.<pkg>.<mod>`)
│   ├── data/                    # Dataset construction, answer generation, grading,
│   │                            # analysis generation, SFT data assembly
│   ├── csa/                     # CSA inference (local vLLM + closed APIs),
│   │                            # CSA evaluation, Capability Ratio computation
│   ├── training/                # SFT, GRPO, and DFW
│   ├── hub/                     # HuggingFace upload helpers
│   └── utils/                   # Shared: prompts, parsing, vLLM, IO, grading
│
├── scripts/                     # Shell launchers + YAML configs
│   ├── data/                    # build_dataset_* / generate_answers* /
│   │                            # generate_analysis_{self,teacher} / build_sft_dataset
│   ├── inference/               # inference_local / inference_api
│   ├── evaluation/              # grade_math / grade_science /
│   │                            # evaluate_csa / capability_ratio
│   ├── training/                # SFT / GRPO YAML configs + run_sft / run_dfw / run_grpo
│   │                            # + accelerate (DeepSpeed ZeRO-3) configs
│   └── hub/                     # upload_dataset / upload_model
│
├── dataset/                     # Pre-built benchmark datasets, ready to use
│   ├── math/                    # GSM8K + MATH-500 + AIME
│   └── science/                 # MMLU-Pro (bio / chem / health / physics)
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup

```bash
git clone https://github.com/Joyyang158/csa-llm.git
cd csa-llm
pip install -r requirements.txt
```

Environment variables (only set the ones you need):

| Variable | Used by |
| --- | --- |
| `HF_TOKEN` | `scripts/hub/upload_*.sh` |
| `TOGETHER_API_KEY` | `scripts/data/generate_answers_api.sh`, `scripts/data/generate_analysis_teacher.sh` |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY` | `scripts/inference/inference_api.sh` |
| `WANDB_API_KEY` | training |

---

## 🧠 Method Overview

Our framework has three stages. **① CSA Label Construction** probes the model to derive per-query `SELF_SOLVE` / `DELEGATE` labels. **② CSA Training Strategies** instills CSA into the model via one of four approaches: three SFT variants and RLVR. **③ CSA Inference & Evaluation** lets the trained model decide on new queries, and verifies both decision quality and that the model's problem-solving ability is preserved.

<!-- =============================================================== -->
<!-- METHOD FIGURE                                                   -->
<!-- Replace `assets/method.png` with the method overview figure     -->
<!-- (the three-stage pipeline: label construction → training        -->
<!-- strategies → inference & evaluation).                           -->
<!-- =============================================================== -->
<p align="center">
  <img src="figures/method_figure.jpg" alt="CSA method overview" width="100%">
</p>

The four training strategies in Stage ②. The table below shows the **output format** (what the model emits at inference time) and whether each strategy requires a **supervision rationale generation** pass before training begins.

| Strategy | Output format | Needs supervision rationale generation? |
| --- | --- | --- |
| **(a) SFT_label** | `<decision>` only | No |
| **(b) SFT_self** | `<analysis>` + `<decision>` | Yes. Rationales come from the training model itself |
| **(c) SFT_teacher** | `<analysis>` + `<decision>` | Yes. Rationales come from a stronger teacher model |
| **(d) RLVR** | `<analysis>` + `<decision>` | No. Rationales emerge during RL rollouts |

---

## 🚀 Quick Start


All local inference paths go through `vllm.LLM` (the HuggingFace `AutoModelForCausalLM` backend appears only inside `src/training/`, where the trainer needs the model directly). Both SFT and GRPO use full-parameter fine-tuning with DeepSpeed ZeRO-3 (no LoRA), and the accelerate configs live in `scripts/training/`. The sections below follow the three stages above. Every Python entry point is invoked as a module (`python -m src.<package>.<module>`); the shell scripts under `scripts/` are thin launchers around them.

### ① CSA Label Construction

We evaluate on two domains:

* **math**: GSM8K, MATH-500, AIME. Open-ended; final answers in `\boxed{...}`.
* **science**: MMLU-Pro restricted to biology / chemistry / health / physics. Multiple-choice; each test query is sampled with multiple option-shuffles and aggregated by majority vote.

Pre-built benchmarks for both domains are provided under [`dataset/`](dataset/). Both `dataset/math/` and `dataset/science/` already contain queries, gold answers, 5-sample rollouts from the target model, and the derived per-row `SELF_SOLVE` / `DELEGATE` label (`is_correct` column). For math we use the any-correct rule; for science we use the majority-correct rule with per-sample option shuffling. Both rules live in `src/utils/grading.py`, so the same logic is reused at evaluation time.

If instead you want to **change the dataset composition** (for example, adjust the per-source ratio, swap in a different base model to probe, or add a new benchmark), the three steps below let you rebuild the data from scratch.

**Step 1: Build raw benchmark splits from upstream sources.**

```bash
bash scripts/data/build_dataset_math.sh
bash scripts/data/build_dataset_science.sh
```

**Step 2: Collect 5 samples per query from the target model** (vLLM-backed; use `generate_answers_api.sh` for hosted models).

```bash
bash scripts/data/generate_answers.sh
```

**Step 3: Grade the samples to derive the per-row `is_correct` label.**

```bash
bash scripts/evaluation/grade_math.sh
bash scripts/evaluation/grade_science.sh
```

### ② CSA Training Strategies

Pick one of the four strategies below.

#### (a) SFT_label: bare label, no rationale

Set `sft_mode: label` in `scripts/training/sft.yaml`, then:

```bash
bash scripts/training/run_sft.sh
```

#### (b) SFT_self: self-generated rationale + label

First have the training model generate rationales for itself (conditioned on the ground-truth label), then assemble the SFT CSV:

```bash
bash scripts/data/generate_analysis_self.sh
bash scripts/data/build_sft_dataset.sh
```

Set `sft_mode: self` in the SFT config and run:

```bash
bash scripts/training/run_sft.sh
```

#### (c) SFT_teacher: teacher-distilled rationale + label

Same as (b), but rationales come from a stronger teacher model (requires `TOGETHER_API_KEY`):

```bash
bash scripts/data/generate_analysis_teacher.sh
bash scripts/data/build_sft_dataset.sh
```

Set `sft_mode: teacher` and run:

```bash
bash scripts/training/run_sft.sh
```

#### (d) RLVR: two-stage GRPO with Diversity-Filtered Warm-up

**Phase 1: DFW (Diversity-Filtered Warm-up).** `run_dfw.sh` first retains only queries whose K=16 rollouts contain *both* `SELF_SOLVE` and `DELEGATE` to construct a diversified subset *D*<sub>div</sub>, then runs GRPO on it. This rescues the within-group reward variance that would otherwise vanish, since base models predict `SELF_SOLVE` on nearly every query. DFW is implemented as a self-contained stage so the resulting subset is inspectable and reusable across experiments without re-sampling.

```bash
bash scripts/training/run_dfw.sh
```

**Phase 2: Full GRPO.** Continue GRPO training on the full dataset, starting from the warm-up checkpoint:

```bash
bash scripts/training/run_grpo.sh
```

For OLMo-2 models, skip Phase 1. Their rollouts are already diverse enough that DFW is unnecessary, so run only Phase 2 on the full dataset.

### ③ CSA Inference & Evaluation

**Inference.** Run the trained model on the held-out test split:

```bash
bash scripts/inference/inference_local.sh
```

The model emits an `<analysis>` block followed by a `<decision>` (`SELF_SOLVE` or `DELEGATE`). Parsing is in `src/utils/parsing.py:parse_decision`.

**Evaluation.** After training, the model's underlying problem-solving ability may have shifted, so the original `is_correct` labels (probed from the model *before* training) no longer reflect what the model can actually solve *after* training. Both evaluations below therefore start from the same prerequisite: re-generating answers with the trained model on the test split.

```bash
# Re-generate 5 samples per test query using the model after training
bash scripts/data/generate_answers.sh
```

**(a) CSA quality.** Does the model make the right `SELF_SOLVE` / `DELEGATE` decisions? `src/csa/evaluate.py` reports **CDS** (Capability Discrimination Score) and **M-F1** as the main metrics, with Accuracy and SSR also reported as references.

The script supports two grading modes via a single `--grade_mode` switch. The two modes differ only in which `is_correct` labels they grade against:

* **`generations` (default, recommended).** Grades against the answers freshly re-generated above, so `is_correct` reflects the model's ability *after* training. This is what you want for honest evaluation, because CSA training may have shifted that ability.
* **`column`.** Uses the `is_correct` column already in the predictions CSV, which reflects the model's ability *before* training. Provided as an option if you specifically want to compare against the original snapshot.

```bash
# Recommended: grade against the model's ability after training
bash scripts/evaluation/evaluate_csa.sh   # GRADE_MODE=generations

# Optional: grade against the original (before-training) labels
bash scripts/evaluation/evaluate_csa.sh   # GRADE_MODE=column
```

**(b) Capability Ratio (CR).** Does CSA training preserve the model's underlying problem-solving ability? CR is the ratio of solve accuracy *after* training to solve accuracy *before* training on the same evaluation set, computed from the re-generated answers above.

```bash
bash scripts/evaluation/capability_ratio.sh
```

<!-- ---

## 📚 Citation

To be added on de-anonymisation.

```bibtex
@article{csa2026,
  title   = {Capability Self-Assessment: Teaching LLMs to Know Their Limits},
  author  = {Anonymous},
  year    = {2026}
}
``` -->