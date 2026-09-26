<h1 align="center">Capability Self-Assessment in Large Language Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2606.00251"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="#setup"><img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white"></a>
</p>

<p align="center">
  <img src="figures/teaser_figure.png" alt="CSA teaser figure" width="100%">
</p>

Code for learning whether a language model should `SELF_SOLVE` a query or
`DELEGATE` it, while measuring how training changes its underlying solving
ability. We compare label-only SFT, self-analysis SFT, teacher-analysis SFT,
and RLVR using GRPO with an optional diversity-filtered warm-up (DFW).

The current manuscript uses the title above. The linked arXiv record was
posted as **Capability Self-Assessment: Teaching LLMs to Know Their Limits**;
the citation below retains that record's title.

## Setup

Run commands from the repository root. Local inference and training require a
Linux environment with CUDA GPUs and a compatible PyTorch/vLLM/TRL/DeepSpeed
installation. CPU-only grading and regression tests need only pandas and NumPy.

```bash
git clone https://github.com/Joyyang158/llm-csa.git
cd llm-csa
python -m pip install -r requirements.txt
```

`requirements.txt` is not a lockfile for the original experiments. Save the
resolved environment alongside each run (`python -m pip freeze`) and record
model revisions, configuration files, and seeds. The SFT trainer uses TRL's
[`max_length` and `completion_only_loss` configuration](https://huggingface.co/docs/trl/sft_trainer).

Set `HF_TOKEN` for gated model downloads or optional Hub uploads,
`TOGETHER_API_KEY` for teacher analysis, and `WANDB_API_KEY` when using W&B.
API inference uses the selected provider's credential (`OPENAI_API_KEY`,
`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, or `TOGETHER_API_KEY`).

## Method Overview

Our framework has three stages: construct capability labels for the initial
model, train its self-assessment policy, and evaluate both its decisions and
its post-training problem-solving ability.

<p align="center">
  <img src="https://raw.githubusercontent.com/Joyyang158/llm-csa/dff952cc7bae604b0c8a8c5aea45b407da7c6d76/figures/method_figure.jpg" alt="CSA method overview" width="100%">
</p>

## Data and labels

The released benchmark splits are under `dataset/`:

| Domain | Sources | Train | Test |
| --- | --- | ---: | ---: |
| Math | GSM8K, MATH500, AIME | 2,260 | 490 |
| Science | MMLU-Pro biology, chemistry, health, physics | 3,266 | 700 |

These files contain questions and gold answers, **not model-specific capability
labels**. Construct labels separately for each initial model. Science uses
`options` lists; training also accepts an already formatted `choices` column.

For the paper's `K=5` protocol, a Math capability label is positive when **any**
attempt is correct. A Science label is positive when **at least three** attempts
are correct, with options independently shuffled and the gold letter remapped
for each attempt. This is majority-correct grading, not a vote over answer
strings. Training labels stay fixed; evaluation labels are recomputed for
each trained checkpoint.

### 1. Generate and grade base-model answers

Launchers process one model at a time. `MODEL_NAME` selects the model;
`MODEL_TAG` selects its output directory. Use different tags for base and
trained models, especially when local checkpoints share a directory name.

```bash
export DOMAIN=math              # or science
export MODEL_NAME=Qwen/Qwen3-4B
export MODEL_TYPE=qwen
export MODEL_TAG=base-qwen3-4b
export SEED=3407

for split in train test; do
    SPLIT="$split" bash scripts/data/generate_answers.sh
    SPLIT="$split" bash "scripts/evaluation/grade_${DOMAIN}.sh"
done
```

Raw outputs: `outputs/answers/$DOMAIN/$MODEL_TAG/{train,test}.csv`, with a
`generation` column. Graded files end in `_graded.csv` and add `is_correct`.
`INPUT_CSV`, `OUTPUT_CSV`, `NUM_GENERATIONS`, and `MAX_TOKENS` can be overridden.
For older outputs, pass `EVAL_COL=benchmark_prediction_vllm` to the grader or
`GENERATIONS_COL=benchmark_prediction_vllm` to evaluation/CR.

### 2. Train CSA

Both SFT and GRPO accept `dataset_name` as a local CSV path or a HuggingFace
dataset repository containing a `train` split. Hub upload is optional; use
`scripts/hub/upload_dataset.sh` if needed. Update model, domain, dataset, and
output paths in the YAML before launching; supplied YAMLs contain placeholders.
The launchers use `scripts/training/deepspeed_zero3.yaml` unless `ACCEL_CONFIG`
is set to another accelerate configuration.

**Label-only SFT:** set `sft_mode: label` and point `dataset_name` at the base
model's `train_graded.csv` in `scripts/training/sft.yaml`, then run:

```bash
bash scripts/training/run_sft.sh scripts/training/sft.yaml
```

**Self-analysis SFT:** retain the base-model environment variables from step 1:

```bash
bash scripts/data/generate_analysis_self.sh
SFT_MODE=self bash scripts/data/build_sft_dataset.sh
```

Set `sft_mode: self` and `dataset_name` to
`outputs/sft/$DOMAIN/$MODEL_TAG/train_self.csv` (expand variables in the YAML),
then run the SFT launcher. The builder copies the matching `routing_analysis`
into `SFT_analysis` and checks that its conditioning label matches the model.

**Teacher-analysis SFT:** with `TOGETHER_API_KEY` set:

```bash
bash scripts/data/generate_analysis_teacher.sh
SFT_MODE=teacher bash scripts/data/build_sft_dataset.sh
```

The teacher generates analyses for both labels. The builder selects the one
matching the target model's label. Set `sft_mode: teacher` and point the SFT
YAML at `outputs/sft/$DOMAIN/$MODEL_TAG/train_teacher.csv` before launching.
`TEACHER_MODEL` can override the default teacher endpoint. Explicit `MODEL_CSV`,
`ANALYSIS_CSV`, and `OUTPUT_CSV` overrides are supported by the builder.

**RLVR / GRPO:** first construct a DFW subset from the **graded** training data:

```bash
INPUT_CSV="outputs/answers/$DOMAIN/$MODEL_TAG/train_graded.csv" \
OUTPUT_CSV="data/$DOMAIN/dfw_subset.csv" \
    bash scripts/training/run_dfw.sh
```

For warm-up, set `model_name` to the base model and `dataset_name` to the DFW
CSV in `scripts/training/grpo.yaml`, then run:

```bash
bash scripts/training/run_grpo.sh scripts/training/grpo.yaml
```

For full GRPO, change `model_name` to the warm-up `final_model` directory,
`dataset_name` to the full `train_graded.csv`, and `base_output_dir` to a new
directory. Run the launcher again. For OLMo2, omit DFW and train directly on
the full graded dataset; set `model_type: olmo` in the YAML and
`MODEL_TYPE=olmo` for inference. Do not reconstruct training labels using
the warm-up checkpoint.

### 3. Evaluate the trained checkpoint

Select the trained model explicitly and give it its own output tag:

```bash
export MODEL_NAME=/path/to/trained/final_model
export MODEL_TAG=trained-qwen3-4b

OUTPUT_CSV="outputs/csa/$DOMAIN/${MODEL_TAG}_test.csv" \
    bash scripts/inference/inference_local.sh
SPLIT=test bash scripts/data/generate_answers.sh
```

Use `BINARY_ONLY=1` for label-only SFT inference. Sampling controls
`TEMPERATURE`, `TOP_P`, and `TOP_K` apply to both single-shot and rollout CSA
inference. `SEED` controls local benchmark/CSA/DFW sampling and Science
option permutations; training uses the YAML `seed` independently.

**CSA quality:** by default evaluate against the checkpoint's freshly generated
answers, with the same query order as the predictions:

```bash
CSV_PATH="outputs/csa/$DOMAIN/${MODEL_TAG}_test.csv" \
GT_CSV_PATH="outputs/answers/$DOMAIN/$MODEL_TAG/test.csv" \
    bash scripts/evaluation/evaluate_csa.sh
```

The report includes M-F1, CDS, decision accuracy, self-solve rate, and invalid
prediction counts. M-F1/accuracy use valid decisions; invalid decisions are
reported separately. CDS is `NA` when a group is empty or its estimated
standard error is zero. `GRADE_MODE=column` is available for explicitly
provided `is_correct` labels, which must belong to the intended checkpoint.

**Capability retention:** CR is the ratio of mean **per-attempt** solving
accuracies, times 100. It does not aggregate attempts into capability labels.
For example, degrading from 5/5 to 1/5 correct attempts gives CR = 20%, even
though both Math any-correct labels remain positive.

```bash
PRE_CSV="outputs/answers/$DOMAIN/base-qwen3-4b/test.csv" \
POST_CSV="outputs/answers/$DOMAIN/$MODEL_TAG/test.csv" \
    bash scripts/evaluation/capability_ratio.sh
```

Both CSVs must contain the same queries in the same order and the same
positive number of attempts per query. A zero base accuracy makes CR undefined
and is reported as `NA`. Re-grade saved generations after parser or metric
changes before comparing outputs from different code revisions. Math grading
extracts balanced `\boxed{...}` answers and compares normalized strings; it
does not perform general symbolic equivalence checking.

## Regression tests

```bash
python -m pip install pandas numpy
python -m unittest discover -s tests -v
```

Tests cover grading, metrics, data assembly, and launcher arguments without
model downloads or API calls. Training configuration tests use stubs and do
not replace a CUDA training smoke test.

## Citation

The citation for the existing arXiv record is:

```bibtex
@misc{yang2026capabilityselfassessmentteachingllms,
  title={Capability Self-Assessment: Teaching LLMs to Know Their Limits},
  author={Haoyan Yang and Reza Shirkavand and Yukai Jin and Jiawei Zhou and Shangqian Gao and Heng Huang},
  year={2026},
  eprint={2606.00251},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2606.00251}
}
```
