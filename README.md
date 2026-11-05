# AI4Industrial Automation @ FIRE 2026: NL2PLC Code Generation

**Team:** TokenX

Our system for the [AI4Industrial Automation shared task at FIRE 2026](https://fire2026-ai4industry.github.io/fire2026-nl2plc/index.html). Given a natural language description of industrial control logic, the pipeline generates:

- **Task A:** IEC 61131-3 **Structured Text (ST)**
- **Task B:** **Rust** code implementing the same control logic

Every test row requires both a Task A and a Task B prediction.

Reference repository: [AI4Industrial-Automation-NL2PLC-Code-Generation-and-Understanding](https://github.com/rachanabn20/AI4Industrial-Automation-NL2PLC-Code-Generation-and-Understanding)

---

## Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Models](#models)
- [Pipeline](#pipeline)
- [The Three Submitted Runs](#the-three-submitted-runs)
- [Installation](#installation)
- [Usage](#usage)
- [Output Format](#output-format)
- [Evaluation](#evaluation)
- [Hardware and Reproducibility](#hardware-and-reproducibility)
- [Shared Task Restrictions](#shared-task-restrictions)
- [Limitations](#limitations)

---

## Overview

The pipeline is a single notebook that runs end to end:

1. Load and validate the training and test CSVs.
2. Clean the data and build separate Task A (NL → ST) and Task B (NL → Rust) datasets.
3. Split into train/validation by **unique description**, so the same task text never lands in both splits.
4. Evaluate two zero-shot baselines and three prompt variants.
5. Fine-tune a LoRA adapter per task on a 4-bit quantized model.
6. Generate test predictions with three different strategies (individual model, alternative model, ensemble).
7. Diagnose structurally broken outputs, regenerate them, validate, and package each run as a submission zip.

---

## Repository Structure

```
.
├── AI4Industrial_Automation_Pipeline.ipynb   # Full pipeline
├── README.md
├── requirements.txt
├── checkpoints/                              # LoRA adapters (created at runtime)
│   ├── lora_qwen_coder_1_5b_taskA/
│   └── lora_qwen_coder_1_5b_taskB/
└── outputs/                                  # Created at runtime
    ├── predictions_cache/                    # JSON caches and resumable partials
    ├── submission_run1.csv / .xlsx
    ├── submission_run2.csv / .xlsx
    ├── submission_run3.csv / .xlsx
    └── submission_packages/
        ├── RUN-TokenX-1.zip
        ├── RUN-TokenX-2.zip
        └── RUN-TokenX-3.zip
```

Each submission zip contains:

```
RUN-TokenX-N/
├── README.md
├── environment_info.json
├── code/
│   ├── pipeline.ipynb
│   └── requirements.txt
└── outputs/
    ├── submission_runN.csv
    └── submission_runN.xlsx
```

---

## Data

The shared task data is **not redistributed** here. Obtain it from the task organizers and place it locally.

| File | Columns | Notes |
|------|---------|-------|
| Training CSV | `TASK`, `ST-CODE`, `RUST-CODE` | Natural language description with reference ST and Rust |
| Test CSV | `TASK` | Only `TASK` carries data; empty artifact columns are dropped |

Neither file has an ID column, so the **row index** is used as the identifier (`ID`).

Update the paths at the top of the notebook:

```python
TRAIN_PATH = r"path/to/Training-Dataset-AI4Industrial-Automation-FIRE-2026.csv"
TEST_PATH  = r"path/to/Test-Dataset-AI4Industrial-Automation.csv"
OUTPUT_DIR = r"path/to/outputs"
CHECKPOINT_DIR = r"path/to/checkpoints"
```

### Preprocessing

- CSVs are read with an encoding fallback (`utf-8`, `utf-8-sig`, `latin1`, `cp1252`).
- Empty, `nan`, `none`, and `null` values are treated as missing.
- Rows with empty inputs are dropped; (input, target) pairs are de-duplicated per task.
- Heuristic checks flag malformed ST (needs `PROGRAM` / `END_PROGRAM`) and Rust (needs `fn`, `impl`, or `struct`).
- Maximum input and target lengths are derived from the 95th percentile token length plus a margin, capped at 512 (input) and 896 (target).

### Train/validation split

An 85/15 split is performed over **normalized unique descriptions** (lowercased, whitespace-collapsed) with seed 42. Both tasks reuse the same description split to prevent leakage.

---

## Models

All models are open-source and at or below 8B parameters (enforced by an assertion in the code).

| Key | Model | Params |
|-----|-------|--------|
| `qwen_coder_1_5b` | Qwen/Qwen2.5-Coder-1.5B-Instruct | 1.5B |
| `qwen_coder_3b` | Qwen/Qwen2.5-Coder-3B-Instruct | 3B |
| `qwen_coder_7b` | Qwen/Qwen2.5-Coder-7B-Instruct | 7B |
| `deepseek_coder_1_3b` | deepseek-ai/deepseek-coder-1.3b-instruct | 1.3B |
| `deepseek_coder_6_7b` | deepseek-ai/deepseek-coder-6.7b-instruct | 6.7B |

The runs submitted use **Qwen2.5-Coder-1.5B-Instruct** and **Qwen2.5-Coder-7B-Instruct**, both loaded in 4-bit NF4 with double quantization and fp16 compute.

---

## Pipeline

### Prompting

Three prompt templates per task are compared:

| Version | Description |
|---------|-------------|
| `v1_direct` | Short instruction, output code only |
| `v2_structured` | Explicit requirements (valid block structure, no markdown, no hallucinated addresses) |
| `v3_fewshot` | One worked example (start/stop conveyor) followed by the new description |

The best prompt per task is selected on a validation subset using the proxy score described under [Evaluation](#evaluation).

### LoRA fine-tuning

| Setting | Value |
|---------|-------|
| Base model | Qwen2.5-Coder-1.5B-Instruct (4-bit) |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Learning rate | 2e-4 |
| Epochs | 2 |
| Batch size / grad. accumulation | 1 / 8 |
| Optimizer | `paged_adamw_8bit` |
| Precision | fp16, gradient checkpointing |

Loss is masked on the prompt tokens, so only the target code contributes to training. A separate adapter is trained for Task A (prompt `v3_fewshot`) and Task B (prompt `v2_structured`). Fine-tuning is skipped if fewer than 30 training examples are available.

### Generation

- Greedy or sampled decoding depending on the run
- `repetition_penalty=1.15`, `no_repeat_ngram_size=4`
- Task-specific stop strings (`END_PROGRAM` for ST; `fn main()` and `std::thread::sleep` for Rust)
- Automatic fallback to halved and quartered `max_new_tokens` on OOM or generation errors
- Post-processing strips markdown fences and conversational preamble lines

### Robustness features

- **Resumable inference:** each prediction is written to a `_partial.json` file, so interrupted runs continue where they stopped.
- **Prediction caching:** finished runs are cached as JSON and reloaded instead of regenerated.
- **Failure eviction:** rows whose ST output fails the structural check are evicted and regenerated.
- **GPU memory management:** explicit model unloading, cache clearing, and release assertions between models.

---

## The Three Submitted Runs

| Run | Strategy | Model(s) | Decoding |
|-----|----------|----------|----------|
| **1** | Best individual fine-tuned model, best prompt per task | Qwen2.5-Coder-1.5B + LoRA | Deterministic |
| **2** | Alternative model and prompt strategy | Qwen2.5-Coder-7B (no fine-tuning) | Sampled (T = 0.3), `v3_fewshot` |
| **3** | Ensemble with candidate ranking | Qwen2.5-Coder-1.5B (+ LoRA) and 7B | Deterministic candidates, best selected by score |

In Run 3, each model produces a candidate per row and the candidate with the highest score is kept. Candidate scoring combines a structural score with a syntax validity score. A semantic similarity term is available when a reference text is supplied, but it is not used at test time because no references exist.

---

## Installation

```bash
conda create -n torch_env python=3.10
conda activate torch_env
pip install -r requirements.txt
```

`requirements.txt`:

```
transformers>=4.44.0
accelerate>=0.33.0
peft>=0.12.0
bitsandbytes>=0.43.0
datasets>=2.20.0
sentencepiece
sentence-transformers
scikit-learn
pandas
numpy
torch
tqdm
safetensors
openpyxl
```

Install a PyTorch build that matches your CUDA version. A CUDA GPU is required for 4-bit quantization with `bitsandbytes`.

---

## Usage

1. Place the training and test CSVs and update the paths in the first configuration cell.
2. Open `AI4Industrial_Automation_Pipeline.ipynb`.
3. Run all cells from top to bottom.

The notebook will:

- print environment and GPU information,
- evaluate baselines and prompt variants,
- train the LoRA adapters,
- generate predictions for Runs 1, 2, and 3,
- validate the submissions,
- write CSV and XLSX files plus zipped packages under `outputs/submission_packages/`.

To force regeneration of a run, delete its files in `outputs/predictions_cache/` (both `runN_predictions.json` and `runN_predictions_partial.json`).

---

## Output Format

Each submission file has three columns:

| Column | Description |
|--------|-------------|
| `ID` | Row index of the test file |
| `ST-CODE` | Generated IEC 61131-3 Structured Text (Task A) |
| `RUST-CODE` | Generated Rust code (Task B) |

CSV and XLSX versions contain identical data. Excel caps cells at 32,767 characters, so the notebook warns if any cell exceeds that. The CSV is never truncated.

### Automated validation

Before packaging, each submission is checked for:

- correct row count, no duplicated or missing IDs
- no missing or empty ST / Rust cells
- no leftover markdown fences
- ST and Rust outputs that pass a minimal structural check

---

## Evaluation

The notebook reports **local proxy metrics** only. They are **not** the official FIRE 2026 leaderboard metrics.

| Metric | Description |
|--------|-------------|
| Structural similarity | Fraction of structural checks passed (block pairing such as `IF`/`END_IF`, balanced brackets, expected keywords) |
| Syntax validity | Heuristic validity check (ST: `PROGRAM`/`END_PROGRAM` and balanced parentheses; Rust: balanced braces and a `fn`) |
| Semantic similarity | Cosine similarity of `all-MiniLM-L6-v2` embeddings between generated and reference code |
| Rank score | `0.4 * structural + 0.3 * syntax + 0.3 * semantic` |

These checks do not compile or execute code, so a high proxy score does not guarantee that the ST or Rust code compiles or behaves correctly.

---

## Hardware and Reproducibility

- Global seed: **42** (Python, NumPy, PyTorch, CUDA; deterministic cuDNN)
- Experiments were designed for a single GPU with roughly 16 GB of VRAM (`MAX_SAFE_VRAM_GB = 15.9`)
- Each packaged run includes `environment_info.json` with the Python, PyTorch, Transformers, and PEFT versions, CUDA version, GPU, and VRAM
- Run 2 uses sampled decoding, so its output can vary slightly across runs even with a fixed seed

---

## Shared Task Restrictions

- Only open-source models with at most 8B parameters are used.
- No proprietary or closed APIs are used for generation or training.
- Test labels and reference outputs are never used during test inference.

---

## Limitations

- Validation uses proxy metrics based on string and embedding heuristics, not compilation or simulation.
- The Rust syntax check is a lightweight brace and keyword test, not `rustc`.
- Generated ST may include illustrative `AT %IX` / `%QX` addresses that do not correspond to a real hardware configuration.
- The failure-diagnosis step checks only ST structure; Rust outputs are validated more lightly.
- Training data is small, so the LoRA adapters may overfit to the surface style of the training descriptions.

---

## Acknowledgements

Thanks to the organizers of the AI4Industrial Automation shared task at FIRE 2026, and to the authors of Qwen2.5-Coder, DeepSeek-Coder, Hugging Face Transformers, PEFT, and bitsandbytes.
