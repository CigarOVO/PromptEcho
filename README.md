# PromptEcho

## 📌 Overview
This repository contains the official implementation of **PromptEcho**, a passive eavesdropping attack designed to reconstruct natural-language responses from encrypted LLM API traffic through a merged token-length side channel. 

## Environment Setup
We recommend using Python 3.9+ and a clean virtual environment:
```bash
# Install required dependencies
pip install -r requirements.txt
```

## Training
We use `torchrun` to fine-tune the model's linguistic priors on single and merged-token character length patterns. This step requires a multi-GPU setup (8 GPUs recommended).


**Stage 1: Deterministic Alignment Fine-Tuning**

Initializes the model to understand the mapping between individual token-length observations and likely semantic token sequences.
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 scripts/single_alignment.py
```

**Stage 2: Reasoning-based Decomposition with CoT**

Based on the Stage 1 model, this phase introduces merged-token sequences with chain-of-thought supervised signal to train the semantics-aware decomposition and reasoning abilities.
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 scripts/merged_decomposition.py
```

- **Input**: Training corpus of dialogue pairs with single and merged-token length metadata.
- **Output**: Saved LoRA checkpoints and updated tokenizer configurations in `./checkpoints/`.

## Inference
Once the model is trained, use the inference script to reconstruct responses from the encrypted packet-length sequences. This script loads the base model together with the learned LoRA parameters.
```bash
# Generate reconstructed text from the test side-channel traces
python scripts/generate_new_refactored.py
```
- **Input**: Loads the test data, trained tokenizer, base model embeddings, and LoRA weights to perform reasoning-guided sequence recovery.
- **Output**: A file named `generated.csv` containing the reconstructed natural language responses.

## Evaluation
Finally, evaluate the quality of the reconstruction using the standard metrics described in the paper (e.g., Cosine Similarity, ROUGE, Edit Distance, etc.). We use `accelerate` for optimized multi-GPU evaluation.
```bash
# Launch multi-GPU evaluation for the evaluation metrics reported in the paper
accelerate launch --multi_gpu --mixed_precision bf16 scripts/eval_5metrics.py \
    --input generated.csv \
    --output metrics_all.csv
```
- **Input**: The `generated.csv` file from Inference step.
- **Output**: `metrics_all.csv`, which provides a detailed breakdown of the reconstruction performance.

## 📂 Repository Structure
```text
├── data/
│   ├── ChatGPT/             # Exemplar .pcap traces from ChatGPT sessions
│   ├── DeepSeek/            # Exemplar .pcap traces from DeepSeek sessions
│   ├── train_align.jsonl    # [Stage 1] Single-Token alignment data
│   ├── test_align.csv       # [Stage 1] Single-Token test data
│   ├── train_decompose.jsonl # [Stage 2] Merged-Token decomposition data
│   └── test_decompose.jsonl  # [Stage 2] Merged-Token test data
├── scripts/
│   ├── single_alignment.py     # Stage 1: Deterministic Alignment Fine-Tuning
│   ├── merged_decomposition.py # Stage 2: Reasoning-based Decomposition with CoT
│   ├── generate_new_refactored.py # Inference engine for sequence reconstruction
│   └── eval_5metrics.py        # Multi-metric evaluation (Cosine, R1, ED, etc.)
└── requirements.txt        # Python dependencies
```

## Data Anonymization Notice
All released network traces are anonymized exemplar traces collected for research purposes. Sensitive identifiers and authentication-related metadata have been removed prior to release.


