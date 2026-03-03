# PromptEcho

## 📌 Overview
This repository contains the official implementation of **PromptEcho**, a generative eavesdropping attack designed to reconstruct natural-language responses from encrypted LLM API traffic through merged token-length side channel. 

## Environment Setup
We recommend using Python 3.9+ and a clean virtual environment:
```bash
# Install required dependencies
pip install -r requirements.txt
```

## Training
We use `torchrun` to fine-tune the model's linguistic priors on single and merged-token character length patterns. This step requires a multi-GPU setup (8 GPUs recommended).


**Stage 1: Deterministic Alignment Fine-Tuning**

Initializes the model to understand the mapping between individual token characters and semantic embeddings.
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 scripts/single_alignment.py
```

**Stage 2: Reasoning-based Decomposition with CoT**

Based on the Stage 1 model, this phase introduces merged-token sequences with cot supervised signal to train the semantics-aware decomposition and reasoning abilities.
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 scripts/merged_decomposition.py
```

- **Input**: Training corpus of dialogue pairs with single and merged-token length metadata.
- **Output**: Saved LoRA checkpoints and updated tokenizer configurations in `./checkpoints/`.

## Inference
Once the model is trained, use the inference script to reconstruct responses from the encrypted packet-length sequences. This script integrates the base model embeddings with the learned LoRA parameters.
```bash
# Generate reconstructed text from the test side-channel traces
python scripts/generate_new_refactored.py
```
- **Input**: Loads the test data, trained tokenizer, base model embeddings, and LoRA weights to perform reasoning-guided sequence recovery.
- **Output**: A file named `generated.csv` containing the reconstructed natural language responses.

## Evaluation
Finally, evaluate the quality of the reconstruction using the standard metrics described in the paper (e.g., Cosine Similarity, ROUGE, BLEU, etc.). We use `accelerate` for optimized multi-GPU evaluation.
```bash
# Launch multi-GPU evaluation for the 5 key security and linguistic metrics
accelerate launch --multi_gpu --mixed_precision bf16 scripts/eval_5metrics.py \
    --input generated.csv \
    --output metrics_all.csv
```
- **Input**: The `generated.csv` file from Inference step.
- **Output**: `metrics_all.csv`, which provides a detailed breakdown of the reconstruction performance.

## 📂 Repository Structure
```text
├── data/
│   ├── ChatGPT/            # Exemplar .pcap traces from ChatGPT sessions
│   ├── DeepSeek/           # Exemplar .pcap traces from DeepSeek sessions
│   ├── train_align.json    # [Stage 1] Subset of Single-Token alignment data
│   ├── test_align.csv     # [Stage 1] Subset of Single-Token test data
│   ├── train_decompose.json # [Stage 2] Subset of Merged-Token decomposition data
│   └── test_decompose.json  # [Stage 2] Subset of Merged-Token test data
├── scripts/
│   ├── single_alignment.py     # Stage 1: Deterministic Alignment Fine-Tuning
│   ├── merged_decomposition.py # Stage 2: Reasoning-based Decomposition with CoT
│   ├── generate_new_refactored.py # Inference engine for sequence reconstruction
│   └── eval_5metrics.py        # Multi-metric evaluation (Cosine, R1, ED, etc.)
├── Dockerfile              # Standardized reproduction environment
└── requirements.txt        # Python dependencies
```