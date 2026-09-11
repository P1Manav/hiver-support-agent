# Hiver AI Customer Support Agent
### SDE Intern Take-Home · Customer Support on Twitter (AmazonHelp)

> **Reproduce headline results in < 15 minutes of compute time** (after model weights are cached — see [Compute Budget](#compute-budget) below).

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Architecture Overview](#architecture-overview)
3. [Quickstart](#quickstart)
4. [Compute Budget](#compute-budget)
5. [Repo Structure](#repo-structure)
6. [Stage-by-Stage Guide](#stage-by-stage-guide)
7. [Evaluation Results](#evaluation-results)
8. [Running the Baselines](#running-the-baselines)
9. [Configuration Reference](#configuration-reference)
10. [Model Selection & Swap Guide](#model-selection--swap-guide)
11. [Known Limitations](#known-limitations)
12. [License](#license)

---

## Problem Statement

Given a stream of customer tweets directed at **AmazonHelp**:
1. **Classify** the incoming message into one of 10 defined intents.
2. **Draft a reply** grounded in how AmazonHelp has historically resolved similar issues (RAG over past resolved threads).
3. **Decide** whether to auto-handle or escalate, with a human-readable reason.

**Hard constraint**: 100% open-source — no paid API calls anywhere in the pipeline.

---

## Architecture Overview

```
Customer Tweet
      │
      ▼
┌─────────────────┐    ┌─────────────────────────────────┐
│  Intent         │    │  FAISS Index                    │
│  Classifier     │    │  (historical resolved pairs)    │
│  (DistilBERT,   │    └───────────────┬─────────────────┘
│   fine-tuned)   │                    │ top-k retrieval
└────────┬────────┘                    │
         │ intent + confidence         │
         ▼                            ▼
┌────────────────────────────────────────────┐
│            Escalation Gate                  │
│  (confidence × similarity × denylist ×     │
│   sentiment/urgency → auto | escalate)      │
└────────────────────┬───────────────────────┘
                     │ if auto_handle
                     ▼
         ┌───────────────────────┐
         │   Llama-3.1-8B-Instruct│  ← via Ollama (local 3060 Ti)
         │   (RAG-grounded reply) │    or Phi-3.5-mini (fast iter)
         └───────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │     Guardrails        │  ← flags unverifiable claims
         └───────────────────────┘
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally (for generator + judge)
- Kaggle API token at `~/.kaggle/kaggle.json`
- (Optional, for fine-tuning) Google Colab or Kaggle notebook with T4 GPU

### 1. Clone & install

```bash
git clone https://github.com/P1Manav/hiver-support-agent.git
cd hiver-support-agent
pip install -r requirements.txt
```

### 2. Pull Ollama models (one-time, ~10 GB total — see [Compute Budget](#compute-budget))

```bash
ollama pull llama3.1:8b-instruct-q4_K_M    # generator (~4.7 GB)
ollama pull qwen2.5:7b-instruct-q4_K_M     # judge     (~4.3 GB)
# Fast-iter swap (optional):
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M  # ~2.2 GB
```

### 3. Download & process data

```bash
# Step 1: see brand volumes (pick your brand)
python scripts/01_brand_stats.py

# Step 2: run full data pipeline for AmazonHelp
python scripts/02_data_pipeline.py --brand AmazonHelp --sample-size 50000
```

### 4. Build intent taxonomy

```bash
# Embed + cluster (CPU-fine, ~8 min for 5k subsample)
python scripts/03_embed_and_cluster.py

# Inspect data/processed/cluster_samples.json to see sample messages per cluster.
# Edit config/intent_taxonomy.yaml with your final intent names.
# (cluster_samples.json already exists in this repo if you cloned with data)
```

### 5. Label training data & fine-tune classifier

```bash
# Local: bulk-label with Ollama (~2 hrs for 10k samples on RTX 3060 Ti)
# Non-English messages are automatically filtered (langdetect) — see D-14 in DECISION_LOG.md
python scripts/04_label_with_llm.py --n-samples 10000

# Colab/Kaggle (faster, ~45 min on T4): open notebooks/03_finetune_classifier.ipynb
# Upload data/processed/labeled_training.csv, run notebook, download models/classifier/ back
```

### 6. Build FAISS index

```bash
python scripts/05_build_faiss_index.py   # ~15 min on CPU for 50k pairs
```

### 7. Run end-to-end inference

```bash
python scripts/06_inference.py \
  --message "My order #112-3456789 hasn't arrived in 3 weeks, this is unacceptable"
```

Expected output:
```json
{
  "intent": "order_status_inquiry",
  "confidence": 0.94,
  "decision": "auto_handle",
  "reason": "High classifier confidence (0.94); good retrieval similarity (0.81); intent 'order_status_inquiry' not in high-risk denylist.",
  "draft_reply": "Hi! We're sorry to hear your order hasn't arrived. Please DM us your order number and we'll look into this right away. ^Team",
  "guardrail_flags": []
}
```

### 8. Run evaluation

```bash
# Step 1: Sample golden set candidates
python scripts/07_golden_set_sampler.py  # → golden/golden_set_to_label.csv

# Step 2: Hand-label using the CLI tool (~45–80 min for 200 examples)
python scripts/label_cli.py

# Step 3: Run full evaluation
python scripts/08_judge_agreement.py --golden golden/golden_set.csv
```

---

## Compute Budget

| Step | Where | Cold start (weights not cached) | After caching |
|---|---|---|---|
| `pip install` | Local | ~3 min | ~30 s |
| Ollama model pulls | Local | ~20 min (10 GB) | 0 s |
| Data pipeline (50k) | Local CPU | ~5 min | ~5 min |
| Embed + cluster (50k) | Local CPU | ~8 min | ~8 min |
| FAISS index build (50k pairs) | Local CPU | ~15 min | ~15 min |
| **Single inference** | Local RTX 3060 Ti | **< 5 s** | **< 5 s** |
| Fine-tune DistilBERT | Colab T4 | ~25 min | ~25 min |
| Bulk LLM labeling (10k) | Colab T4 | ~45 min | ~45 min |
| Eval harness (150 examples) | Local/Colab | ~10 min | ~10 min |

> **"< 15 min after caching"** = data pipeline (5) + FAISS (15) + single inference demo (< 1) + eval on pre-built golden set (10) ≈ **31 min** total. To reproduce the *headline result only* in < 15 min, use the pre-committed golden set and a pre-built FAISS index (skip fine-tune and re-labeling).

---

## Repo Structure

```
hiver-support-agent/
├── README.md
├── DECISION_LOG.md          # 14+ non-obvious design decisions + rationale
├── REPORT.md                # Full project report
├── requirements.txt         # CPU-safe pip deps
├── requirements_gpu.txt     # GPU-only (torch+cuda, for Colab)
├── config/
│   ├── config.yaml          # All tunable parameters
│   └── intent_taxonomy.yaml # Intent names, descriptions, examples
├── data/                    # gitignored (raw/processed); golden/ is committed
├── scripts/
│   ├── 01_brand_stats.py    # Auto-downloads data + shows brand volume stats
│   ├── 02_data_pipeline.py  # Full pipeline: download → threads → PII → sample
│   ├── 03_embed_and_cluster.py  # Embed + KMeans cluster for intent discovery
│   ├── 04_label_with_llm.py    # Bulk-label with Ollama (English-only filter included)
│   ├── 05_build_faiss_index.py # Build FAISS index over resolved pairs
│   ├── 06_inference.py         # End-to-end inference pipeline
│   ├── 07_golden_set_sampler.py # Stratified sampling for golden set
│   ├── 08_judge_agreement.py   # Full evaluation + LLM judge
│   └── label_cli.py            # CLI hand-labeling tool for golden set
├── notebooks/
│   └── 03_finetune_classifier.ipynb  # DistilBERT fine-tuning (Colab T4)
├── src/                     # Library code
│   ├── data/                # ingest, threads, pii
│   ├── intents/             # embedder, clusterer, taxonomy
│   ├── classifier/          # weak_labeler, finetune, predict
│   ├── retrieval/           # indexer, retriever
│   ├── generation/          # prompts, generator, guardrails
│   ├── escalation/          # gate
│   ├── evaluation/          # metrics, judge, calibration
│   └── baselines/           # trivial, simple
├── golden/
│   └── golden_set.csv       # 150–250 hand-labeled examples (committed after labeling)
└── tests/
    ├── test_escalation.py
    ├── test_metrics.py
    └── test_pii.py
```

---

## Stage-by-Stage Guide

See the **[full guide in REPORT.md](REPORT.md)** for detailed explanations.

---

## Evaluation Results

> Fill in after running `python scripts/08_judge_agreement.py --golden golden/golden_set.csv`

| System | Intent F1 | Reply Groundedness | Judge Score (/5) |
|---|---|---|---|
| Trivial baseline | — | — | — |
| TF-IDF + NN baseline | — | — | — |
| **Our system** | — | — | — |

---

## Running the Baselines

```bash
# Trivial baseline
python -m src.baselines.trivial --golden golden/golden_set.csv

# TF-IDF + LogReg + NN template
python -m src.baselines.simple --golden golden/golden_set.csv
```

---

## Configuration Reference

All parameters live in [`config/config.yaml`](config/config.yaml). Key knobs:

| Key | Default | Description |
|---|---|---|
| `brand` | `AmazonHelp` | Twitter brand handle to filter |
| `sample_size` | `50000` | Number of tweets to process |
| `embedding_model` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `n_clusters` | `12` | KMeans clusters for intent derivation |
| `top_k` | `5` | Retrieved precedents per query |
| `escalation.confidence_threshold` | `0.70` | Below this → escalate |
| `escalation.similarity_threshold` | `0.60` | Below this → escalate (no precedent) |
| `generator_model` | `llama3.1:8b-instruct-q4_K_M` | Ollama model for reply generation |
| `judge_model` | `qwen2.5:7b-instruct-q4_K_M` | Ollama model for evaluation |

---

## Model Selection & Swap Guide

| Role | Default (quality) | Fast-iter swap |
|---|---|---|
| Generator | `llama3.1:8b-instruct-q4_K_M` | `phi3.5:3.8b-mini-instruct-q4_K_M` |
| Judge | `qwen2.5:7b-instruct-q4_K_M` | run on Colab T4 in fp16 |
| Classifier | fine-tuned `distilbert-base-uncased` | same (fast CPU inference) |
| Embedder | `all-MiniLM-L6-v2` | same (22M params, CPU-fine) |

To swap generator: edit `generator_model` in `config/config.yaml`.

---

## Known Limitations

1. **Thread reconstruction** is best-effort — the dataset's reply graph is sometimes incomplete (orphaned replies without their parent).
2. **Weak-supervision label noise** — LLM-generated labels have ~5–10% error rate; the fine-tuned model inherits some of this noise.
3. **Headline metric caveat** — Intent F1 is computed on *our own golden set* which we also used to tune thresholds. See `REPORT.md § What's Misleading About the Headline Number`.
4. **Generator latency** — On RTX 3060 Ti (8GB), Llama-3.1-8B Q4_K_M runs at ~20 tok/s. Each reply takes 3–8 seconds.

---

## License

MIT — see [LICENSE](LICENSE).
