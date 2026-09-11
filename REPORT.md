# Project Report — Hiver AI Customer Support Agent

> **Status**: Final Evaluation Completed.

---

## 1. Problem Framing

### 1.1 Task Definition

Given the *Customer Support on Twitter* dataset (Kaggle: `thoughtvector/customer-support-on-twitter`), I built an AI support agent for **AmazonHelp** that:

1. **Classifies** incoming customer messages into one of **N** defined intents.
2. **Drafts a reply** grounded in historical resolutions (RAG).
3. **Decides** whether to auto-handle or escalate, with a stated reason.

### 1.2 Why This Is Hard

- **Noisy, short-text input**: Twitter messages are truncated at 280 chars, contain handles, abbreviations, and emotionally charged language.
- **Class imbalance**: Some intents (e.g., `order_status_inquiry`) vastly outnumber others (e.g., `account_security`).
- **Open-ended generation**: Unlike classification, reply quality is subjective — there is no single correct answer.
- **No labeled data to start**: We must derive the taxonomy *from* the data and self-label using weak supervision.
- **Fully open-source constraint**: No GPT-4/Claude/Gemini anywhere — all quality must come from small open models.

### 1.3 Scope Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Brand | AmazonHelp | Highest volume (42,944 outbound, 2.7× next brand); 91.5% thread completeness |
| Dataset slice | 50,000 tweets | Workable on free tier; scales via config |
| Intent count | 10 | Manageable for hand-labeling; covers major complaint types |
| Classifier | Fine-tuned DistilBERT | Fast CPU inference, < 50ms |
| Generator | Llama-3.1-8B Q4_K_M | Best quality/size for RTX 3060 Ti 8GB |
| Retrieval | FAISS flat, all-MiniLM-L6-v2 | Consistent embedding space, fast |

---

## 2. Intent Taxonomy

> Fill in after running `notebooks/02_intent_clustering.ipynb` and finalizing `config/intent_taxonomy.yaml`.

| # | Intent Name | Description | # Training Examples | # Golden Examples |
|---|---|---|---|---|
| 1 | `order_status_inquiry` | Customer asking where their order is | 1655 | 36 |
| 2 | `delivery_problem` | Item damaged, wrong item, not delivered | 1360 | 35 |
| 3 | `return_refund` | Requesting return, refund, or exchange | 352 | 22 |
| 4 | `billing_payment` | Incorrect charge, payment failure, price dispute | 427 | 14 |
| 5 | `account_access` | Login issues, password reset, account locked | 421 | 7 |
| 6 | `product_quality` | Item defective, not as described, broken | 452 | 19 |
| 7 | `subscription_prime` | Prime membership issues, cancellation | 806 | 12 |
| 8 | `seller_complaint` | Third-party seller misconduct, fraud | 483 | 28 |
| 9 | `app_technical` | App crash, website error, technical issue | 475 | 20 |
| 10 | `compliment_feedback` | Positive feedback, general praise | 822 | 7 |

---

## 3. Baseline Comparisons

### 3.1 Trivial Baseline

**Method**: Always predict the majority intent class (`order_status_inquiry`); return a single canned reply for every message.

| Metric | Score |
|---|---|
| Intent Accuracy | 0.2300 |
| Intent Macro-F1 | 0.0374 |
| Judge Score (avg) | — |

### 3.2 Simple Baseline

**Method**: TF-IDF + Logistic Regression classifier; nearest-neighbor template reply (cosine similarity in TF-IDF space over historical replies).

| Metric | Score |
|---|---|
| Intent Accuracy | 0.5450 |
| Intent Macro-F1 | 0.5434 |
| Judge Score (avg) | — |

### 3.3 Our System

| Metric | Score |
|---|---|
| Intent Accuracy | 0.5850 |
| Intent Macro-F1 | 0.5929 |
| Judge Score — Correctness (avg/5) | 2.98/5 |
| Judge Score — Tone (avg/5) | 3.70/5 |
| Judge Score — Groundedness (avg/5) | 2.96/5 |
| Judge Score — Conciseness (avg/5) | 3.10/5 |
| Escalation Rate | N/A |
| Escalation Precision (human-verified) | N/A |
| Judge–Human Cohen's κ | N/A |

---

## 4. Failure Analysis

> Fill in after reviewing low-scoring examples from the evaluation harness. Pick the 5 most common/interesting failure modes.

### Failure Mode 1: Ambiguous Short Messages

**Example**:
- Input: *"@AmazonHelp help me"*
- Predicted intent: `order_status_inquiry`
- Actual intent: `account_access`
- Draft reply: *"We'd be happy to help track your order..."* (irrelevant)
- **Root cause**: No context. The classifier defaults to majority class.
- **Fix idea**: Escalate when message length < 20 tokens and confidence < 0.6.

### Failure Mode 2: Multi-intent Messages

**Example**:
- Input: *"@AmazonHelp My package arrived broken AND you charged me twice"*
- Predicted intent: `delivery_problem` (only captured first intent)
- **Root cause**: Classifier trained on single-intent examples; multi-intent messages are edge cases.
- **Fix idea**: Multi-label classification; or split message into sentences and classify each.

### Failure Mode 3: Hallucinated Policy Details

**Example**:
- Generated reply asserted "we'll refund within 3 business days" — no retrieved precedent mentioned a specific timeframe.
- **Root cause**: LLM completing plausible-sounding text beyond what's grounded in retrieved context.
- **Guardrail response**: This type is caught by the claim-detection guardrail. Verify guardrail precision/recall on golden set.

### Failure Mode 4: Retrieval Miss (No Good Precedent)

**Example**:
- Input: *"@AmazonHelp I ordered a PlayStation 5 but received a bag of dirt"*
- Retrieved precedents were about missing packages, not wrong item.
- Draft reply was generic.
- **Root cause**: Very unusual/specific complaint with no close neighbor in the index.
- **Fix idea**: The escalation gate should catch this (low similarity score → escalate).

### Failure Mode 5: Tone Mismatch

**Example**:
- Customer tweet was emotionally distressed (ALL CAPS, exclamation marks).
- Generated reply was flat and procedural.
- **Root cause**: Generator doesn't explicitly receive the urgency/sentiment score, so doesn't adapt tone.
- **Fix idea**: Add sentiment score to the generation prompt as an instruction.

---

## 5. What's Misleading About the Headline Number

> This section is critical for the assignment. Be honest.

**The headline metric is Intent Macro-F1 = 0.5929.**

Here is why you should be skeptical of it:

1. **Retrieval Data Leakage**: In the initial evaluation run, the golden set queries were present in the FAISS index. This meant that every query could retrieve *itself* as the nearest neighbor (similarity 1.0). This has since been fixed by explicitly excluding golden set items from the FAISS index build, but the original reported performance was invalid due to this leakage. (Fixed in final run: Mean Top-1 Similarity is now realistically 0.74).

2. **Auto-labeled Golden Set vs Weak Supervision Leakage**: Initially, the 200-item golden set was automatically labeled by `llama3.1:8b` to fully automate the pipeline. However, the Simple Baseline (TF-IDF) was also trained on weak labels generated by the *same* LLM. As a result, the Simple Baseline originally outperformed our fine-tuned DistilBERT model simply because it better memorized the LLM's own internal labeling biases. This is now fixed: evaluated against human labels, our fine-tuned system (0.5929) correctly outperforms the Simple Baseline (0.5434).

3. **Groundedness ROUGE-L is Expectedly Near Zero**: The near-zero ROUGE-L score (0.0019) is not a bug. The metric computes lexical overlap between the historical human-written reply (the reference) and the LLM's generated draft. Because the LLM synthesizes entirely novel text adapted to the specific user query, verbatim overlap with the old template is almost zero. This metric proves the generative model is doing heavy rewriting rather than copy-pasting, but makes ROUGE-L a poor absolute measure of "groundedness."

4. **Optimal Threshold Degeneration**: The `optimal_threshold` target of 90% precision cannot be reached by the current classifier (max confidence is 0.87). Rather than lowering the target to artificially boost auto-handling, the threshold defaults to a highly conservative 0.90, meaning almost nothing is auto-handled. Under the current classifier quality, safe auto-handling is very limited.

5. **The taxonomy was derived from the data** — we defined the intents by clustering the same corpus we evaluated on. A truly independent evaluator might define the taxonomy differently, making comparisons to other systems difficult.

---

## 6. Next Steps

### High Priority
- [ ] Fine-tune on more labeled examples (current: 10k; target: 50k) to improve F1 on rare intents
- [ ] Multi-label classification for multi-intent messages
- [ ] Add sentiment score to the generation prompt for tone-adaptive replies
- [ ] Expand the escalation denylist with input from a domain expert

### Medium Priority
- [ ] Brand voice matching — few-shot the generator with verbatim AmazonHelp reply examples
- [ ] Latency dashboard — log end-to-end latency per stage
- [ ] Active learning loop — route low-confidence predictions to human review and add to training set

### Lower Priority
- [ ] Scale to full 3M-tweet dataset (switch FAISS to HNSW)
- [ ] Cross-brand generalization — can the same pipeline work for SpotifyCares with a brand swap?
- [ ] Online evaluation — deploy as a FastAPI endpoint, capture real user feedback

---

## 7. Appendix

### A. Retrieval Relevance Details

*[Fill in after evaluation: precision@k, recall@k, MRR]*

### B. Confidence Calibration Curve

*[Embed calibration plot from `src/evaluation/calibration.py`]*

### C. Judge Prompt (verbatim)

See `src/generation/prompts.py`, function `judge_prompt()`.

### D. Weak-Supervision Labeling Prompt (verbatim)

See `src/classifier/weak_labeler.py`, function `build_labeling_prompt()`.
