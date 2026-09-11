# Project Report — Hiver AI Customer Support Agent

> **Status**: Template — fill in results after running evaluation harness.

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
| 1 | `order_status_inquiry` | Customer asking where their order is | — | — |
| 2 | `delivery_problem` | Item damaged, wrong item, not delivered | — | — |
| 3 | `return_refund` | Requesting return, refund, or exchange | — | — |
| 4 | `billing_payment` | Incorrect charge, payment failure, price dispute | — | — |
| 5 | `account_access` | Login issues, password reset, account locked | — | — |
| 6 | `product_quality` | Item defective, not as described, broken | — | — |
| 7 | `subscription_prime` | Prime membership issues, cancellation | — | — |
| 8 | `seller_complaint` | Third-party seller misconduct, fraud | — | — |
| 9 | `app_technical` | App crash, website error, technical issue | — | — |
| 10 | `compliment_feedback` | Positive feedback, general praise | — | — |

---

## 3. Baseline Comparisons

### 3.1 Trivial Baseline

**Method**: Always predict the majority intent class (`order_status_inquiry`); return a single canned reply for every message.

| Metric | Score |
|---|---|
| Intent Accuracy | 0.1800 |
| Intent Macro-F1 | 0.0305 |
| Judge Score (avg) | — |

### 3.2 Simple Baseline

**Method**: TF-IDF + Logistic Regression classifier; nearest-neighbor template reply (cosine similarity in TF-IDF space over historical replies).

| Metric | Score |
|---|---|
| Intent Accuracy | 0.4900 |
| Intent Macro-F1 | 0.4934 |
| Judge Score (avg) | — |

### 3.3 Our System

| Metric | Score |
|---|---|
| Intent Accuracy | 0.4200 |
| Intent Macro-F1 | 0.4250 |
| Judge Score — Correctness (avg/5) | 3.10/5 |
| Judge Score — Tone (avg/5) | 3.54/5 |
| Judge Score — Groundedness (avg/5) | 3.22/5 |
| Judge Score — Conciseness (avg/5) | 3.17/5 |
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

**The headline metric is Intent Macro-F1 = 0.4250.**

Here is why you should be skeptical of it:

1. **We built the golden set ourselves** — using stratified sampling from the same distribution we trained on. There is potential for label leakage if the same edge cases were used to tune the escalation thresholds.

2. **The taxonomy was derived from the data** — we defined the intents by clustering the same corpus we evaluated on. A truly independent evaluator might define the taxonomy differently, making comparisons to other systems meaningless.

3. **10 intents may be the wrong granularity** — "delivery_problem" bundles together "never arrived", "arrived damaged", "wrong item", which have very different resolutions. A finer taxonomy would show lower F1; a coarser one would show higher F1.

4. **Judge–human agreement is on a subset** — Cohen's κ computed on ~50 examples. Small sample → wide confidence interval.

5. **Generator evaluation ignores brand voice** — the judge scores correctness and tone but has no ground-truth reference for what AmazonHelp's actual reply would be. A reply that's generically correct may differ substantially from brand voice.

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
