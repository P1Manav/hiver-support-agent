# Decision Log — Hiver AI Support Agent

> Fill this in **as you go**, not at the end. Each decision should take < 5 minutes to record.
> Format: **Decision → Rationale → Alternatives considered → Trade-offs**

---

## D-01: Brand = AmazonHelp (default)

**Decision**: Use AmazonHelp as the default brand throughout the pipeline.

**Rationale**: AmazonHelp has the highest message volume in the dataset (~500k+ tweets), the best thread completeness (many customer→brand→customer chains), and covers a wide range of intents (shipping, returns, billing, account, product quality). This maximizes RAG retrieval quality and gives the classifier enough signal per class.

**Alternatives considered**: AppleSupport (good brand voice, but narrower intent range), SpotifyCares (mid-size, but many intents map to "app bug" which is hard to resolve via RAG).

**Trade-off**: Higher volume = longer data pipeline runtime. Mitigated by the `sample_size` config parameter.

---

## D-02: Embedding model = all-MiniLM-L6-v2

**Decision**: Use `all-MiniLM-L6-v2` (22M params) for both clustering and FAISS retrieval.

**Rationale**: Fast CPU inference (~1000 sentences/sec), 384-dim embeddings, strong performance on semantic similarity benchmarks. Same model used for both clustering and retrieval keeps the embedding space consistent.

**Alternatives considered**: `all-mpnet-base-v2` (better quality, 3× slower), `bge-small-en-v1.5` (similar quality, slightly smaller).

**Trade-off**: Slightly lower retrieval quality vs. mpnet, but 3× faster — important for the "15-min reproduce" budget.

---

## D-03: Clustering = KMeans over HDBSCAN (initial pass)

**Decision**: Default to KMeans with k=12 for intent derivation, with HDBSCAN available as an option.

**Rationale**: KMeans produces exactly k clusters (predictable), is faster on 50k points, and is easier to tune (just change k). HDBSCAN finds natural clusters but produces noise points and variable cluster counts — harder to map to a fixed taxonomy.

**Trade-off**: KMeans assumes spherical clusters in embedding space, which is approximately true for sentence embeddings but not perfect. HDBSCAN is available via `--method hdbscan` flag for comparison.

---

## D-04: Weak supervision via Ollama (not a HF zero-shot model)

**Decision**: Use a local Ollama LLM (Llama-3.1-8B) for bulk labeling, not a HuggingFace zero-shot classifier.

**Rationale**: Zero-shot HF classifiers (e.g. `facebook/bart-large-mnli`) require the taxonomy labels in natural-language form and are brittle to label wording. An instruction-tuned LLM can take the full taxonomy YAML as context and produce more consistent labels.

**Trade-off**: Slower (token generation) vs. zero-shot forward pass. Mitigated by batching and running on Colab T4 for bulk labeling.

---

## D-05: Fine-tune DistilBERT (not run LLM at inference time)

**Decision**: Distill LLM labels into a DistilBERT classifier for runtime inference, not call the LLM on every message.

**Rationale**: LLM inference at 20 tok/s is too slow for a production classifier. DistilBERT is 66M params, runs in < 50ms on CPU, and achieves near-LLM accuracy after fine-tuning on LLM-generated labels.

**Trade-off**: Two-step training (label → fine-tune) vs. one-step. Adds ~45 min of compute but yields a 100× faster runtime classifier.

---

## D-06: FAISS flat index (not HNSW) for RAG retrieval

**Decision**: Use `faiss.IndexFlatIP` (exact inner-product search) for the retrieval index.

**Rationale**: The dataset slice is ~50k examples — exact search is fast enough (< 50ms). HNSW would only matter at 1M+ scale. Exact search also gives us true similarity scores for the escalation gate, while HNSW scores are approximations.

**Trade-off**: Index fits in RAM (~75MB for 50k×384 float32). If you scale to the full 3M dataset, switch to `faiss.IndexHNSWFlat`.

---

## D-07: Escalation gate = rules + signals, no LLM

**Decision**: Implement escalation as a transparent signal-fusion function, not an LLM call.

**Rationale**: (a) LLMs are a black box — you can't audit why it escalated. (b) The gate runs on *every* message, so latency matters. (c) Rules are easier to debug and tune. (d) The assignment specifically rewards explainability.

**Trade-off**: Rules may miss subtle escalation cues that an LLM would catch. Mitigated by including a sentiment/urgency score as a signal.

---

## D-08: Judge model ≠ generator model

**Decision**: Use Qwen2.5-7B-Instruct as the judge, not Llama-3.1-8B (the generator).

**Rationale**: Using the same model to evaluate its own output is a conflict of interest — it will systematically prefer its own writing style. A different model family provides more objective scoring.

**Trade-off**: Requires pulling a second Ollama model (~4.3 GB). Worth it for evaluation validity.

---

## D-09: PII redaction before embedding (not after)

**Decision**: Redact PII (handles, order numbers, phone-like strings) *before* any text is embedded or stored in the FAISS index.

**Rationale**: If PII lands in the index, it could be retrieved and echoed back in generated replies — a privacy risk. Redacting early means all downstream stages are clean.

**Trade-off**: Redaction is lossy — order numbers that are part of the complaint context get replaced with `[ORDER_ID]`. This is acceptable; the model can still understand the complaint type.

---

## D-10: Golden set sampling = stratified by intent + edge cases

**Decision**: Sample the golden set stratified across intents (ensuring ~15 examples per intent) plus a deliberate oversample of edge cases (ambiguous, multi-intent, very short, non-English).

**Rationale**: Random sampling would undersample rare intents and edge cases — exactly the failures you want to catch. Stratified sampling ensures evaluation coverage is uniform across the taxonomy.

**Trade-off**: Manual selection of edge cases is subjective. Document your edge-case criteria in the labeling notebook.

---

## D-11: Guardrails = regex over retrieved context, not LLM

**Decision**: Flag unverifiable claims by checking if specific assertions (dates, amounts, policy specifics) in the draft reply appear in the retrieved precedents — regex-based, not LLM-based.

**Rationale**: An LLM guardrail would add latency and require another model call. Regex matching against the retrieved context is fast, interpretable, and catches the most common failure mode (hallucinated refund amounts or delivery dates).

**Trade-off**: Regex may miss paraphrased claims. An LLM could catch "we'll refund you within 5 days" even if the retrieved context says "refunds processed in 3–7 business days". Accept this limitation for now; log it in Known Limitations.

---

## D-12: Config-first design (no hardcoded values)

**Decision**: Every tunable parameter (brand, sample size, thresholds, model names) lives in `config/config.yaml`, not hardcoded in scripts.

**Rationale**: The assignment requires reproducibility. A single config file means a reviewer can swap the brand or adjust thresholds without reading source code.

**Trade-off**: Slightly more boilerplate per script (load config at top).

---

## D-13: Multi-turn context via thread reconstruction

**Decision**: Reconstruct full conversation threads before classification and generation, not process individual tweets in isolation.

**Rationale**: "My package still hasn't arrived" is ambiguous in isolation. In context of "I ordered 3 weeks ago and got no update", it's clearly an `order_status_inquiry`. Thread context dramatically improves classification accuracy and reply relevance.

**Trade-off**: Thread reconstruction requires parsing the reply_tweet_id graph, which is O(n) but requires holding the full dataset in memory. Use chunked processing for the full 3M-tweet dataset.

---

*Add more decisions here as the project evolves.*
