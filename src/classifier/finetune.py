"""
src/classifier/finetune.py
───────────────────────────
Fine-tune DistilBERT on weak-supervision labels from the LLM.

Design decision (D-05): We distill the LLM labels into a tiny classifier
so runtime inference is fast (< 50ms CPU) instead of calling the LLM
on every message.

WHERE IT RUNS: Colab/Kaggle T4 GPU. Expected runtime: ~25 min for 10k examples.
Do NOT run this locally on CPU — it will take hours.

COMMAND (in Colab after uploading labeled_training.csv):
  python -c "from src.classifier.finetune import run_finetune; run_finetune()"
OR open notebooks/03_finetune_classifier.ipynb

OUTPUT: models/classifier/ (HuggingFace model directory, ~250MB)
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def run_finetune(
    labeled_csv: str = "data/processed/labeled_training.csv",
    model_dir: str = "models/classifier",
    base_model: str = "distilbert-base-uncased",
    max_length: int = 128,
    num_epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    eval_split: float = 0.1,
    random_seed: int = 42,
    taxonomy_path: str = "config/intent_taxonomy.yaml",
):
    """
    Fine-tune DistilBERT on weak-supervision labels.

    Args:
        labeled_csv: Path to CSV with columns [text, intent].
        model_dir: Where to save the fine-tuned model.
        base_model: HuggingFace model ID for the base.
        max_length: Max token length for truncation.
        num_epochs: Training epochs.
        batch_size: Per-device training batch size.
        learning_rate: AdamW learning rate.
        eval_split: Fraction of data for evaluation.
        random_seed: For reproducibility.
        taxonomy_path: Intent taxonomy YAML.
    """
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    import evaluate

    from src.intents.taxonomy import IntentTaxonomy

    taxonomy = IntentTaxonomy(taxonomy_path)
    label2id = {name: i for i, name in enumerate(taxonomy.intent_names)}
    id2label = {i: name for name, i in label2id.items()}

    # Load labeled data
    logger.info(f"Loading labeled data from {labeled_csv}...")
    df = pd.read_csv(labeled_csv)
    df = df.dropna(subset=["text", "intent"])
    df = df[df["intent"].isin(label2id)]  # drop any bad labels
    df["label"] = df["intent"].map(label2id)

    logger.info(f"Loaded {len(df):,} examples across {df['intent'].nunique()} intents.")
    logger.info(f"Label distribution:\n{df['intent'].value_counts()}")

    # Train/eval split
    from sklearn.model_selection import train_test_split
    train_df, eval_df = train_test_split(df, test_size=eval_split, random_state=random_seed, stratify=df["label"])

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df[["text", "label"]].reset_index(drop=True))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(taxonomy),
        id2label=id2label,
        label2id=label2id,
    )

    # Metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
        return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}

    # Training arguments
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        seed=random_seed,
        report_to="none",  # disable wandb etc.
        fp16=torch.cuda.is_available(),  # use fp16 on GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # Save final model
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")

    # Print final eval metrics
    metrics = trainer.evaluate()
    logger.info(f"Final eval metrics: {metrics}")
    return metrics
