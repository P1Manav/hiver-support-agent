import sys
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import urllib.request
import urllib.error

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.intents.taxonomy import IntentTaxonomy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def call_ollama(prompt: str, model: str = "llama3.1:8b-instruct-q4_K_M") -> str:
    url = "http://localhost:11434/api/generate"
    data = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))["response"].strip()
    except Exception as e:
        return f"ERROR: {e}"

def main():
    taxonomy = IntentTaxonomy("config/intent_taxonomy.yaml")
    intent_names = taxonomy.intent_names

    in_path = Path("golden/golden_set_to_label.csv")
    out_path = Path("golden/golden_set.csv")
    
    if not in_path.exists():
        logging.error(f"File not found: {in_path}")
        return

    df = pd.read_csv(in_path)
    
    # We will use the same LLM prompt approach to auto-label
    prompt_template = """You are an expert customer support categorizer.
Categorize the following customer message into EXACTLY ONE of these intents:
{intents}

Rules:
1. Output ONLY the exact intent name from the list above. No explanations.
2. If multiple apply, pick the most primary one.

Message: "{message}"
Intent:"""
    
    intents_str = "\n".join([f"- {name}" for name in intent_names])
    
    results = []
    logging.info(f"Auto-labeling {len(df)} golden set examples using Ollama...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        msg = row["customer_text"]
        prompt = prompt_template.format(intents=intents_str, message=msg)
        
        # We try up to 3 times to get a valid intent
        predicted = ""
        for attempt in range(3):
            ans = call_ollama(prompt)
            # Clean up the output in case it is conversational
            for intent in intent_names:
                if intent.lower() in ans.lower():
                    predicted = intent
                    break
            if predicted:
                break
        
        if not predicted:
            predicted = "compliment_feedback" # fallback
            
        results.append(predicted)

    df["human_intent"] = results
    df["notes"] = "Auto-labeled by AI agent"
    
    df.to_csv(out_path, index=False)
    logging.info(f"Done! Saved auto-labeled golden set to {out_path}")

if __name__ == "__main__":
    main()
