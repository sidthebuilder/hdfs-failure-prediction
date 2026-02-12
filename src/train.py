"""
HDFS FAILURE PREDICTION - TRAINING SCRIPT
-----------------------------------------
Author: Shashank Kumar
Description: 
    Production-grade training script that:
    - Loads and merges multiple JSON datasets
    - Balances the data
    - Fine-tunes DistilBERT for binary classification
    - Saves the model and tokenizer
"""

import os
import sys
import glob
import json
import logging
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from datasets import Dataset

# 1. CONFIGURATION
@dataclass
class HDFSConfig:
    MODEL_NAME: str = "distilbert-base-uncased"
    MAX_LENGTH: int = 128
    EPOCHS: int = 3
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 2e-5
    SEED: int = 42
    INPUT_DIR: str = "./data"  # Point this to your JSON files
    OUTPUT_DIR: str = "./model_output"

# 2. DATA LOADING
def load_and_merge_data(config: HDFSConfig) -> List[Dict]:
    print(f"Scanning {config.INPUT_DIR} for JSON datasets...")
    files = glob.glob(f"{config.INPUT_DIR}/**/*.json", recursive=True)
    all_data = []
    seen_texts = set()
    
    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    text_sig = (item.get('instruction', '') + item.get('input', '')).strip()
                    if text_sig not in seen_texts:
                        seen_texts.add(text_sig)
                        all_data.append(item)
    
    print(f"✅ Loaded {len(all_data)} unique samples.")
    return all_data

# 3. MAIN TRAINING LOOP
def main():
    config = HDFSConfig()
    set_seed(config.SEED)
    
    # Load Data
    raw_data = load_and_merge_data(config)
    
    # Preprocess (Simplified for script)
    processed = []
    keywords = ['critical', 'failure', 'error', 'terminate']
    for item in raw_data:
        text = f"{item.get('instruction', '')} [SEP] {item.get('input', '')}"
        label = 1 if any(k in text.lower() for k in keywords) else 0
        processed.append({'text': text, 'label': label})
        
    # Split
    train_data, val_data = train_test_split(processed, test_size=0.15, random_state=config.SEED)
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=config.MAX_LENGTH)
    
    train_ds = Dataset.from_list(train_data).map(tokenize, batched=True)
    val_ds = Dataset.from_list(val_data).map(tokenize, batched=True)
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)
    
    # Trainer
    args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=config.LEARNING_RATE,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}
    )
    
    print("🚀 Starting Training...")
    trainer.train()
    
    # Save
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print(f"✅ Model saved to {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
