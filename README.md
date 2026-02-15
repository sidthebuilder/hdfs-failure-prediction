# HDFS Failure Prediction Model

[![Hugging Face](https://img.shields.io/badge/Model-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/Sha09090/hdfs-failure-prediction)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

A text classification model that detects failure-indicative patterns in HDFS (Hadoop Distributed File System) log entries. This project demonstrates end-to-end machine learning workflow from data preprocessing to model deployment.

## Overview

This repository contains a fine-tuned DistilBERT model for binary classification of HDFS logs. The model learns to identify logs containing failure indicators (errors, exceptions, termination messages) versus normal operational logs.

**Key Characteristics:**
- **Model Architecture:** DistilBERT (distilbert-base-uncased) fine-tuned for sequence classification
- **Dataset:** ~575k HDFS log entries with balanced failure/normal samples
- **Test Performance:** 99.95% accuracy on 74k held-out samples
- **Inference Speed:** ~15ms per log entry on GPU

## Performance Metrics

The model was evaluated on a held-out test set of 74,393 log entries. Results reflect performance on this specific dataset with its characteristic failure patterns.

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Accuracy** | 99.95% | High performance on keyword-based failure detection |
| **Precision** | 99.99% | Very few false positives (11 in 74k samples) |
| **Recall** | 99.96% | Missed 26 actual failures |
| **F1-Score** | 99.97% | Balanced precision/recall trade-off |

### Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

**Test Set Breakdown:**
- True Positives: 73,392 (correctly identified failures)
- False Positives: 11 (normal logs flagged as failures)
- False Negatives: 26 (failures missed)
- True Negatives: 981 (correctly identified normal logs)

### Important Caveats

**What the model does well:**
- Detects logs containing explicit failure keywords (ERROR, CRITICAL, Exception, etc.)
- Generalizes to similar error message formats
- Fast inference suitable for real-time monitoring

**Known limitations:**
- Performance may degrade on logs with novel failure patterns not seen in training data
- Does not perform temporal analysis (sequence of logs over time)
- Not designed for silent failures that don't generate error logs
- High accuracy is partly due to distinctive keyword patterns in the dataset

## Usage

### Installation

```bash
pip install torch transformers
```

### Inference Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model from Hugging Face Hub
model_name = "Sha09090/hdfs-failure-prediction"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_failure(log_text):
    inputs = tokenizer(log_text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get probability of failure (Class 1)
    probability = torch.softmax(logits, dim=1)[0][1].item()
    prediction = "FAILURE" if probability > 0.5 else "NORMAL"
    
    return prediction, probability

# Example
log = "PacketResponder: error for block blk_12345 terminating"
status, confidence = predict_failure(log)
print(f"Status: {status} | Confidence: {confidence:.2%}")
```
```

## Repository Structure

```
├── assets/           # Performance visualizations
├── src/
│   ├── train.py      # Training script
│   └── inference.py  # Inference utilities
└── README.md
```

## License

MIT License - See LICENSE file for details.

## Links

- **Model Weights:** [Hugging Face Hub](https://huggingface.co/Sha09090/hdfs-failure-prediction)
- **Author:** [Shashank](https://github.com/Sidthebuilder)
