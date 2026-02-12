import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# You can change this to a local path if you have the model downloaded
MODEL_NAME = "Sha09090/hdfs-failure-prediction"

class HDFSPredictor:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device.upper()}")

    def predict(self, log_text, threshold=0.5):
        inputs = self.tokenizer(
            log_text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            failure_prob = probs[0][1].item()
            
        status = "FAILURE ❌" if failure_prob > threshold else "NORMAL ✅"
        return {
            "status": status,
            "confidence": failure_prob,
            "log": log_text
        }

if __name__ == "__main__":
    # Example Usage
    predictor = HDFSPredictor()
    
    test_logs = [
        "PacketResponder: error for block blk_12345 terminating",
        "Received block blk_9999 of size 512 from /10.10.10.10",
        "Verification failed for block blk_5555"
    ]
    
    print("\n--- LIVE PREDICTIONS ---")
    for log in test_logs:
        result = predictor.predict(log)
        print(f"Log: {result['log']}")
        print(f"Result: {result['status']} (Conf: {result['confidence']:.2%})\n")
