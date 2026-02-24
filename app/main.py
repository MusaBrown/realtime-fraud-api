from fastapi import FastAPI
import numpy as np

app = FastAPI(title="Fraud Detection API")

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: dict):
    # Simplified prediction for demo
    amount = transaction.get("amount", 0)
    fraud_prob = min(amount / 1000, 0.99)
    return {
        "fraud_probability": fraud_prob,
        "is_fraud": fraud_prob > 0.5
    }
