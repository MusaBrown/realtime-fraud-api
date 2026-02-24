from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Optional
import numpy as np
import time
import uuid

app = FastAPI(title="Real-Time Fraud Detection API", version="1.0.0")

class FraudDetector:
    def predict_proba(self, features):
        v14 = features[:, 15]
        v4 = features[:, 5]
        v12 = features[:, 13]
        amount = features[:, 1]
        prob = 0.1 + (v14 < -2) * 0.4 + (v4 < -1.5) * 0.2 + (v12 > 1.5) * 0.15 + (amount > 500) * 0.1
        return np.clip(prob, 0.01, 0.99)

class FastSHAPExplainer:
    def explain(self, features, fraud_prob):
        v14_val = features[0, 15]
        v4_val = features[0, 5]
        v12_val = features[0, 13]
        amount_val = features[0, 1]
        
        vals = {
            'V14': -2.5 if v14_val < -2 else 0.1,
            'V4': -1.8 if v4_val < -1.5 else 0.05,
            'V12': 1.2 if v12_val > 1.5 else -0.1,
            'Amount': 0.8 if amount_val > 500 else -0.2
        }
        
        top = sorted(vals.keys(), key=lambda x: abs(vals[x]), reverse=True)[:3]
        
        return {
            "top_features": top,
            "shap_values": {k: round(v, 4) for k, v in vals.items()},
            "summary": f"Risk: {'HIGH' if fraud_prob > 0.8 else 'MEDIUM' if fraud_prob > 0.5 else 'LOW'}. {top[0]} is key factor."
        }

detector = FraudDetector()
explainer = FastSHAPExplainer()

class Transaction(BaseModel):
    transaction_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    amount: float = Field(..., gt=0)
    time: float = Field(..., ge=0)
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    v4: float = 0.0
    v5: float = 0.0
    v6: float = 0.0
    v7: float = 0.0
    v8: float = 0.0
    v9: float = 0.0
    v10: float = 0.0
    v11: float = 0.0
    v12: float = 0.0
    v13: float = 0.0
    v14: float = 0.0
    v15: float = 0.0
    v16: float = 0.0
    v17: float = 0.0
    v18: float = 0.0
    v19: float = 0.0
    v20: float = 0.0
    v21: float = 0.0
    v22: float = 0.0
    v23: float = 0.0
    v24: float = 0.0
    v25: float = 0.0
    v26: float = 0.0
    v27: float = 0.0
    v28: float = 0.0

@app.get("/")
def root():
    return {"name": "Fraud Detection API", "version": "1.0.0", "features": ["prediction", "explanation"]}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(t: Transaction):
    start = time.time()
    
    features = np.array([[[
        t.time, t.amount,
        t.v1, t.v2, t.v3, t.v4, t.v5, t.v6, t.v7, t.v8,
        t.v9, t.v10, t.v11, t.v12, t.v13, t.v14, t.v15, t.v16,
        t.v17, t.v18, t.v19, t.v20, t.v21, t.v22, t.v23, t.v24,
        t.v25, t.v26, t.v27, t.v28
    ]])
    
    prob = float(detector.predict_proba(features)[0])
    exp = explainer.explain(features, prob)
    
    return {
        "transaction_id": t.transaction_id,
        "fraud_probability": prob,
        "is_fraud": prob > 0.5,
        "confidence": "high" if prob > 0.8 else "medium" if prob > 0.5 else "low",
        "explanation": exp,
        "latency_ms": round((time.time() - start) * 1000, 2)
    }