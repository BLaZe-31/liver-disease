from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import asyncio
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware


# Load the saved model
model = joblib.load("liver_model.pkl")

# Define the API app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)
# Define the input structure
class LiverInput(BaseModel):
    age: int
    gender: int  # 0 for female, 1 for male
    total_bilirubin: float
    direct_bilirubin: float
    alkphos: int
    sgpt: int
    sgot: int

# Define the prediction endpoint
@app.post("/predict")
def predict(data: LiverInput):
    input_data = np.array([[data.age, data.gender, data.total_bilirubin,
                            data.direct_bilirubin, data.alkphos, data.sgpt, data.sgot]])
    
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][prediction] * 100

    if prediction == 1:
        return {
            "prediction": "At risk of liver disease",
            "confidence": f"{confidence:.2f}%"
        }
    else:
        return {
            "prediction": "Healthy",
            "confidence": f"{confidence:.2f}%"
        }
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

