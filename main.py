import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# 1. FastAPI App initialize karein
app = FastAPI(
    title="Student Math Score Predictor API",
    description="ML model for predicting student math scores based on demographic and academic features.",
    version="1.0.0",
)

# 2. CORS Middleware (Lovable aur doosre web frontends se connect hone ke liye zaroori hai)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Real apps mein frontend ka specific domain dete hain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Model load karein
MODEL_FILE = "student_model.pkl"

if os.path.exists(MODEL_FILE):
  model_pipeline = joblib.load(MODEL_FILE)
else:
  model_pipeline = None


# 4. Pydantic Input Schema
class StudentInput(BaseModel):
  gender: str = Field(..., example="female")
  race_ethnicity: str = Field(
      ..., alias="race/ethnicity", example="group B"
  )
  parental_education: str = Field(
      ..., alias="parental level of education", example="bachelor's degree"
  )
  lunch: str = Field(..., example="standard")
  test_prep_course: str = Field(
      ..., alias="test preparation course", example="none"
  )
  reading_score: float = Field(..., ge=0, le=100, example=72.0)
  writing_score: float = Field(..., ge=0, le=100, example=74.0)

  class Config:
    populate_by_name = True


# 5. Root endpoint (Health check ke liye)
@app.get("/")
def home():
  return {
      "status": "online",
      "message": "Student Score Predictor API is running successfully!",
  }


# 6. Predict endpoint
@app.post("/predict")
def predict_score(data: StudentInput):
  if model_pipeline is None:
    raise HTTPException(
        status_code=500,
        detail="Model file 'student_model.pkl' not found on server.",
    )

  # Input ko DataFrame mein convert karein
  input_data = {
      "gender": [data.gender],
      "race/ethnicity": [data.race_ethnicity],
      "parental level of education": [data.parental_education],
      "lunch": [data.lunch],
      "test preparation course": [data.test_prep_course],
      "reading score": [data.reading_score],
      "writing score": [data.writing_score],
  }
  df_input = pd.DataFrame(input_data)

  try:
    prediction = model_pipeline.predict(df_input)[0]
    final_score = round(max(0, min(100, float(prediction))), 2)

    return {
        "status": "success",
        "predicted_math_score": final_score,
        "inputs_received": {
            "gender": data.gender,
            "reading_score": data.reading_score,
            "writing_score": data.writing_score,
        },
    }
  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")