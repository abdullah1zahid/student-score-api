from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Student Math Score Predictor API",
    description="ML model for predicting student math scores",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Load
model = joblib.load("student_model.pkl")

class StudentInput(BaseModel):
    gender: str = Field(..., example="female")
    race_ethnicity: str = Field(..., alias="race/ethnicity", example="group B")
    parental_level_of_education: str = Field(..., alias="parental level of education", example="bachelor's degree")
    lunch: str = Field(..., example="standard")
    test_preparation_course: str = Field(..., alias="test preparation course", example="none")
    reading_score: float = Field(..., example=72)
    writing_score: float = Field(..., example=74)

    class Config:
        populate_by_name = True

@app.get("/")
def home():
    return {"message": "API is working perfectly!"}

@app.post("/predict")
def predict_score(data: StudentInput):
    try:
        # 1. Raw input dictionary
        input_dict = {
            "gender": [data.gender],
            "race/ethnicity": [data.race_ethnicity],
            "parental level of education": [data.parental_level_of_education],
            "lunch": [data.lunch],
            "test preparation course": [data.test_preparation_course],
            "reading score": [data.reading_score],
            "writing score": [data.writing_score]
        }
        df = pd.DataFrame(input_dict)

        # 2. Convert categories to One-Hot Encoding (Dummies)
        df_encoded = pd.get_dummies(df)

        # 3. Model ke feature names ke sath align karein
        if hasattr(model, "feature_names_in_"):
            df_encoded = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

        # 4. Predict
        prediction = model.predict(df_encoded)[0]
        score = round(float(prediction), 2)

        # 5. Logic
        status = "Pass" if score >= 50 else "Fail"
        if score >= 80:
            grade = "A"
        elif score >= 70:
            grade = "B"
        elif score >= 60:
            grade = "C"
        elif score >= 50:
            grade = "D"
        else:
            grade = "F"

        return {
            "status": "success",
            "predicted_math_score": score,
            "result_status": status,
            "grade": grade,
            "inputs_received": {
                "gender": data.gender,
                "reading_score": data.reading_score,
                "writing_score": data.writing_score
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")