from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import os


app = FastAPI(title="Anime Discovery ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")
GENRES_PATH = os.path.join(BASE_DIR, "top_genres.pkl")

# --------------------------------------------------
# Globals
# --------------------------------------------------

model = None
model_features: List[str] = []
top_genres: List[str] = []

# --------------------------------------------------
# Schemas
# --------------------------------------------------

class AnimeInput(BaseModel):
    episodes: int
    members: int
    genres: List[str]
    type: str
    episode_bucket: str


class PredictionResponse(BaseModel):
    predicted_score: float


# --------------------------------------------------
# Startup: Load Artifacts
# --------------------------------------------------

@app.on_event("startup")
def load_model():
    global model, model_features, top_genres

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("model.pkl not found")

    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)
    top_genres = joblib.load(GENRES_PATH)

    print("✅ Model and artifacts loaded")


# --------------------------------------------------
# Helper: Episode Bucket
# --------------------------------------------------

def bucket_episode(ep: int) -> str:
    if ep <= 1:
        return "single"
    elif ep <= 12:
        return "short"
    elif ep <= 26:
        return "medium"
    elif ep <= 60:
        return "long"
    else:
        return "very_long"


# --------------------------------------------------
# Metadata (Frontend use)
# --------------------------------------------------

@app.get("/metadata")
def get_metadata():
    return {
        "genres": top_genres,
        "episode_buckets": ["single", "short", "medium", "long", "very_long"],
        "types": ["TV", "Movie", "OVA", "ONA", "Special"]
    }


# --------------------------------------------------
# Predict Endpoint
# --------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict_score(payload: AnimeInput):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Base features
    data = {
        "episodes": payload.episodes,
        "members": payload.members
    }

    # Genre one-hot
    for genre in top_genres:
        data[genre] = 1 if genre in payload.genres else 0

    # Type one-hot
    type_col = f"type_{payload.type}"
    data[type_col] = 1

    # Episode bucket one-hot
    bucket_col = f"episode_bucket_{payload.episode_bucket}"
    data[bucket_col] = 1

    # Create DataFrame
    df = pd.DataFrame([data])

    # Ensure all columns exist
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Correct column order
    df = df[model_features]

    try:
        prediction = model.predict(df)[0]
        return {"predicted_score": round(float(prediction), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "Anime Discovery ML API is running "
    }
