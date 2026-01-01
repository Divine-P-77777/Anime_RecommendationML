from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Anime Discovery ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")
GENRES_PATH = os.path.join(BASE_DIR, "top_genres.pkl")

ANIME_CSV_PATH = os.path.join(BASE_DIR, "data", "anime.csv")


model = None
model_features: List[str] = []
top_genres: List[str] = []
anime_df: pd.DataFrame | None = None


# Schemas

class UserPreference(BaseModel):
    genres: List[str]
    preferred_episodes: int
    type: str = "TV"
    top_n: int = 10

# Helpers

def bucket_episode(ep: int) -> str:
    if ep <= 1:
        return "single"
    if ep <= 12:
        return "short"
    if ep <= 26:
        return "medium"
    if ep <= 60:
        return "long"
    return "very_long"

def prepare_features_for_df(df: pd.DataFrame, top_genres: List[str], model_features: List[str]) -> pd.DataFrame:
    """
    Vectorized feature engineering to prepare input for the model.
    """
    logger.info("⚙️ Starting feature engineering...")
    
    # Initialize feature matrix with zeros
    X = pd.DataFrame(0, index=df.index, columns=model_features)
    
    # 1. Episodes & Members (if in features)
    if "episodes" in model_features:
        X["episodes"] = df["episodes"]
    if "members" in model_features:
        X["members"] = df["members"]

    # 2. Type (One-hot)
    for t in df["type"].unique():
        col = f"type_{t}"
        if col in model_features:
            X.loc[df["type"] == t, col] = 1

    # 3. Episode Buckets
    buckets = df["episodes"].apply(bucket_episode)
    for b in ["single", "short", "medium", "long", "very_long"]:
        col = f"episode_bucket_{b}"
        if col in model_features:
            X.loc[buckets == b, col] = 1

    # 4. Genres (One-hot)
  
    logger.info("⚙️ Processing genres...")

    genre_rows = []
    for _, row in df.iterrows():
        g_list = [g.strip() for g in str(row["genre"]).split(",")]
        row_map = {}
        for g in g_list:
            if g in top_genres:
                row_map[g] = 1
        genre_rows.append(row_map)
        
    genre_df = pd.DataFrame(genre_rows, index=df.index).fillna(0)
    
    # Assign back to X
    for col in genre_df.columns:
        if col in model_features:
            X[col] = genre_df[col]


    for col in model_features:
        if col not in X.columns:
            X[col] = 0
            
    return X[model_features]

# Startup

@app.on_event("startup")
def load_artifacts():
    global model, model_features, top_genres, anime_df

    logger.info("🚀 Starting Anime Discovery ML API")

    try:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("model.pkl not found")

        model = joblib.load(MODEL_PATH)
        model_features = joblib.load(FEATURES_PATH)
        top_genres = joblib.load(GENRES_PATH)
        
        # Load CSV
        if not os.path.exists(ANIME_CSV_PATH):
            logger.error(f"❌ CSV not found at {ANIME_CSV_PATH}")
            anime_df = pd.DataFrame()
            return

        logger.info(f"📂 Loading anime data from {ANIME_CSV_PATH}")
        full_df = pd.read_csv(ANIME_CSV_PATH)
        
        # Cleaning
        full_df["episodes"] = pd.to_numeric(full_df["episodes"], errors='coerce')
        full_df["members"] = pd.to_numeric(full_df["members"], errors='coerce')
        
        # Drop invalid rows
        anime_df = full_df.dropna(subset=["genre", "type", "episodes", "members"]).copy()
        
        logger.info(f"✅ Data Loaded: {len(anime_df)} valid rows")

 
        # OPTIMIZATION: Predict SCORES for ALL anime at 
        if not anime_df.empty:
            X = prepare_features_for_df(anime_df, top_genres, model_features)
            
            logger.info("🔮 Predicting base scores for all anime...")
            anime_df["base_score"] = model.predict(X)
            
       
            anime_df["genre_set"] = anime_df["genre"].apply(lambda x: set([g.strip() for g in str(x).split(",")]))
            
            logger.info("✅ Startup sequence complete. API Ready.")
            
    except Exception as e:
        logger.error(f"❌ Startup Error: {e}")
 
        raise e

# Routes

@app.get("/")
def root():
    return {"status": "ok", "message": "Anime Discovery ML API running 🚀"}


@app.get("/metadata")
def metadata():
    return {
        "genres": top_genres,
        "types": ["TV", "Movie", "OVA", "ONA", "Special"],
        "episode_buckets": ["single", "short", "medium", "long", "very_long"],
    }

# Surprise Recommendation (CORE FEATURE)

@app.post("/surprise")
def surprise(preference: UserPreference):
    """
    Predicts top anime names based on user preference.
    Uses pre-calculated ML scores + dynamic preference boosting.
    """

    if model is None or anime_df is None or anime_df.empty:
        raise HTTPException(status_code=500, detail="Server not ready (model/data missing)")

    logger.info("🎲 /surprise request received")
    logger.info(f"User preference → {preference.dict()}")

    try:
 
        user_genres_set = set(preference.genres)
        
 
        def calc_boost(anime_genres):
            return len(anime_genres.intersection(user_genres_set)) * 0.15
            

        boosts = anime_df["genre_set"].apply(calc_boost)
        final_scores = anime_df["base_score"] + boosts
        
 
        top_indices = final_scores.nlargest(preference.top_n).index

        results = []
        for idx in top_indices:
            row = anime_df.loc[idx]
            results.append({
                "anime_id": int(row["anime_id"]),
                "title": row["name"],
                "type": row["type"],
                "episodes": int(row["episodes"]),
                "members": int(row["members"]),
                "genres": list(row["genre_set"]),
                "predicted_rating": round(row["base_score"], 2),
                "final_score": round(final_scores[idx], 2),
            })

        logger.info(f"🏆 Recommendation generated: Top {len(results)}")
        logger.info(preference.top_n)
        return {
            "recommended_for": preference.genres,
            "top_n": preference.top_n,
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")
