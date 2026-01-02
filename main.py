# app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os
import logging
import numpy as np

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------

app = FastAPI(title="Anime Discovery ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")
GENRES_PATH = os.path.join(BASE_DIR, "top_genres.pkl")
ANIME_CSV_PATH = os.path.join(BASE_DIR, "data", "anime.csv")

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------

model = None
model_features: List[str] = []
top_genres: List[str] = []
anime_df: pd.DataFrame | None = None

# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------

class UserPreference(BaseModel):
    genres: List[str]
    preferred_episodes: int
    type: str = "TV"
    top_n: int = 10

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

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


def prepare_features_for_df(
    df: pd.DataFrame,
    top_genres: List[str],
    model_features: List[str],
) -> pd.DataFrame:
    X = pd.DataFrame(0, index=df.index, columns=model_features)

    if "episodes" in model_features:
        X["episodes"] = df["episodes"]

    if "members" in model_features:
        X["members"] = df["members"]

    for t in df["type"].unique():
        col = f"type_{t}"
        if col in model_features:
            X.loc[df["type"] == t, col] = 1

    buckets = df["episodes"].apply(bucket_episode)
    for b in ["single", "short", "medium", "long", "very_long"]:
        col = f"episode_bucket_{b}"
        if col in model_features:
            X.loc[buckets == b, col] = 1

    genre_rows = []
    for _, row in df.iterrows():
        g_list = [g.strip() for g in str(row["genre"]).split(",")]
        row_map = {g: 1 for g in g_list if g in top_genres}
        genre_rows.append(row_map)

    genre_df = pd.DataFrame(genre_rows, index=df.index).fillna(0)

    for col in genre_df.columns:
        if col in model_features:
            X[col] = genre_df[col]

    return X[model_features]

# ---------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------

@app.on_event("startup")
def load_artifacts():
    global model, model_features, top_genres, anime_df

    logger.info("🚀 Starting Anime Discovery ML API")

    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)
    top_genres = joblib.load(GENRES_PATH)

    full_df = pd.read_csv(ANIME_CSV_PATH)

    full_df["episodes"] = pd.to_numeric(full_df["episodes"], errors="coerce")
    full_df["members"] = pd.to_numeric(full_df["members"], errors="coerce")

    anime_df = full_df.dropna(
        subset=["genre", "type", "episodes", "members"]
    ).copy()

    X = prepare_features_for_df(anime_df, top_genres, model_features)

    anime_df["base_score"] = model.predict(X)

    # 🔑 normalize base score so user intent can dominate
    anime_df["base_score"] = (
        anime_df["base_score"] - anime_df["base_score"].mean()
    ) / anime_df["base_score"].std()

    anime_df["genre_set"] = anime_df["genre"].apply(
        lambda x: set(g.strip() for g in str(x).split(","))
    )

    logger.info(f"✅ Loaded {len(anime_df)} anime")

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/metadata")
def metadata():
    return {
        "genres": top_genres,
        "types": ["TV", "Movie", "OVA", "ONA", "Special"],
    }

# ---------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------

@app.post("/surprise")
def surprise(preference: UserPreference):
    if anime_df is None or anime_df.empty:
        raise HTTPException(500, "Server not ready")

    user_genres = set(preference.genres)

    # 1️⃣ Candidate filtering (kills global bias)
    candidates = anime_df[
        anime_df["genre_set"].apply(lambda g: len(g & user_genres) > 0)
    ]

    if candidates.empty:
        candidates = anime_df.sample(300)

    # 2️⃣ User-centric scoring
    def compute_final_score(row):
        score = row["base_score"]

        genre_overlap = len(row["genre_set"] & user_genres)
        score += genre_overlap * 0.6

        ep_distance = abs(row["episodes"] - preference.preferred_episodes)
        score -= ep_distance * 0.015

        if row["type"] == preference.type:
            score += 0.3

        return score

    candidates = candidates.copy()
    candidates["final_score"] = candidates.apply(compute_final_score, axis=1)

    # 3️⃣ Diversity sampling
    pool = candidates.sort_values(
        "final_score", ascending=False
    ).head(preference.top_n * 3)

    final = pool.sample(
        min(preference.top_n, len(pool))
    )

    results = [
        {
            "anime_id": int(r.anime_id),
            "title": r.name,
            "type": r.type,
            "episodes": int(r.episodes),
            "members": int(r.members),
            "genres": list(r.genre_set),
            "predicted_rating": round(r.base_score, 2),
            "final_score": round(r.final_score, 2),
        }
        for _, r in final.iterrows()
    ]

    return {
        "recommended_for": preference.genres,
        "results": results,
    }
