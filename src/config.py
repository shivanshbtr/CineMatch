"""
CineMatch — Centralized Configuration

Per docs/09_Development_Guide.md §6 (Implementation Guidelines):
"Config values (thresholds, α weight table, Top-K bounds) are centralized
in a single config.py, not hardcoded across files."
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_ML_DIR = ROOT_DIR / "data" / "raw" / "ml-latest-small"
RAW_TMDB_DIR = ROOT_DIR / "data" / "raw" / "tmdb-5000"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

RAW_ML_DIR_FILES = {
    "movies": RAW_ML_DIR / "movies.csv",
    "ratings": RAW_ML_DIR / "ratings.csv",
    "links": RAW_ML_DIR / "links.csv",
    "tags": RAW_ML_DIR / "tags.csv",
}

RAW_TMDB_FILES = {
    "movies": RAW_TMDB_DIR / "tmdb_5000_movies.csv",
    "credits": RAW_TMDB_DIR / "tmdb_5000_credits.csv",
}

PROCESSED_FILES = {
    "movies_master": PROCESSED_DIR / "movies_master.csv",
    "ratings_clean": PROCESSED_DIR / "ratings_clean.csv",
}

MODEL_ARTIFACTS = {
    "tfidf_matrix": MODELS_DIR / "tfidf_matrix.pkl",
    "count_matrix": MODELS_DIR / "count_matrix.pkl",
    "svd_model": MODELS_DIR / "svd_model.pkl",
    "movie_index_map": MODELS_DIR / "movie_index_map.pkl",
}

# ---------------------------------------------------------------------------
# Cold-start / hybrid scoring config
# Per docs/07_Algorithms_and_Scoring_Logic.md §4-5
# ---------------------------------------------------------------------------
MIN_RATINGS_THRESHOLD = 5  # below this, a user is treated as cold-start

# α (collaborative weight) lookup table, keyed by (lower_bound_inclusive, upper_bound_inclusive)
ALPHA_TABLE = [
    ((0, 0), 0.0),
    ((1, 4), 0.2),
    ((5, 19), 0.5),
    ((20, float("inf")), 0.8),
]

# ---------------------------------------------------------------------------
# API / recommendation bounds
# Per docs/04_Software_Requirements_Specification.md §7 (Business Rules)
# ---------------------------------------------------------------------------
TOP_K_DEFAULT = 10
TOP_K_MIN = 5
TOP_K_MAX = 20

# ---------------------------------------------------------------------------
# Content-based feature engineering
# Per docs/02_Feature_Engineering.md §3
# ---------------------------------------------------------------------------
TOP_N_CAST = 3  # top-N cast members included in the metadata soup
