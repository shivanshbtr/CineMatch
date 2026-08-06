"""
CineMatch — Data Preprocessing Pipeline

Loads MovieLens (ml-latest-small) and TMDB 5000 raw data, cleans it,
joins it on tmdbId per docs/00_Dataset_Sources.md §5 (Joining Strategy),
and produces the processed artifacts consumed by src/content_based.py
and src/collaborative_filtering.py.

Output schema follows docs/05_Database_Design_and_ER_Diagram.md §2.1 (Movies).

Run: python src/data_preprocessing.py
"""

import ast
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


# ---------------------------------------------------------------------------
# Helpers for parsing TMDB's JSON-stringified columns
# ---------------------------------------------------------------------------

def _safe_literal_eval(value):
    """TMDB stores genres/keywords/cast/crew as stringified Python lists of dicts."""
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def _extract_names(list_of_dicts, key="name"):
    return [d[key] for d in list_of_dicts if key in d]


def _extract_top_cast(cast_json, top_n=config.TOP_N_CAST):
    cast_list = _safe_literal_eval(cast_json)
    names = _extract_names(cast_list, "name")
    return names[:top_n]


def _extract_director(crew_json):
    crew_list = _safe_literal_eval(crew_json)
    for member in crew_list:
        if member.get("job") == "Director":
            return member.get("name")
    return None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_movielens():
    movies = pd.read_csv(config.RAW_ML_DIR_FILES["movies"])
    ratings = pd.read_csv(config.RAW_ML_DIR_FILES["ratings"])
    links = pd.read_csv(config.RAW_ML_DIR_FILES["links"])
    return movies, ratings, links


def load_tmdb():
    tmdb_movies = pd.read_csv(config.RAW_TMDB_FILES["movies"])
    tmdb_credits = pd.read_csv(config.RAW_TMDB_FILES["credits"])
    return tmdb_movies, tmdb_credits


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

def clean_movielens_movies(movies, links):
    """Dedup by movieId, extract release_year from title, attach tmdb/imdb IDs."""
    movies = movies.drop_duplicates(subset="movieId").copy()

    # Extract release year from "Title (YYYY)" pattern; NaN if not found
    year_extracted = movies["title"].str.extract(r"\((\d{4})\)\s*$")
    movies["release_year"] = pd.to_numeric(year_extracted[0], errors="coerce")

    movies = movies.merge(links[["movieId", "imdbId", "tmdbId"]], on="movieId", how="left")
    return movies


def clean_tmdb(tmdb_movies, tmdb_credits):
    tmdb_movies = tmdb_movies.drop_duplicates(subset="id").copy()
    tmdb_credits = tmdb_credits.drop_duplicates(subset="movie_id").copy()

    # Fill missing text fields per docs/02_Feature_Engineering.md §7
    for col in ["overview"]:
        tmdb_movies[col] = tmdb_movies[col].fillna("")

    tmdb_movies["genre_names"] = tmdb_movies["genres"].apply(
        lambda x: _extract_names(_safe_literal_eval(x))
    )
    tmdb_movies["keyword_names"] = tmdb_movies["keywords"].apply(
        lambda x: _extract_names(_safe_literal_eval(x))
    )

    tmdb_credits["top_cast"] = tmdb_credits["cast"].apply(_extract_top_cast)
    tmdb_credits["director"] = tmdb_credits["crew"].apply(_extract_director)

    tmdb = tmdb_movies.merge(
        tmdb_credits[["movie_id", "top_cast", "director"]],
        left_on="id",
        right_on="movie_id",
        how="left",
    )
    return tmdb


# ---------------------------------------------------------------------------
# Join (MovieLens as the spine — every MovieLens movie is retained;
# unmatched movies simply have NULL content-features, per
# docs/00_Dataset_Sources.md §5, point 4)
# ---------------------------------------------------------------------------

def build_movies_master(ml_movies, tmdb):
    merged = ml_movies.merge(
        tmdb[
            ["id", "overview", "genre_names", "keyword_names", "top_cast", "director",
             "vote_average", "vote_count"]
        ],
        left_on="tmdbId",
        right_on="id",
        how="left",
    )

    merged["has_content_features"] = merged["overview"].notna() & (merged["id"].notna())

    # Prefer MovieLens's pipe-separated genres as the canonical `genres` column
    # (per docs/05_Database_Design_and_ER_Diagram.md §2.1: "genres TEXT NOT NULL").
    # TMDB genre_names supplements it when available but never overrides it.
    merged["genres"] = merged["genres"].replace("(no genres listed)", "")

    # Fill content fields for unmatched movies so downstream vectorization never breaks
    merged["overview"] = merged["overview"].fillna("")
    merged["keyword_names"] = merged["keyword_names"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    merged["top_cast"] = merged["top_cast"].apply(lambda x: x if isinstance(x, list) else [])
    merged["director"] = merged["director"].fillna("")

    master = merged.rename(
        columns={"movieId": "movie_id", "tmdbId": "tmdb_id", "imdbId": "imdb_id"}
    )[
        [
            "movie_id",
            "tmdb_id",
            "imdb_id",
            "title",
            "genres",
            "overview",
            "keyword_names",
            "top_cast",
            "director",
            "release_year",
            "vote_average",
            "vote_count",
            "has_content_features",
        ]
    ]

    # Build the "metadata soup" per docs/02_Feature_Engineering.md §3
    def make_soup(row):
        genre_tokens = row["genres"].replace("|", " ").replace("-", "") if row["genres"] else ""
        keyword_tokens = " ".join(str(k).replace(" ", "") for k in row["keyword_names"])
        cast_tokens = " ".join(str(c).replace(" ", "") for c in row["top_cast"])
        director_token = str(row["director"]).replace(" ", "")
        return " ".join([genre_tokens, keyword_tokens, cast_tokens, director_token]).strip()

    master["metadata_soup"] = master.apply(make_soup, axis=1)

    return master


def clean_ratings(ratings, valid_movie_ids):
    """Drop ratings pointing at movie IDs that don't exist in the master table."""
    before = len(ratings)
    ratings_clean = ratings[ratings["movieId"].isin(valid_movie_ids)].copy()
    ratings_clean = ratings_clean.rename(columns={"movieId": "movie_id", "userId": "user_id"})
    dropped = before - len(ratings_clean)
    if dropped:
        print(f"  Dropped {dropped} ratings referencing unknown movie IDs")
    return ratings_clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw datasets...")
    ml_movies, ratings, links = load_movielens()
    tmdb_movies, tmdb_credits = load_tmdb()

    print("Cleaning MovieLens movies...")
    ml_movies = clean_movielens_movies(ml_movies, links)

    print("Cleaning TMDB movies + credits...")
    tmdb = clean_tmdb(tmdb_movies, tmdb_credits)

    print("Joining MovieLens <-> TMDB on tmdbId...")
    movies_master = build_movies_master(ml_movies, tmdb)

    print("Cleaning ratings...")
    ratings_clean = clean_ratings(ratings, set(movies_master["movie_id"]))

    # --- Save ---
    movies_master.to_csv(config.PROCESSED_FILES["movies_master"], index=False)
    ratings_clean.to_csv(config.PROCESSED_FILES["ratings_clean"], index=False)

    # --- EDA summary (also feeds justification.md's "Data understanding & EDA" claim) ---
    n_total = len(movies_master)
    n_with_content = int(movies_master["has_content_features"].sum())
    n_ratings = len(ratings_clean)
    n_users = ratings_clean["user_id"].nunique()
    sparsity = 1 - (n_ratings / (n_users * n_total))

    print("\n=== Preprocessing Summary ===")
    print(f"Total movies (MovieLens spine):     {n_total}")
    print(f"Movies with TMDB content features:  {n_with_content} ({n_with_content/n_total:.1%})")
    print(f"Movies collaborative-only:          {n_total - n_with_content}")
    print(f"Total ratings:                      {n_ratings}")
    print(f"Unique users:                       {n_users}")
    print(f"User-item matrix sparsity:          {sparsity:.4%}")
    print(f"\nSaved: {config.PROCESSED_FILES['movies_master']}")
    print(f"Saved: {config.PROCESSED_FILES['ratings_clean']}")


if __name__ == "__main__":
    main()
