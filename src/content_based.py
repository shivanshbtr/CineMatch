"""
CineMatch — Content-Based Filtering Engine

Implements the algorithm documented in docs/07_Algorithms_and_Scoring_Logic.md §2:
1. Vectorize each movie's "metadata soup" (genres + keywords + top cast + director)
2. Compute pairwise cosine similarity across all movies
3. Given a movie, return the Top-K most similar movies

Precomputes and caches artifacts per docs/06_System_Architecture.md §7
(never compute the full similarity matrix inside a request handler).

Run: python src/content_based.py
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


def load_movies_master():
    df = pd.read_csv(config.PROCESSED_FILES["movies_master"])
    # CSV round-trips empty strings as NaN — restore them per docs/02_Feature_Engineering.md §7
    df["metadata_soup"] = df["metadata_soup"].fillna("")
    df["overview"] = df["overview"].fillna("")
    return df


def build_count_matrix(movies_master: pd.DataFrame):
    """
    Builds the sparse term-count matrix from the metadata soup.

    Uses CountVectorizer (not TF-IDF) on the soup per docs/07_Algorithms_and_Scoring_Logic.md §2 —
    the soup is a bag of discrete tags (genres/keywords/names), not free-flowing prose, so raw
    term counts work better here than TF-IDF's document-frequency down-weighting, which would
    unfairly penalize common-but-meaningful tags shared across many movies (e.g. "Drama").

    NOTE: The full n×n cosine similarity matrix is not precomputed or stored.
    For ~9,700 movies that matrix is ~725MB (n^2 float64 storage), which is impractical
    to persist or load. The sparse count matrix (a few MB) is cached instead, and
    per-query similarity is computed on demand — a single sparse row against the full
    matrix, sub-millisecond and well within the 2s latency budget in NFR-1
    (docs/04_Software_Requirements_Specification.md). See docs/07_Algorithms_and_Scoring_Logic.md §2.
    """
    vectorizer = CountVectorizer(stop_words="english")
    count_matrix = vectorizer.fit_transform(movies_master["metadata_soup"])
    return count_matrix, vectorizer


def build_movie_index_map(movies_master: pd.DataFrame):
    """Maps movie_id <-> row position in the similarity matrix, and title -> movie_id."""
    id_to_index = pd.Series(movies_master.index, index=movies_master["movie_id"]).to_dict()
    index_to_id = {v: k for k, v in id_to_index.items()}
    title_to_id = pd.Series(
        movies_master["movie_id"].values, index=movies_master["title"]
    ).to_dict()
    return {"id_to_index": id_to_index, "index_to_id": index_to_id, "title_to_id": title_to_id}


def get_similar_movies(movie_id, count_matrix, movie_index_map, movies_master, top_k=10):
    """
    Returns Top-K similar movies for a given movie_id.
    Computes cosine similarity for this one movie on demand against the cached sparse
    count matrix (see build_count_matrix docstring for why we don't precompute the full
    dense matrix). Mirrors the response contract of GET /recommend/content in
    docs/08_API_Specification.md §4.1.
    """
    id_to_index = movie_index_map["id_to_index"]
    index_to_id = movie_index_map["index_to_id"]

    if movie_id not in id_to_index:
        return None  # returns None if movie_id not found; endpoint maps this to 404

    idx = id_to_index[movie_id]
    sim_row = cosine_similarity(count_matrix[idx], count_matrix).flatten()
    sim_scores = list(enumerate(sim_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_k]  # exclude the movie itself

    results = []
    for i, score in sim_scores:
        rid = index_to_id[i]
        row = movies_master.loc[movies_master["movie_id"] == rid].iloc[0]
        results.append(
            {
                "movie_id": int(rid),
                "title": row["title"],
                "genres": row["genres"].split("|") if isinstance(row["genres"], str) else [],
                "score": round(float(score), 4),
                "reason": f"Shares genre/theme with the selected movie",
            }
        )
    return results


def find_movie_id_by_title(title, movie_index_map):
    """
    Case-insensitive exact title match, per docs/08_API_Specification.md §6
    ("title must match an existing movie (case-insensitive)").
    Returns None if not found.
    """
    title_to_id = movie_index_map["title_to_id"]
    target = title.strip().lower()
    for known_title, movie_id in title_to_id.items():
        if known_title.strip().lower() == target:
            return movie_id
    return None


def main():
    print("Loading processed movies master table...")
    movies_master = load_movies_master()

    n_empty_soup = (movies_master["metadata_soup"].str.strip() == "").sum()
    print(f"Movies with empty metadata soup (no content features): {n_empty_soup}")

    print("Building CountVectorizer + sparse term-count matrix...")
    count_matrix, vectorizer = build_count_matrix(movies_master)
    print(f"Count matrix shape: {count_matrix.shape} (sparse, nnz={count_matrix.nnz})")

    print("Building movie index map...")
    movie_index_map = build_movie_index_map(movies_master)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.MODEL_ARTIFACTS["count_matrix"], "wb") as f:
        pickle.dump(count_matrix, f)
    with open(config.MODEL_ARTIFACTS["movie_index_map"], "wb") as f:
        pickle.dump(movie_index_map, f)
    with open(config.MODEL_ARTIFACTS["tfidf_matrix"], "wb") as f:
        # Artifact stored as tfidf_matrix.pkl to match the naming convention in
        # docs/02_Feature_Engineering.md §8. CountVectorizer is used in place of
        # TF-IDF (see build_count_matrix docstring for rationale).
        pickle.dump(vectorizer, f)

    print(f"Saved: {config.MODEL_ARTIFACTS['count_matrix']}")
    print(f"Saved: {config.MODEL_ARTIFACTS['movie_index_map']}")
    print(f"Saved: {config.MODEL_ARTIFACTS['tfidf_matrix']} (vectorizer)")

    # --- Sanity check demo ---
    print("\n=== Sanity Check: Similar movies to 'Toy Story (1995)' ===")
    toy_story_id = movie_index_map["title_to_id"].get("Toy Story (1995)")
    if toy_story_id:
        results = get_similar_movies(
            toy_story_id, count_matrix, movie_index_map, movies_master, top_k=5
        )
        for r in results:
            print(f"  {r['title']:45s} score={r['score']}")


if __name__ == "__main__":
    main()
