"""
CineMatch — Hybrid Scoring Module

Implements docs/07_Algorithms_and_Scoring_Logic.md §4-5:
- Normalizes content-based and collaborative scores to [0, 1]
- Blends them using an alpha weight determined by user rating count
- Applies the cold-start decision tree so a recommendation is always returned

Depends on the precomputed artifacts from content_based.py and collaborative_filtering.py.

Run: python src/hybrid_model.py   (runs a demo against a few sample users)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.content_based import get_similar_movies, load_movies_master
from src.collaborative_filtering import predict_rating


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_all_artifacts():
    with open(config.MODEL_ARTIFACTS["count_matrix"], "rb") as f:
        count_matrix = pickle.load(f)
    with open(config.MODEL_ARTIFACTS["movie_index_map"], "rb") as f:
        movie_index_map = pickle.load(f)
    with open(config.MODEL_ARTIFACTS["svd_model"], "rb") as f:
        svd = pickle.load(f)
    movies_master = load_movies_master()
    ratings = pd.read_csv(config.PROCESSED_FILES["ratings_clean"])
    return count_matrix, movie_index_map, svd, movies_master, ratings


# ---------------------------------------------------------------------------
# Alpha lookup, per docs/07_Algorithms_and_Scoring_Logic.md §4 step 2
# and config.ALPHA_TABLE (docs/09_Development_Guide.md §6 centralization)
# ---------------------------------------------------------------------------

def get_alpha(rating_count: int) -> float:
    for (lo, hi), alpha in config.ALPHA_TABLE:
        if lo <= rating_count <= hi:
            return alpha
    return 0.0  # fallback, should be unreachable given an inf-bounded table


# ---------------------------------------------------------------------------
# Popularity fallback (IMDB weighted rating), per §6 of the Algorithms doc
# ---------------------------------------------------------------------------

def compute_trending(movies_master: pd.DataFrame, top_k=10, min_votes_percentile=0.5):
    df = movies_master.copy()

    # Compute the vote-count threshold (m) and global mean (C) from movies that actually
    # HAVE TMDB vote data — filling NaN with 0 first would corrupt the quantile, since ~64%
    # of movies (collaborative-only, no TMDB match) have no vote data at all, not zero votes.
    has_votes = df["vote_count"].notna() & (df["vote_count"] > 0)
    m = df.loc[has_votes, "vote_count"].quantile(min_votes_percentile)
    C = df.loc[has_votes, "vote_average"].mean()

    eligible = df[has_votes & (df["vote_count"] >= m)].copy()
    eligible["weighted_rating"] = (
        (eligible["vote_count"] / (eligible["vote_count"] + m)) * eligible["vote_average"]
        + (m / (eligible["vote_count"] + m)) * C
    )
    top = eligible.sort_values("weighted_rating", ascending=False).head(top_k)

    return [
        {
            "movie_id": int(row["movie_id"]),
            "title": row["title"],
            "genres": row["genres"].split("|") if isinstance(row["genres"], str) else [],
            "score": round(float(row["weighted_rating"]) / 10, 4),
            "reason": "Popular and highly rated",
        }
        for _, row in top.iterrows()
    ]


# ---------------------------------------------------------------------------
# Hybrid recommendation — implements the decision tree in
# docs/07_Algorithms_and_Scoring_Logic.md §5
# ---------------------------------------------------------------------------

def get_hybrid_recommendations(
    user_id,
    reference_movie_id,
    count_matrix,
    movie_index_map,
    svd,
    movies_master,
    ratings,
    top_k=10,
):
    user_rating_count = 0
    if user_id is not None:
        user_rating_count = int((ratings["user_id"] == user_id).sum())

    known_user = user_id is not None and user_id in svd["user_to_idx"]

    # --- Cold-start branch: no user history, no reference movie ---
    if user_rating_count == 0 and reference_movie_id is None:
        return compute_trending(movies_master, top_k=top_k), "fallback_popularity"

    # --- Cold-start branch: no user history, but a reference movie given ---
    if user_rating_count == 0 and reference_movie_id is not None:
        results = get_similar_movies(
            reference_movie_id, count_matrix, movie_index_map, movies_master, top_k=top_k
        )
        if results is None:
            return compute_trending(movies_master, top_k=top_k), "fallback_popularity"
        return results, "fallback_content"

    # --- User has history: run hybrid scoring ---
    alpha = get_alpha(user_rating_count)

    # Content-side candidate pool: similar movies to the reference movie if given,
    # otherwise similar movies to the user's own highest-rated movie.
    if reference_movie_id is not None:
        seed_movie_id = reference_movie_id
    else:
        # Sorted by (rating desc, movie_id asc) with a stable sort for deterministic
        # tie-breaking — see the same fix in src/evaluate.py for the full rationale.
        user_ratings = ratings[ratings["user_id"] == user_id].sort_values(
            ["rating", "movie_id"], ascending=[False, True], kind="mergesort"
        )
        seed_movie_id = int(user_ratings.iloc[0]["movie_id"]) if len(user_ratings) else None

    content_candidates = (
        get_similar_movies(
            seed_movie_id, count_matrix, movie_index_map, movies_master, top_k=top_k * 3
        )
        if seed_movie_id is not None
        else []
    ) or []

    already_rated = set(ratings.loc[ratings["user_id"] == user_id, "movie_id"]) if known_user else set()
    candidates = [c for c in content_candidates if c["movie_id"] not in already_rated]

    if not candidates:
        # No content candidates available (e.g. seed movie has no metadata) — fall back
        return compute_trending(movies_master, top_k=top_k), "fallback_popularity"

    content_scores = np.array([c["score"] for c in candidates])
    content_min, content_max = content_scores.min(), content_scores.max()
    content_norm = (
        (content_scores - content_min) / (content_max - content_min)
        if content_max > content_min
        else np.ones_like(content_scores)
    )

    scored = []
    for c, c_norm in zip(candidates, content_norm):
        if known_user and c["movie_id"] in svd["movie_to_idx"]:
            pred = predict_rating(
                svd["user_to_idx"][user_id],
                svd["movie_to_idx"][c["movie_id"]],
                svd["U"], svd["sigma"], svd["Vt"],
                svd["global_mean"], svd["user_bias"], svd["item_bias"],
            )
            cf_norm = pred / 5.0
        else:
            # No CF signal for this movie (too few ratings to appear in the training index).
            # Fall back to the global mean rating as a neutral prior — NOT the content score,
            # which would double-count content similarity into both terms and let obscure,
            # near-unrated movies outrank movies with genuine strong CF predictions.
            cf_norm = svd["global_mean"] / 5.0

        final_score = alpha * cf_norm + (1 - alpha) * c_norm
        reason = (
            "Recommended based on similar users' ratings"
            if alpha >= 0.5
            else "Recommended based on movies you've liked"
        )
        scored.append({**c, "score": round(float(final_score), 4), "reason": reason})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k], "hybrid"


def main():
    print("Loading artifacts (content, collaborative, movies, ratings)...")
    count_matrix, movie_index_map, svd, movies_master, ratings = load_all_artifacts()

    print("\n=== Demo 1: Cold-start user, no reference movie -> Trending ===")
    results, strategy = get_hybrid_recommendations(
        user_id=None, reference_movie_id=None,
        count_matrix=count_matrix, movie_index_map=movie_index_map,
        svd=svd, movies_master=movies_master, ratings=ratings, top_k=5,
    )
    print(f"strategy_used: {strategy}")
    for r in results:
        print(f"  {r['title']:45s} score={r['score']}  ({r['reason']})")

    print("\n=== Demo 2: Cold-start user WITH reference movie 'Toy Story (1995)' ===")
    toy_story_id = movie_index_map["title_to_id"].get("Toy Story (1995)")
    results, strategy = get_hybrid_recommendations(
        user_id=None, reference_movie_id=toy_story_id,
        count_matrix=count_matrix, movie_index_map=movie_index_map,
        svd=svd, movies_master=movies_master, ratings=ratings, top_k=5,
    )
    print(f"strategy_used: {strategy}")
    for r in results:
        print(f"  {r['title']:45s} score={r['score']}  ({r['reason']})")

    print("\n=== Demo 3: Established user (user_id=1) -> Hybrid ===")
    user_1_count = int((ratings["user_id"] == 1).sum())
    print(f"user_id=1 has {user_1_count} ratings -> alpha={get_alpha(user_1_count)}")
    results, strategy = get_hybrid_recommendations(
        user_id=1, reference_movie_id=None,
        count_matrix=count_matrix, movie_index_map=movie_index_map,
        svd=svd, movies_master=movies_master, ratings=ratings, top_k=5,
    )
    print(f"strategy_used: {strategy}")
    for r in results:
        print(f"  {r['title']:45s} score={r['score']}  ({r['reason']})")


if __name__ == "__main__":
    main()
