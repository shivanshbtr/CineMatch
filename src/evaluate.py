"""
CineMatch — Consolidated Evaluation

Implements docs/07_Algorithms_and_Scoring_Logic.md §7 (Evaluation Metrics table):
- RMSE                  -> Collaborative Filtering
- Precision@K/Recall@K  -> Content-only, Collaborative-only, and Hybrid (comparison table)
- Coverage              -> Content-Based

All ranking metrics use the SAME candidate-restricted methodology as
src/collaborative_filtering.py's precision_recall_at_k, so the three strategies
are compared on equal footing: for each test user, rank only the movies that
appear in that user's held-out test set (a standard simplified leave-out
evaluation), and check how many of the top-K by predicted score were actually
rated >= relevance_threshold by the user.

Run: python src/evaluate.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.collaborative_filtering import (
    build_index_maps,
    compute_baselines,
    build_residual_matrix,
    train_svd,
    predict_rating,
    train_test_split_ratings,
    evaluate_rmse,
    precision_recall_at_k as cf_precision_recall_at_k,
    N_FACTORS,
)
from src.content_based import load_movies_master


RELEVANCE_THRESHOLD = 4.0
TOP_K = 10


def load_artifacts():
    with open(config.MODEL_ARTIFACTS["count_matrix"], "rb") as f:
        count_matrix = pickle.load(f)
    with open(config.MODEL_ARTIFACTS["movie_index_map"], "rb") as f:
        movie_index_map = pickle.load(f)
    movies_master = load_movies_master()
    ratings = pd.read_csv(config.PROCESSED_FILES["ratings_clean"])
    return count_matrix, movie_index_map, movies_master, ratings


# ---------------------------------------------------------------------------
# Content-only and Hybrid precision/recall, using the same candidate-restricted
# methodology as collaborative_filtering.precision_recall_at_k for a fair
# apples-to-apples comparison across strategies.
# ---------------------------------------------------------------------------

def content_and_hybrid_precision_recall_at_k(
    train_df, test_df, count_matrix, movie_index_map,
    svd, k=TOP_K, relevance_threshold=RELEVANCE_THRESHOLD, alpha_override=None,
):
    """
    alpha_override=None -> use the documented alpha table (i.e. true hybrid behavior)
    alpha_override=0.0  -> pure content-based (for the comparison table)
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from src.hybrid_model import get_alpha

    id_to_index = movie_index_map["id_to_index"]
    precisions, recalls = [], []

    # Seed movie per user: their highest-rated movie in the TRAINING set.
    # Sorted by (rating desc, movie_id asc) with a stable sort so tie-breaking
    # among a user's multiple max-rated movies is deterministic across machines
    # and pandas/numpy versions — quicksort's tie order is not guaranteed stable.
    top_train_movie = (
        train_df.sort_values(
            ["rating", "movie_id"], ascending=[False, True], kind="mergesort"
        )
        .drop_duplicates(subset="user_id", keep="first")
        .set_index("user_id")["movie_id"]
    )

    for uid, group in test_df.groupby("user_id"):
        if uid not in top_train_movie.index:
            continue
        seed_movie_id = top_train_movie.loc[uid]
        if seed_movie_id not in id_to_index:
            continue
        seed_idx = id_to_index[seed_movie_id]

        relevant = set(group[group["rating"] >= relevance_threshold]["movie_id"])
        if not relevant:
            continue

        candidates = [m for m in group["movie_id"] if m in id_to_index]
        if not candidates:
            continue

        rating_count = int((train_df["user_id"] == uid).sum())
        alpha = alpha_override if alpha_override is not None else get_alpha(rating_count)

        # Compute similarity between the seed movie and ALL movies once (one sparse row
        # multiply), then index into it per candidate — avoids recomputing per pair.
        sim_row = cosine_similarity(count_matrix[seed_idx], count_matrix).flatten()
        raw_content = np.array([sim_row[id_to_index[m]] for m in candidates])
        c_min, c_max = raw_content.min(), raw_content.max()
        content_norm = (
            (raw_content - c_min) / (c_max - c_min) if c_max > c_min else np.ones_like(raw_content)
        )

        scored = []
        for m, c_norm in zip(candidates, content_norm):
            if alpha > 0 and uid in svd["user_to_idx"] and m in svd["movie_to_idx"]:
                pred = predict_rating(
                    svd["user_to_idx"][uid], svd["movie_to_idx"][m],
                    svd["U"], svd["sigma"], svd["Vt"],
                    svd["global_mean"], svd["user_bias"], svd["item_bias"],
                )
                cf_norm = pred / 5.0
            else:
                cf_norm = svd["global_mean"] / 5.0  # neutral prior, per hybrid_model.py
            final_score = alpha * cf_norm + (1 - alpha) * c_norm
            scored.append((m, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_k_movies = {m for m, _ in scored[:k]}

        hits = len(top_k_movies & relevant)
        precisions.append(hits / min(k, len(top_k_movies)) if top_k_movies else 0)
        recalls.append(hits / len(relevant))

    return (
        float(np.mean(precisions)) if precisions else None,
        float(np.mean(recalls)) if recalls else None,
        len(precisions),
    )


def compute_content_coverage(count_matrix, movie_index_map, movies_master, top_k=TOP_K, sample_size=1000):
    """
    Coverage per docs/07_Algorithms_and_Scoring_Logic.md §7:
    what fraction of the catalog is EVER recommended across many requests.
    Sampled (not exhaustive) for speed on the full 9,742-movie catalog.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    index_to_id = movie_index_map["index_to_id"]
    has_features = movies_master.set_index("movie_id")["has_content_features"]

    candidate_ids = [mid for mid in movie_index_map["id_to_index"] if has_features.get(mid, False)]
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(candidate_ids, size=min(sample_size, len(candidate_ids)), replace=False)

    recommended_ever = set()
    for mid in sample_ids:
        idx = movie_index_map["id_to_index"][mid]
        sim_row = cosine_similarity(count_matrix[idx], count_matrix).flatten()
        sim_scores = list(enumerate(sim_row))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_k]
        recommended_ever.update(index_to_id[i] for i, _ in sim_scores)

    total_catalog = len(candidate_ids)
    coverage = len(recommended_ever) / total_catalog
    return coverage, len(recommended_ever), total_catalog, len(sample_ids)


def main():
    print("Loading artifacts...")
    count_matrix, movie_index_map, movies_master, ratings = load_artifacts()

    print("Reproducing train/test split (same random_state as collaborative_filtering.py)...")
    train_df, test_df = train_test_split_ratings(ratings)

    print("Rebuilding SVD model on this split for evaluation consistency...")
    user_to_idx, movie_to_idx = build_index_maps(train_df)
    global_mean, user_bias, item_bias = compute_baselines(train_df, user_to_idx, movie_to_idx)
    residual_matrix = build_residual_matrix(train_df, user_to_idx, movie_to_idx, global_mean, user_bias, item_bias)
    U, sigma, Vt = train_svd(residual_matrix, n_factors=N_FACTORS)
    svd = {
        "U": U, "sigma": sigma, "Vt": Vt,
        "global_mean": global_mean, "user_bias": user_bias, "item_bias": item_bias,
        "user_to_idx": user_to_idx, "movie_to_idx": movie_to_idx,
    }

    report = {}

    # --- Collaborative Filtering: RMSE, Precision@K, Recall@K ---
    print("\nEvaluating Collaborative Filtering (RMSE)...")
    rmse, skipped, n_eval = evaluate_rmse(test_df, user_to_idx, movie_to_idx, U, sigma, Vt, global_mean, user_bias, item_bias)
    cf_precision, cf_recall = cf_precision_recall_at_k(
        test_df, user_to_idx, movie_to_idx, U, sigma, Vt, global_mean, user_bias, item_bias, k=TOP_K
    )
    report["collaborative_filtering"] = {
        "rmse": round(rmse, 4), "precision_at_10": round(cf_precision, 4), "recall_at_10": round(cf_recall, 4),
    }
    print(f"  RMSE={rmse:.4f}  Precision@10={cf_precision:.4f}  Recall@10={cf_recall:.4f}")

    # --- Content-Based only: Precision@K, Recall@K, Coverage ---
    print("\nEvaluating Content-Based (alpha forced to 0)...")
    content_precision, content_recall, n_users_content = content_and_hybrid_precision_recall_at_k(
        train_df, test_df, count_matrix, movie_index_map, svd, k=TOP_K, alpha_override=0.0
    )
    print(f"  Precision@10={content_precision:.4f}  Recall@10={content_recall:.4f}  (n_users={n_users_content})")

    print("Computing content-based coverage (sampled)...")
    coverage, n_recommended, total_catalog, n_sampled = compute_content_coverage(
        count_matrix, movie_index_map, movies_master
    )
    print(f"  Coverage={coverage:.4f}  ({n_recommended}/{total_catalog} movies ever recommended, sampled {n_sampled} queries)")

    report["content_based"] = {
        "precision_at_10": round(content_precision, 4), "recall_at_10": round(content_recall, 4),
        "coverage": round(coverage, 4),
    }

    # --- Hybrid: Precision@K, Recall@K, using the real documented alpha table ---
    print("\nEvaluating Hybrid (alpha per documented rating-count table)...")
    hybrid_precision, hybrid_recall, n_users_hybrid = content_and_hybrid_precision_recall_at_k(
        train_df, test_df, count_matrix, movie_index_map, svd, k=TOP_K, alpha_override=None
    )
    print(f"  Precision@10={hybrid_precision:.4f}  Recall@10={hybrid_recall:.4f}  (n_users={n_users_hybrid})")
    report["hybrid"] = {"precision_at_10": round(hybrid_precision, 4), "recall_at_10": round(hybrid_recall, 4)}

    # --- Comparison table ---
    print("\n=== Model Comparison (Precision@10 / Recall@10) ===")
    print(f"{'Strategy':<25s}{'Precision@10':>15s}{'Recall@10':>15s}")
    print(f"{'Content-based only':<25s}{content_precision:>15.4f}{content_recall:>15.4f}")
    print(f"{'Collaborative only':<25s}{cf_precision:>15.4f}{cf_recall:>15.4f}")
    print(f"{'Hybrid':<25s}{hybrid_precision:>15.4f}{hybrid_recall:>15.4f}")

    out_path = config.MODELS_DIR / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
