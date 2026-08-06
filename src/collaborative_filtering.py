"""
CineMatch — Collaborative Filtering Engine

Implements the algorithm documented in docs/07_Algorithms_and_Scoring_Logic.md §3:
1. Build the user-item ratings matrix
2. Compute baseline predictors (global mean + user bias + item bias)
3. Factorize the bias-adjusted residual matrix via truncated SVD (scipy.sparse.linalg.svds)
4. Reconstruct predicted ratings = baseline + latent factor contribution
5. Evaluate via RMSE on a held-out test split

Run: python src/collaborative_filtering.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

N_FACTORS = 30  # latent dimensionality
RANDOM_STATE = 42


def load_ratings():
    return pd.read_csv(config.PROCESSED_FILES["ratings_clean"])


def train_test_split_ratings(ratings, test_frac=0.2, random_state=RANDOM_STATE):
    """Per-user split so every user with enough ratings has some in both sets."""
    rng = np.random.default_rng(random_state)
    test_idx = []
    for user_id, group in ratings.groupby("user_id"):
        n = len(group)
        if n < 5:
            continue  # too few ratings to hold any out for this user
        n_test = max(1, int(n * test_frac))
        chosen = rng.choice(group.index.values, size=n_test, replace=False)
        test_idx.extend(chosen)
    test_idx = set(test_idx)
    test_df = ratings.loc[ratings.index.isin(test_idx)]
    train_df = ratings.loc[~ratings.index.isin(test_idx)]
    return train_df, test_df


def build_index_maps(ratings):
    user_ids = sorted(ratings["user_id"].unique())
    movie_ids = sorted(ratings["movie_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}
    return user_to_idx, movie_to_idx


def compute_baselines(train_df, user_to_idx, movie_to_idx):
    """Global mean + user bias + item bias, per docs/07_Algorithms_and_Scoring_Logic.md §3 step 2."""
    global_mean = train_df["rating"].mean()

    user_bias = train_df.groupby("user_id")["rating"].mean() - global_mean
    item_bias = train_df.groupby("movie_id")["rating"].mean() - global_mean

    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)
    user_bias_arr = np.zeros(n_users)
    item_bias_arr = np.zeros(n_movies)

    for uid, b in user_bias.items():
        user_bias_arr[user_to_idx[uid]] = b
    for mid, b in item_bias.items():
        item_bias_arr[movie_to_idx[mid]] = b

    return global_mean, user_bias_arr, item_bias_arr


def build_residual_matrix(train_df, user_to_idx, movie_to_idx, global_mean, user_bias, item_bias):
    rows = train_df["user_id"].map(user_to_idx).values
    cols = train_df["movie_id"].map(movie_to_idx).values
    baseline = global_mean + user_bias[rows] + item_bias[cols]
    residuals = train_df["rating"].values - baseline

    matrix = csr_matrix(
        (residuals, (rows, cols)), shape=(len(user_to_idx), len(movie_to_idx))
    )
    return matrix


def train_svd(residual_matrix, n_factors=N_FACTORS):
    k = min(n_factors, min(residual_matrix.shape) - 1)
    U, sigma, Vt = svds(residual_matrix.astype(float), k=k)
    # svds returns factors in ascending singular-value order; reverse for convention
    U, sigma, Vt = U[:, ::-1], sigma[::-1], Vt[::-1, :]
    return U, sigma, Vt


def predict_rating(user_idx, movie_idx, U, sigma, Vt, global_mean, user_bias, item_bias):
    baseline = global_mean + user_bias[user_idx] + item_bias[movie_idx]
    latent = U[user_idx, :] @ np.diag(sigma) @ Vt[:, movie_idx]
    pred = baseline + latent
    return float(np.clip(pred, 0.5, 5.0))


def evaluate_rmse(test_df, user_to_idx, movie_to_idx, U, sigma, Vt, global_mean, user_bias, item_bias):
    """RMSE on held-out ratings, per docs/07_Algorithms_and_Scoring_Logic.md §7."""
    errors = []
    skipped = 0
    for _, row in test_df.iterrows():
        uid, mid, actual = row["user_id"], row["movie_id"], row["rating"]
        if uid not in user_to_idx or mid not in movie_to_idx:
            skipped += 1
            continue  # unseen in training (cold-start) — not part of CF's evaluable scope
        pred = predict_rating(
            user_to_idx[uid], movie_to_idx[mid], U, sigma, Vt, global_mean, user_bias, item_bias
        )
        errors.append((pred - actual) ** 2)
    rmse = float(np.sqrt(np.mean(errors))) if errors else None
    return rmse, skipped, len(errors)


def precision_recall_at_k(test_df, user_to_idx, movie_to_idx, U, sigma, Vt,
                           global_mean, user_bias, item_bias, k=10, relevance_threshold=4.0):
    """
    Precision@K / Recall@K per docs/07_Algorithms_and_Scoring_Logic.md §7.
    "Relevant" = held-out rating >= relevance_threshold.
    """
    precisions, recalls = [], []
    for uid, group in test_df.groupby("user_id"):
        if uid not in user_to_idx:
            continue
        relevant = set(group[group["rating"] >= relevance_threshold]["movie_id"])
        if not relevant:
            continue

        candidate_movies = [m for m in group["movie_id"] if m in movie_to_idx]
        if not candidate_movies:
            continue

        preds = [
            (m, predict_rating(user_to_idx[uid], movie_to_idx[m], U, sigma, Vt,
                                global_mean, user_bias, item_bias))
            for m in candidate_movies
        ]
        preds.sort(key=lambda x: x[1], reverse=True)
        top_k_movies = {m for m, _ in preds[:k]}

        hits = len(top_k_movies & relevant)
        precisions.append(hits / min(k, len(top_k_movies)) if top_k_movies else 0)
        recalls.append(hits / len(relevant))

    return (float(np.mean(precisions)) if precisions else None,
            float(np.mean(recalls)) if recalls else None)


def get_collaborative_recommendations(user_id, svd, movies_master, ratings, top_k=10):
    """
    Predicts ratings for all of a known user's unseen movies and returns the Top-K.
    Returns None if the user is unknown to the trained model; the calling endpoint
    handles the fallback per docs/08_API_Specification.md §5.
    """
    if user_id not in svd["user_to_idx"]:
        return None

    user_idx = svd["user_to_idx"][user_id]
    already_rated = set(ratings.loc[ratings["user_id"] == user_id, "movie_id"])
    movie_lookup = movies_master.set_index("movie_id")

    predictions = []
    for movie_id, movie_idx in svd["movie_to_idx"].items():
        if movie_id in already_rated:
            continue
        pred = predict_rating(
            user_idx, movie_idx, svd["U"], svd["sigma"], svd["Vt"],
            svd["global_mean"], svd["user_bias"], svd["item_bias"],
        )
        predictions.append((movie_id, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top = predictions[:top_k]

    results = []
    for movie_id, score in top:
        if movie_id not in movie_lookup.index:
            continue
        row = movie_lookup.loc[movie_id]
        results.append(
            {
                "movie_id": int(movie_id),
                "title": row["title"],
                "genres": row["genres"].split("|") if isinstance(row["genres"], str) else [],
                "score": round(float(score) / 5.0, 4),
                "reason": "Users with similar taste also enjoyed this",
            }
        )
    return results


def main():
    print("Loading cleaned ratings...")
    ratings = load_ratings()

    print("Splitting train/test (per-user, 80/20)...")
    train_df, test_df = train_test_split_ratings(ratings)
    print(f"  Train: {len(train_df)} ratings | Test: {len(test_df)} ratings")

    print("Building index maps...")
    user_to_idx, movie_to_idx = build_index_maps(train_df)
    print(f"  {len(user_to_idx)} users, {len(movie_to_idx)} movies in training set")

    print("Computing baseline predictors (global mean + user/item bias)...")
    global_mean, user_bias, item_bias = compute_baselines(train_df, user_to_idx, movie_to_idx)
    print(f"  Global mean rating: {global_mean:.3f}")

    print("Building residual matrix...")
    residual_matrix = build_residual_matrix(
        train_df, user_to_idx, movie_to_idx, global_mean, user_bias, item_bias
    )
    print(f"  Matrix shape: {residual_matrix.shape}, nnz={residual_matrix.nnz}")

    print(f"Training truncated SVD (k={N_FACTORS})...")
    U, sigma, Vt = train_svd(residual_matrix, n_factors=N_FACTORS)
    print(f"  U: {U.shape}, sigma: {sigma.shape}, Vt: {Vt.shape}")

    print("Evaluating on held-out test set...")
    rmse, skipped, n_eval = evaluate_rmse(
        test_df, user_to_idx, movie_to_idx, U, sigma, Vt, global_mean, user_bias, item_bias
    )
    print(f"  RMSE: {rmse:.4f}  (evaluated on {n_eval} ratings, skipped {skipped} unseen-in-train)")

    precision, recall = precision_recall_at_k(
        test_df, user_to_idx, movie_to_idx, U, sigma, Vt, global_mean, user_bias, item_bias, k=10
    )
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")

    print("Saving SVD model artifact...")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "U": U,
        "sigma": sigma,
        "Vt": Vt,
        "global_mean": global_mean,
        "user_bias": user_bias,
        "item_bias": item_bias,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "n_factors": N_FACTORS,
        "eval_metrics": {"rmse": rmse, "precision_at_10": precision, "recall_at_10": recall},
    }
    with open(config.MODEL_ARTIFACTS["svd_model"], "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"Saved: {config.MODEL_ARTIFACTS['svd_model']}")


if __name__ == "__main__":
    main()
