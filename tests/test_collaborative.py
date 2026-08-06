"""
Tests for src/collaborative_filtering.py

Run: pytest tests/test_collaborative.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.collaborative_filtering import (
    train_test_split_ratings,
    build_index_maps,
    compute_baselines,
    build_residual_matrix,
    train_svd,
    predict_rating,
    evaluate_rmse,
    get_collaborative_recommendations,
    N_FACTORS,
)
from src.content_based import load_movies_master


@pytest.fixture(scope="module")
def ratings():
    return pd.read_csv(config.PROCESSED_FILES["ratings_clean"])


@pytest.fixture(scope="module")
def split(ratings):
    return train_test_split_ratings(ratings)


@pytest.fixture(scope="module")
def trained_model(split):
    train_df, test_df = split
    user_to_idx, movie_to_idx = build_index_maps(train_df)
    global_mean, user_bias, item_bias = compute_baselines(train_df, user_to_idx, movie_to_idx)
    residual_matrix = build_residual_matrix(
        train_df, user_to_idx, movie_to_idx, global_mean, user_bias, item_bias
    )
    U, sigma, Vt = train_svd(residual_matrix, n_factors=N_FACTORS)
    return {
        "U": U, "sigma": sigma, "Vt": Vt,
        "global_mean": global_mean, "user_bias": user_bias, "item_bias": item_bias,
        "user_to_idx": user_to_idx, "movie_to_idx": movie_to_idx,
    }


class TestTrainTestSplit:
    def test_split_produces_both_sets(self, split):
        train_df, test_df = split
        assert len(train_df) > 0
        assert len(test_df) > 0

    def test_no_overlap_between_train_and_test(self, split):
        train_df, test_df = split
        assert len(set(train_df.index) & set(test_df.index)) == 0

    def test_split_ratio_approximately_80_20(self, ratings, split):
        train_df, test_df = split
        total_split = len(train_df) + len(test_df)
        # Not all rows are split (users with <5 ratings kept entirely in train)
        assert total_split <= len(ratings)
        test_frac = len(test_df) / total_split
        assert 0.15 <= test_frac <= 0.25

    def test_reproducible_with_same_seed(self, ratings):
        train1, test1 = train_test_split_ratings(ratings, random_state=42)
        train2, test2 = train_test_split_ratings(ratings, random_state=42)
        assert set(test1.index) == set(test2.index)


class TestBaselines:
    def test_global_mean_within_valid_range(self, split):
        train_df, _ = split
        user_to_idx, movie_to_idx = build_index_maps(train_df)
        global_mean, _, _ = compute_baselines(train_df, user_to_idx, movie_to_idx)
        assert 0.5 <= global_mean <= 5.0

    def test_bias_arrays_correct_length(self, split):
        train_df, _ = split
        user_to_idx, movie_to_idx = build_index_maps(train_df)
        _, user_bias, item_bias = compute_baselines(train_df, user_to_idx, movie_to_idx)
        assert len(user_bias) == len(user_to_idx)
        assert len(item_bias) == len(movie_to_idx)


class TestSVDTraining:
    def test_factor_shapes_consistent(self, trained_model):
        n_users = len(trained_model["user_to_idx"])
        n_movies = len(trained_model["movie_to_idx"])
        assert trained_model["U"].shape[0] == n_users
        assert trained_model["Vt"].shape[1] == n_movies
        assert trained_model["U"].shape[1] == trained_model["Vt"].shape[0]

    def test_singular_values_positive_and_descending(self, trained_model):
        sigma = trained_model["sigma"]
        assert all(sigma >= 0)
        assert list(sigma) == sorted(sigma, reverse=True)


class TestPredictRating:
    def test_prediction_within_valid_range(self, trained_model):
        pred = predict_rating(
            0, 0, trained_model["U"], trained_model["sigma"], trained_model["Vt"],
            trained_model["global_mean"], trained_model["user_bias"], trained_model["item_bias"],
        )
        assert 0.5 <= pred <= 5.0

    def test_prediction_is_clipped_at_bounds(self, trained_model):
        # Artificially inflate a user/item factor to force an out-of-range raw prediction
        U = trained_model["U"].copy()
        U[0, :] = 100  # force an extreme prediction
        pred = predict_rating(
            0, 0, U, trained_model["sigma"], trained_model["Vt"],
            trained_model["global_mean"], trained_model["user_bias"], trained_model["item_bias"],
        )
        assert pred <= 5.0


class TestRMSEEvaluation:
    def test_rmse_is_reasonable(self, split, trained_model):
        _, test_df = split
        rmse, skipped, n_eval = evaluate_rmse(
            test_df, trained_model["user_to_idx"], trained_model["movie_to_idx"],
            trained_model["U"], trained_model["sigma"], trained_model["Vt"],
            trained_model["global_mean"], trained_model["user_bias"], trained_model["item_bias"],
        )
        assert rmse is not None
        # Published MovieLens SVD benchmarks are typically in the 0.85-0.95 range;
        # allow a wider margin here since this is a smoke test, not a regression check
        assert 0.5 <= rmse <= 1.5
        assert n_eval > 0


class TestCollaborativeRecommendations:
    def test_known_user_returns_results(self, ratings, trained_model):
        movies_master = load_movies_master()
        # user_id=1 exists in ml-latest-small and has substantial history
        results = get_collaborative_recommendations(1, trained_model, movies_master, ratings, top_k=10)
        assert results is not None
        assert len(results) == 10

    def test_results_exclude_already_rated_movies(self, ratings, trained_model):
        movies_master = load_movies_master()
        already_rated = set(ratings.loc[ratings["user_id"] == 1, "movie_id"])
        results = get_collaborative_recommendations(1, trained_model, movies_master, ratings, top_k=10)
        result_ids = {r["movie_id"] for r in results}
        assert result_ids.isdisjoint(already_rated)

    def test_unknown_user_returns_none(self, ratings, trained_model):
        movies_master = load_movies_master()
        results = get_collaborative_recommendations(999999999, trained_model, movies_master, ratings, top_k=10)
        assert results is None
