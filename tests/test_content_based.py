"""
Tests for src/content_based.py

Run: pytest tests/test_content_based.py -v
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.content_based import (
    load_movies_master,
    build_movie_index_map,
    get_similar_movies,
    find_movie_id_by_title,
    build_count_matrix,
)


@pytest.fixture(scope="module")
def movies_master():
    return load_movies_master()


@pytest.fixture(scope="module")
def artifacts(movies_master):
    count_matrix, _ = build_count_matrix(movies_master)
    movie_index_map = build_movie_index_map(movies_master)
    return count_matrix, movie_index_map


class TestMovieMasterLoad:
    def test_loads_without_error(self, movies_master):
        assert movies_master is not None

    def test_has_required_columns(self, movies_master):
        required = ["movie_id", "title", "genres", "metadata_soup", "has_content_features"]
        for col in required:
            assert col in movies_master.columns, f"Missing column: {col}"

    def test_no_null_metadata_soup(self, movies_master):
        assert movies_master["metadata_soup"].isna().sum() == 0

    def test_row_count(self, movies_master):
        assert len(movies_master) == 9742, "Expected 9742 movies from MovieLens ml-latest-small"

    def test_content_features_count(self, movies_master):
        n_with = movies_master["has_content_features"].sum()
        assert 3400 <= n_with <= 3700, f"Unexpected join coverage: {n_with}"


class TestCountMatrix:
    def test_matrix_shape(self, artifacts, movies_master):
        count_matrix, _ = artifacts
        assert count_matrix.shape[0] == len(movies_master)
        assert count_matrix.shape[1] > 1000

    def test_matrix_is_sparse(self, artifacts):
        count_matrix, _ = artifacts
        from scipy.sparse import issparse
        assert issparse(count_matrix)


class TestMovieIndexMap:
    def test_has_required_keys(self, artifacts):
        _, movie_index_map = artifacts
        assert "id_to_index" in movie_index_map
        assert "index_to_id" in movie_index_map
        assert "title_to_id" in movie_index_map

    def test_id_index_round_trip(self, artifacts):
        _, movie_index_map = artifacts
        for movie_id, idx in list(movie_index_map["id_to_index"].items())[:10]:
            assert movie_index_map["index_to_id"][idx] == movie_id


class TestFindMovieByTitle:
    def test_exact_match(self, artifacts):
        _, movie_index_map = artifacts
        mid = find_movie_id_by_title("Toy Story (1995)", movie_index_map)
        assert mid is not None
        assert isinstance(mid, (int, float))

    def test_case_insensitive(self, artifacts):
        _, movie_index_map = artifacts
        mid_lower = find_movie_id_by_title("toy story (1995)", movie_index_map)
        mid_upper = find_movie_id_by_title("TOY STORY (1995)", movie_index_map)
        assert mid_lower == mid_upper

    def test_unknown_title_returns_none(self, artifacts):
        _, movie_index_map = artifacts
        result = find_movie_id_by_title("ThisMovieDoesNotExist12345", movie_index_map)
        assert result is None


class TestGetSimilarMovies:
    def test_returns_top_k_results(self, artifacts, movies_master):
        count_matrix, movie_index_map = artifacts
        toy_story_id = find_movie_id_by_title("Toy Story (1995)", movie_index_map)
        results = get_similar_movies(toy_story_id, count_matrix, movie_index_map, movies_master, top_k=10)
        assert results is not None
        assert len(results) == 10

    def test_result_schema(self, artifacts, movies_master):
        count_matrix, movie_index_map = artifacts
        toy_story_id = find_movie_id_by_title("Toy Story (1995)", movie_index_map)
        results = get_similar_movies(toy_story_id, count_matrix, movie_index_map, movies_master, top_k=5)
        for r in results:
            assert "movie_id" in r
            assert "title" in r
            assert "genres" in r
            assert "score" in r
            assert "reason" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_input_movie_excluded_from_results(self, artifacts, movies_master):
        count_matrix, movie_index_map = artifacts
        toy_story_id = find_movie_id_by_title("Toy Story (1995)", movie_index_map)
        results = get_similar_movies(toy_story_id, count_matrix, movie_index_map, movies_master, top_k=10)
        result_ids = [r["movie_id"] for r in results]
        assert toy_story_id not in result_ids

    def test_scores_are_descending(self, artifacts, movies_master):
        count_matrix, movie_index_map = artifacts
        toy_story_id = find_movie_id_by_title("Toy Story (1995)", movie_index_map)
        results = get_similar_movies(toy_story_id, count_matrix, movie_index_map, movies_master, top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_movie_id_returns_none(self, artifacts, movies_master):
        count_matrix, movie_index_map = artifacts
        result = get_similar_movies(999999999, count_matrix, movie_index_map, movies_master, top_k=5)
        assert result is None
