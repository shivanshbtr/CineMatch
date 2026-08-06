"""
CineMatch API — Recommendation Router

Implements docs/08_API_Specification.md §4 endpoints exactly:
- GET /recommend/content
- GET /recommend/collaborative
- GET /recommend/hybrid
- GET /trending

Validation rules per §6 of the same doc are enforced here (title match,
user existence, top_k clamping, missing-input check on /hybrid).
"""

import sys
from pathlib import Path

from fastapi import APIRouter, Request, Query
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.content_based import get_similar_movies, find_movie_id_by_title
from src.collaborative_filtering import get_collaborative_recommendations
from src.hybrid_model import get_hybrid_recommendations, compute_trending

router = APIRouter()


def _clamp_top_k(top_k: int) -> int:
    """Server-side clamp, never rejected — per docs/08_API_Specification.md §5."""
    return max(config.TOP_K_MIN, min(config.TOP_K_MAX, top_k))


class AppExceptionLocal(Exception):
    pass


def _app_exception(status_code, error_code, message):
    from api.main import AppException
    raise AppException(status_code, error_code, message)


@router.get("/recommend/content")
def recommend_content(
    request: Request,
    title: str = Query(..., description="Movie title (case-insensitive match)"),
    top_k: int = Query(config.TOP_K_DEFAULT),
):
    top_k = _clamp_top_k(top_k)
    movie_index_map = request.app.state.movie_index_map
    count_matrix = request.app.state.count_matrix
    movies_master = request.app.state.movies_master

    movie_id = find_movie_id_by_title(title, movie_index_map)
    if movie_id is None:
        _app_exception(404, "MOVIE_NOT_FOUND", "No movie found matching the given title.")

    results = get_similar_movies(movie_id, count_matrix, movie_index_map, movies_master, top_k=top_k)
    if not results:
        # Movie exists but has no content features (e.g. no TMDB match) — degrade to trending
        # rather than return an empty result, per NFR-5 (Reliability).
        results = compute_trending(movies_master, top_k=top_k)
        return {"status": "success", "strategy_used": "fallback_popularity", "count": len(results), "results": results}

    return {"status": "success", "strategy_used": "content", "count": len(results), "results": results}


@router.get("/recommend/collaborative")
def recommend_collaborative(
    request: Request,
    user_id: int = Query(...),
    top_k: int = Query(config.TOP_K_DEFAULT),
):
    top_k = _clamp_top_k(top_k)
    ratings = request.app.state.ratings
    svd = request.app.state.svd
    movies_master = request.app.state.movies_master
    movie_index_map = request.app.state.movie_index_map
    count_matrix = request.app.state.count_matrix

    user_exists_in_data = (ratings["user_id"] == user_id).any()
    if not user_exists_in_data:
        _app_exception(404, "USER_NOT_FOUND", "No user found with the given user_id.")

    results = get_collaborative_recommendations(user_id, svd, movies_master, ratings, top_k=top_k)

    if results is None or len(results) == 0:
        # Insufficient history to be in the trained CF index — degrade gracefully,
        # per docs/08_API_Specification.md §5 (never a hard error for a sparse user).
        user_ratings = ratings.loc[ratings["user_id"] == user_id, "movie_id"]
        if len(user_ratings) > 0:
            # Sorted by (rating desc, movie_id asc) with a stable sort for deterministic
            # tie-breaking — see src/evaluate.py for the full rationale.
            seed_movie_id = int(
                ratings.loc[ratings["user_id"] == user_id]
                .sort_values(["rating", "movie_id"], ascending=[False, True], kind="mergesort")
                .iloc[0]["movie_id"]
            )
            results = get_similar_movies(seed_movie_id, count_matrix, movie_index_map, movies_master, top_k=top_k)
            strategy = "fallback_content"
        else:
            results = None
            strategy = None

        if not results:
            results = compute_trending(movies_master, top_k=top_k)
            strategy = "fallback_popularity"

        return {"status": "success", "strategy_used": strategy, "count": len(results), "results": results}

    return {"status": "success", "strategy_used": "collaborative", "count": len(results), "results": results}


@router.get("/recommend/hybrid")
def recommend_hybrid(
    request: Request,
    user_id: Optional[int] = Query(None),
    reference_movie: Optional[str] = Query(None),
    top_k: int = Query(config.TOP_K_DEFAULT),
):
    top_k = _clamp_top_k(top_k)

    if user_id is None and reference_movie is None:
        _app_exception(400, "MISSING_INPUT", "At least one of user_id or reference_movie must be provided.")

    movie_index_map = request.app.state.movie_index_map
    count_matrix = request.app.state.count_matrix
    svd = request.app.state.svd
    movies_master = request.app.state.movies_master
    ratings = request.app.state.ratings

    reference_movie_id = None
    if reference_movie is not None:
        reference_movie_id = find_movie_id_by_title(reference_movie, movie_index_map)
        if reference_movie_id is None:
            _app_exception(404, "MOVIE_NOT_FOUND", "No movie found matching the given reference_movie title.")

    results, strategy = get_hybrid_recommendations(
        user_id=user_id,
        reference_movie_id=reference_movie_id,
        count_matrix=count_matrix,
        movie_index_map=movie_index_map,
        svd=svd,
        movies_master=movies_master,
        ratings=ratings,
        top_k=top_k,
    )

    return {"status": "success", "strategy_used": strategy, "count": len(results), "results": results}


@router.get("/trending")
def trending(request: Request, top_k: int = Query(config.TOP_K_DEFAULT)):
    top_k = _clamp_top_k(top_k)
    movies_master = request.app.state.movies_master
    results = compute_trending(movies_master, top_k=top_k)
    return {"status": "success", "strategy_used": "fallback_popularity", "count": len(results), "results": results}
