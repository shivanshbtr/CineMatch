"""
Integration tests for the CineMatch API (api/main.py + api/routers/recommend.py)

Uses FastAPI's TestClient, which triggers the real startup event (loading model
artifacts) without needing a running uvicorn server.

Run: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True


class TestTrendingEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/trending")
        assert resp.status_code == 200

    def test_response_schema(self, client):
        data = client.get("/trending").json()
        assert data["status"] == "success"
        assert data["strategy_used"] == "fallback_popularity"
        assert isinstance(data["results"], list)
        assert data["count"] == len(data["results"])

    def test_top_k_respected(self, client):
        data = client.get("/trending", params={"top_k": 5}).json()
        assert len(data["results"]) <= 5

    def test_top_k_clamped_below_min(self, client):
        # per docs/08_API_Specification.md §5: clamped, not rejected
        data = client.get("/trending", params={"top_k": 1}).json()
        assert len(data["results"]) >= 5  # clamped up to TOP_K_MIN

    def test_top_k_clamped_above_max(self, client):
        data = client.get("/trending", params={"top_k": 1000}).json()
        assert len(data["results"]) <= 20  # clamped down to TOP_K_MAX


class TestContentEndpoint:
    def test_valid_title_returns_200(self, client):
        resp = client.get("/recommend/content", params={"title": "Toy Story (1995)"})
        assert resp.status_code == 200

    def test_response_schema(self, client):
        data = client.get("/recommend/content", params={"title": "Toy Story (1995)"}).json()
        assert data["status"] == "success"
        assert data["strategy_used"] in ("content", "fallback_popularity")
        for r in data["results"]:
            assert "movie_id" in r and "title" in r and "score" in r and "reason" in r

    def test_case_insensitive_title_match(self, client):
        resp = client.get("/recommend/content", params={"title": "toy story (1995)"})
        assert resp.status_code == 200

    def test_unknown_title_returns_404(self, client):
        resp = client.get("/recommend/content", params={"title": "NotARealMovieTitle999"})
        assert resp.status_code == 404
        data = resp.json()
        assert data["status"] == "error"
        assert data["error_code"] == "MOVIE_NOT_FOUND"

    def test_missing_title_returns_422(self, client):
        # FastAPI/Pydantic validation error for a required query param
        resp = client.get("/recommend/content")
        assert resp.status_code == 422


class TestCollaborativeEndpoint:
    def test_known_user_returns_200(self, client):
        resp = client.get("/recommend/collaborative", params={"user_id": 1})
        assert resp.status_code == 200

    def test_unknown_user_returns_404(self, client):
        resp = client.get("/recommend/collaborative", params={"user_id": 999999999})
        assert resp.status_code == 404
        data = resp.json()
        assert data["error_code"] == "USER_NOT_FOUND"

    def test_response_never_errors_for_sparse_user(self, client):
        # Any user_id present in ratings.csv should get SOME response, even if
        # sparse (fallback), never a 500 — per docs/08_API_Specification.md §5
        resp = client.get("/recommend/collaborative", params={"user_id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy_used"] in ("collaborative", "fallback_content", "fallback_popularity")


class TestHybridEndpoint:
    def test_user_id_only(self, client):
        resp = client.get("/recommend/hybrid", params={"user_id": 1})
        assert resp.status_code == 200

    def test_reference_movie_only(self, client):
        resp = client.get("/recommend/hybrid", params={"reference_movie": "Toy Story (1995)"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy_used"] in ("fallback_content", "hybrid")

    def test_both_params(self, client):
        resp = client.get(
            "/recommend/hybrid",
            params={"user_id": 1, "reference_movie": "Toy Story (1995)"},
        )
        assert resp.status_code == 200

    def test_missing_both_params_returns_400(self, client):
        resp = client.get("/recommend/hybrid")
        assert resp.status_code == 400
        data = resp.json()
        assert data["error_code"] == "MISSING_INPUT"

    def test_new_user_gets_cold_start_fallback(self, client):
        # A user_id not in ratings.csv at all, with no reference movie either,
        # should still get a non-empty, non-error response (trending fallback)
        resp = client.get("/recommend/hybrid", params={"user_id": 888888888})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) > 0


class TestErrorEnvelope:
    def test_error_responses_follow_standard_envelope(self, client):
        resp = client.get("/recommend/content", params={"title": "NotARealMovieTitle999"})
        data = resp.json()
        assert set(["status", "error_code", "message"]).issubset(data.keys())
        assert data["status"] == "error"
