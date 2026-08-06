"""
CineMatch API — Pydantic Schemas

Mirrors the response envelope and error format documented in
docs/08_API_Specification.md §3 (Common Response Envelope).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class MovieResult(BaseModel):
    movie_id: int
    title: str
    genres: List[str]
    score: float
    reason: str


class RecommendationResponse(BaseModel):
    status: str = "success"
    strategy_used: str
    count: int
    results: List[MovieResult]


class ErrorResponse(BaseModel):
    status: str = "error"
    error_code: str
    message: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
