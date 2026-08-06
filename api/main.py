"""
CineMatch API — FastAPI Application Entrypoint

Loads all precomputed model artifacts ONCE at startup (not per-request), per
docs/06_System_Architecture.md §7 and docs/09_Development_Guide.md §4.

Run: uvicorn api.main:app --reload --port 8000
"""

import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.content_based import load_movies_master
from api.routers import recommend


class AppException(Exception):
    """Base for all CineMatch API errors, mapped to the standard error envelope."""

    def __init__(self, status_code: int, error_code: str, message: str):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model artifacts...")
    with open(config.MODEL_ARTIFACTS["count_matrix"], "rb") as f:
        app.state.count_matrix = pickle.load(f)
    with open(config.MODEL_ARTIFACTS["movie_index_map"], "rb") as f:
        app.state.movie_index_map = pickle.load(f)
    with open(config.MODEL_ARTIFACTS["svd_model"], "rb") as f:
        app.state.svd = pickle.load(f)
    app.state.movies_master = load_movies_master()
    app.state.ratings = pd.read_csv(config.PROCESSED_FILES["ratings_clean"])
    app.state.models_loaded = True
    print("Model artifacts loaded successfully.")

    yield

    # No teardown steps needed — model artifacts are in-memory only.


app = FastAPI(
    title="CineMatch API",
    description="Hybrid movie recommendation engine — content-based + collaborative filtering",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Guarantees the standard error envelope from docs/08_API_Specification.md §3 is
    always returned, never a raw stack trace (per NFR-5, Reliability)."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error_code": exc.error_code, "message": exc.message},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred.",
        },
    )


app.include_router(recommend.router)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": getattr(app.state, "models_loaded", False)}
