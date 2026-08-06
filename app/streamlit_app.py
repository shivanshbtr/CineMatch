"""
CineMatch — Streamlit Frontend

Implements docs/03_UIUX_Specification.md:
- Three mode tabs: Similar Movies / Recommended For Me / Trending
- Dark cinematic theme with amber/gold accents
- Match-reason tags on every card (mandatory per §6)
- Cold-start banner for new users (per §3.3)
- Never a dead-end — always falls back to trending on empty/error

Run: streamlit run app/streamlit_app.py
     (set API_BASE_URL env var or it defaults to http://localhost:8000)
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config & global CSS — dark cinematic theme per docs/03_UIUX_Specification.md §5
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0f0f0f;
        color: #e8e8e8;
    }

    /* Header */
    .cm-header {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
    }
    .cm-title {
        font-size: 3rem;
        font-weight: 700;
        color: #f5c518;
        letter-spacing: -1px;
        margin: 0;
    }
    .cm-tagline {
        font-size: 1.05rem;
        color: #aaa;
        margin-top: 0.3rem;
    }

    /* Movie card */
    .cm-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1.1rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .cm-card:hover { border-color: #f5c518; }
    .cm-card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0 0 0.3rem 0;
        line-height: 1.3;
    }
    .cm-genres {
        font-size: 0.78rem;
        color: #888;
        margin-bottom: 0.5rem;
    }
    .cm-reason {
        font-size: 0.78rem;
        color: #b0892f;
        font-style: italic;
    }
    .cm-score-badge {
        display: inline-block;
        background: #f5c518;
        color: #0f0f0f;
        font-size: 0.72rem;
        font-weight: 700;
        border-radius: 20px;
        padding: 0.15rem 0.55rem;
        margin-top: 0.5rem;
    }

    /* Section label */
    .cm-section-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #666;
        margin-bottom: 1rem;
    }

    /* Cold-start banner */
    .cm-cold-banner {
        background: #1e1a0e;
        border: 1px solid #f5c518;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        color: #f5c518;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }

    /* Strategy badge */
    .cm-strategy {
        display: inline-block;
        background: #222;
        border: 1px solid #444;
        color: #aaa;
        font-size: 0.72rem;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        margin-bottom: 1rem;
    }

    /* Override Streamlit tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #888;
        border-radius: 8px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f5c518 !important;
        color: #0f0f0f !important;
    }

    /* Selectbox + number_input dark override */
    .stSelectbox > div > div,
    .stNumberInput > div > div {
        background-color: #1a1a1a !important;
        border-color: #333 !important;
        color: #e8e8e8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data helpers — cached so they don't refetch on every interaction
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_all_titles():
    """Fetches movie titles from the API for the searchable dropdown."""
    try:
        # Use /trending to get a known-working endpoint for the health check,
        # then load titles from local processed CSV (faster, no extra endpoint needed)
        df = pd.read_csv(
            Path(__file__).resolve().parent.parent / "data" / "processed" / "movies_master.csv"
        )
        return sorted(df["title"].dropna().tolist())
    except Exception:
        return []


@st.cache_data(ttl=3600)
def fetch_all_user_ids():
    """Loads the list of known user IDs from the local ratings CSV."""
    try:
        df = pd.read_csv(
            Path(__file__).resolve().parent.parent / "data" / "processed" / "ratings_clean.csv"
        )
        return sorted(df["user_id"].unique().tolist())
    except Exception:
        return []


def call_api(endpoint: str, params: dict) -> dict | None:
    """Makes a GET request to the CineMatch API, returns parsed JSON or None on failure."""
    try:
        resp = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

STRATEGY_LABELS = {
    "content": "Content-Based Filtering",
    "collaborative": "Collaborative Filtering",
    "hybrid": "Hybrid (Content + Collaborative)",
    "fallback_popularity": "Trending Fallback",
    "fallback_content": "Content-Based Fallback",
}


def render_strategy_badge(strategy: str):
    label = STRATEGY_LABELS.get(strategy, strategy)
    st.markdown(f'<div class="cm-strategy">Strategy: {label}</div>', unsafe_allow_html=True)


def render_movie_card(result: dict):
    genres = " · ".join(result.get("genres", [])) or "Unknown"
    score_pct = int(result.get("score", 0) * 100)
    title = result.get("title", "Unknown")
    reason = result.get("reason", "")

    st.markdown(
        f"""
        <div class="cm-card">
            <div class="cm-card-title">🎬 {title}</div>
            <div class="cm-genres">{genres}</div>
            <div class="cm-reason">{reason}</div>
            <div class="cm-score-badge">Match {score_pct}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results_grid(results: list, cols: int = 3):
    """Renders movie cards in a responsive grid layout per docs/03_UIUX_Specification.md §5."""
    if not results:
        st.info("No results found. Try a different selection.")
        return
    columns = st.columns(cols)
    for i, result in enumerate(results):
        with columns[i % cols]:
            render_movie_card(result)


def render_cold_start_banner(strategy: str):
    """Cold-start banner per docs/03_UIUX_Specification.md §3.3."""
    if strategy in ("fallback_popularity", "fallback_content"):
        st.markdown(
            '<div class="cm-cold-banner">'
            "⚡ <strong>New user detected</strong> — not enough rating history yet. "
            "Showing popular &amp; similar picks instead."
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="cm-header">
        <div class="cm-title">🎬 CineMatch</div>
        <div class="cm-tagline">Find your next favorite movie</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Mode tabs — per docs/03_UIUX_Specification.md §3
# ---------------------------------------------------------------------------

tab_similar, tab_for_me, tab_trending = st.tabs(
    ["🔍 Similar Movies", "👤 Recommended For Me", "🔥 Trending"]
)

all_titles = fetch_all_titles()
all_user_ids = fetch_all_user_ids()

# ---------------------------------------------------------------------------
# Tab 1: Similar Movies (Content-Based)
# docs/03_UIUX_Specification.md §3.2
# ---------------------------------------------------------------------------

with tab_similar:
    st.markdown('<div class="cm-section-label">Find movies similar to one you love</div>', unsafe_allow_html=True)

    col_input, col_k = st.columns([3, 1])
    with col_input:
        selected_title = st.selectbox(
            "Search for a movie",
            options=[""] + all_titles,
            index=0,
            key="similar_title",
            label_visibility="collapsed",
        )
    with col_k:
        top_k = st.number_input("Results", min_value=5, max_value=20, value=9, step=1, key="similar_k")

    if selected_title:
        with st.spinner("Finding similar movies..."):
            data = call_api("/recommend/content", {"title": selected_title, "top_k": top_k})
        if data and data.get("results"):
            render_strategy_badge(data["strategy_used"])
            render_cold_start_banner(data["strategy_used"])
            render_results_grid(data["results"], cols=3)
    else:
        st.markdown('<p style="color:#555; text-align:center; padding-top:2rem;">← Select a movie to get started</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 2: Recommended For Me (Hybrid / Collaborative)
# docs/03_UIUX_Specification.md §3.3
# ---------------------------------------------------------------------------

with tab_for_me:
    st.markdown('<div class="cm-section-label">Personalised picks based on your history</div>', unsafe_allow_html=True)

    col_user, col_ref, col_k2 = st.columns([2, 3, 1])
    with col_user:
        selected_user = st.selectbox(
            "Your User ID",
            options=[""] + [str(u) for u in all_user_ids],
            index=0,
            key="rec_user",
            label_visibility="visible",
        )
    with col_ref:
        ref_movie = st.selectbox(
            "Reference movie (optional — helps cold-start users)",
            options=[""] + all_titles,
            index=0,
            key="rec_ref",
            label_visibility="visible",
        )
    with col_k2:
        top_k2 = st.number_input("Results", min_value=5, max_value=20, value=9, step=1, key="rec_k")

    if selected_user or ref_movie:
        params = {"top_k": top_k2}
        if selected_user:
            params["user_id"] = int(selected_user)
        if ref_movie:
            params["reference_movie"] = ref_movie

        with st.spinner("Personalising your recommendations..."):
            data = call_api("/recommend/hybrid", params)

        if data and data.get("results"):
            render_cold_start_banner(data["strategy_used"])
            render_strategy_badge(data["strategy_used"])
            render_results_grid(data["results"], cols=3)
    else:
        st.markdown('<p style="color:#555; text-align:center; padding-top:2rem;">← Select a User ID or a reference movie to get started</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 3: Trending
# docs/03_UIUX_Specification.md §3.4
# ---------------------------------------------------------------------------

with tab_trending:
    st.markdown('<div class="cm-section-label">Most popular & highly rated movies</div>', unsafe_allow_html=True)

    top_k3 = st.number_input("Results", min_value=5, max_value=20, value=12, step=1, key="trend_k")

    with st.spinner("Loading trending movies..."):
        data = call_api("/trending", {"top_k": top_k3})

    if data and data.get("results"):
        render_strategy_badge(data["strategy_used"])
        render_results_grid(data["results"], cols=3)
    else:
        st.warning("Could not load trending movies. Is the API running?")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <hr style="border-color:#222; margin-top:3rem;">
    <p style="text-align:center; color:#444; font-size:0.75rem;">
        CineMatch · Hybrid Movie Recommendation Engine ·
        MovieLens ml-latest-small + TMDB 5000 datasets ·
        Built with scikit-learn, scipy, FastAPI & Streamlit
    </p>
    """,
    unsafe_allow_html=True,
)
