# 09. Development Guide

## CineMatch — Setup, Structure & Workflow

---

### 1. Project Folder Structure

```
CineMatch/
├── data/
│   ├── raw/                     # Original MovieLens + TMDB CSVs (see docs/00_Dataset_Sources.md)
│   └── processed/               # Cleaned datasets
├── models/
│   ├── tfidf_matrix.pkl
│   ├── count_matrix.pkl
│   ├── svd_model.pkl
│   └── movie_index_map.pkl
├── src/
│   ├── data_preprocessing.py    # Cleaning, merging, feature prep
│   ├── content_based.py         # CBF engine (TF-IDF + cosine similarity)
│   ├── collaborative_filtering.py # CF engine (SVD training + inference)
│   ├── hybrid_model.py          # Hybrid scoring + cold-start logic
│   └── evaluate.py              # RMSE, Precision@K, Recall@K
├── api/
│   ├── main.py                  # FastAPI app entrypoint
│   ├── routers/
│   │   └── recommend.py         # Endpoint definitions
│   └── schemas.py               # Pydantic request/response models
├── app/
│   └── streamlit_app.py         # Frontend UI
├── notebooks/
│   └── EDA_and_Model_Training.ipynb
├── docs/                        # This documentation set
├── tests/
│   ├── test_content_based.py
│   ├── test_collaborative.py
│   └── test_api.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

### 2. Coding Standards

- **Style:** PEP8, enforced via `black` and `flake8`
- **Naming:** `snake_case` for functions/variables, `PascalCase` for Pydantic models/classes
- **Docstrings:** Every public function has a docstring describing inputs, outputs, and purpose
- **Type hints:** Required on all function signatures in `src/` and `api/`
- **Modularity:** Model logic (`src/`) is kept fully independent of the API layer (`api/`) — the API imports from `src/`, never the other way around

---

### 3. Setup Instructions

```bash
# 1. Clone the repository
git clone <repo-url>
cd CineMatch

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets (see docs/00_Dataset_Sources.md for links & details)
# Place MovieLens ml-latest-small files (ratings.csv, movies.csv, links.csv, tags.csv)
#   into data/raw/ml-latest-small/
# Place TMDB 5000 files (tmdb_5000_movies.csv, tmdb_5000_credits.csv)
#   into data/raw/tmdb-5000/

# 5. Run preprocessing + train models
python src/data_preprocessing.py
python src/content_based.py
python src/collaborative_filtering.py

# 6. Start the API
uvicorn api.main:app --reload --port 8000

# 7. Start the frontend (in a separate terminal)
streamlit run app/streamlit_app.py
```

---

### 4. Development Workflow

1. **Data stage** — run `data_preprocessing.py` whenever the raw dataset changes; outputs cleaned CSVs to `data/processed/`.
2. **Model stage** — run each model script independently; each saves its own artifact(s) to `models/`. This modularity means retraining one model doesn't require retraining the other.
3. **API stage** — `api/main.py` loads all precomputed artifacts once at startup (not per-request) for performance.
4. **UI stage** — `streamlit_app.py` only talks to the API over HTTP; it never imports `src/` directly, preserving the service boundary described in the System Architecture doc.
5. **Testing stage** — run `pytest tests/` before any commit that touches `src/` or `api/`.

---

### 5. Testing Approach

| Test Type         | Coverage                                                                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Unit tests        | Individual scoring functions (cosine similarity lookup, hybrid weight selection, popularity formula)                                      |
| Integration tests | API endpoints return correct schema and status codes for valid/invalid inputs                                                             |
| Regression checks | Precision@K / Recall@K / RMSE re-computed after any model change and compared against the last recorded baseline in`evaluate.py` output |

---

### 6. Implementation Guidelines

- Never compute the full cosine similarity matrix inside a request handler — always precompute offline (see Algorithms doc, §7 rationale).
- All model artifacts are versioned by filename convention (e.g., `svd_model_v1.pkl`) so a regression can be traced back to a specific model version.
- Config values (thresholds, α weight table, Top-K bounds) are centralized in a single `config.py`, not hardcoded across files.

---

### 7. Deployment Instructions

**Docker build (API):**

```bash
docker build -t cinematch-api .
docker run -p 8000:8000 cinematch-api
```

**Streamlit Community Cloud (Frontend):**

1. Push repository to GitHub
2. Connect repo in Streamlit Community Cloud
3. Set `API_BASE_URL` environment variable to point to the deployed API
4. Deploy — app is live at a shareable public URL

**Environment Variables**

| Variable                  | Purpose                             |
| ------------------------- | ----------------------------------- |
| `API_BASE_URL`          | Base URL the Streamlit app calls    |
| `MODEL_DIR`             | Path to precomputed model artifacts |
| `MIN_RATINGS_THRESHOLD` | Cold-start cutoff (default: 5)      |

---

### 8. Contribution Checklist

- [ ] All 10 docs (00–09) + justification.md present in `docs/`
- [ ] `src/`, `api/`, `app/` implemented and runnable end-to-end
- [ ] `notebooks/EDA_and_Model_Training.ipynb` contains EDA, training, and evaluation with visible metric outputs
- [ ] `README.md` links to this documentation set and shows a quickstart
- [ ] Tests pass via `pytest tests/`
