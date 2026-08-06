# CineMatch

**A Hybrid Movie Recommendation Engine with Content-Based + Collaborative Filtering, served via REST API and a Streamlit web app.**

---

## Overview

CineMatch recommends movies to users by combining two machine learning approaches:

- **Content-Based Filtering** — recommends movies similar in genre, cast, and theme using CountVectorizer + cosine similarity on a metadata soup of genres, keywords, cast, and director
- **Collaborative Filtering** — recommends movies based on rating patterns across similar users, using SVD-based matrix factorization (truncated SVD via `scipy.sparse.linalg.svds`)
- **Hybrid Scoring** — blends both approaches using an α-weighted formula, automatically adjusting the blend based on how much rating history a user has, and gracefully handling cold-start users who have no history at all

The project is delivered as a full, layered system — model layer, REST API, and web UI — to reflect how recommendation engines are actually structured and deployed in production.

---

## Problem Statement

With thousands of movies available on any streaming platform, users experience decision fatigue and miss content they would genuinely enjoy. Most naive recommenders either fail for new users with no history (the cold-start problem) or return repetitive suggestions that never introduce anything new. CineMatch addresses both by:

1. Falling back to content-based similarity when no collaborative signal is available yet
2. Blending both signals as user history grows, using a principled α-weighting strategy
3. Always returning a ranked, explainable recommendation list — never an error or empty state

---

## Project Documentation

Full design documentation lives in [`docs/`](./docs):

| Doc                                                                                        | Description                                                                           |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| [00. Dataset Sources](./docs/00_Dataset_Sources.md)                                         | MovieLens ml-latest-small + TMDB 5000 — provenance, licensing, schema, join strategy |
| [01. Vision &amp; Product Strategy](./docs/01_Vision_and_Product_Strategy.md)               | Problem statement, objectives, target users, USP                                      |
| [02. Feature Engineering](./docs/02_Feature_Engineering.md)                                 | Every feature used across both models                                                 |
| [03. UI/UX Specification](./docs/03_UIUX_Specification.md)                                  | Screens, navigation flow, design system                                               |
| [04. Software Requirements Specification](./docs/04_Software_Requirements_Specification.md) | IEEE-style functional & non-functional requirements                                   |
| [05. Database &amp; ER Diagram](./docs/05_Database_Design_and_ER_Diagram.md)                | Schema, entities, relationships                                                       |
| [06. System Architecture](./docs/06_System_Architecture.md)                                 | Layered architecture, tech stack, deployment                                          |
| [07. Algorithms &amp; Scoring Logic](./docs/07_Algorithms_and_Scoring_Logic.md)             | CBF, CF, hybrid scoring, cold-start logic, empirical results                          |
| [08. API Specification](./docs/08_API_Specification.md)                                     | REST endpoints, request/response schemas                                              |
| [09. Development Guide](./docs/09_Development_Guide.md)                                     | Folder structure, setup, workflow, testing, deployment                                |
| [justification.md](./docs/justification.md)                                                 | ML competency mapping and project rationale                                           |

---

## Project Structure

```
CineMatch/
├── data/
│   ├── raw/            # Original datasets (MovieLens ml-latest-small, TMDB 5000)
│   └── processed/      # Cleaned and joined CSVs produced by the pipeline
├── models/             # Serialized model artifacts (count matrix, SVD model, index maps)
├── src/                # Core ML pipeline
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── content_based.py
│   ├── collaborative_filtering.py
│   ├── hybrid_model.py
│   └── evaluate.py
├── api/                # FastAPI backend
│   ├── main.py
│   ├── schemas.py
│   └── routers/
│       └── recommend.py
├── app/
│   └── streamlit_app.py
├── notebooks/          # EDA and model training notebook
├── docs/               # Full project documentation
├── tests/              # Unit and integration tests
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── Project_Submission.pdf
└── README.md
```

---

## Tech Stack

| Layer      | Technology                                         |
| ---------- | -------------------------------------------------- |
| Language   | Python 3.10+                                       |
| ML / Data  | pandas, numpy, scikit-learn, scipy                 |
| API        | FastAPI, Uvicorn                                   |
| Frontend   | Streamlit                                          |
| Datasets   | MovieLens ml-latest-small, TMDB 5000 Movie Dataset |
| Deployment | Docker, Streamlit Community Cloud                  |

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd CineMatch
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Place datasets if souces are updated (see docs/00_Dataset_Sources.md for download links)
#    data/raw/ml-latest-small/   <- ratings.csv, movies.csv, links.csv, tags.csv
#    data/raw/tmdb-5000/         <- tmdb_5000_movies.csv, tmdb_5000_credits.csv

# 3. Run the ML pipeline
python src/data_preprocessing.py
python src/content_based.py
python src/collaborative_filtering.py

# 4. (Optional) Run evaluation
python src/evaluate.py

# 5. Start the API
uvicorn api.main:app --reload --port 8000

# 6. Start the frontend (separate terminal)
streamlit run app/streamlit_app.py
```

The interactive API documentation is available at `http://localhost:8000/docs` once the server is running.

Full setup, testing, and deployment instructions: [Development Guide](./docs/09_Development_Guide.md).

---

## Docker

The API can also be built and run as a container:

```bash
docker build -t cinematch-api .
docker run -p 8000:8000 cinematch-api
```

The model training pipeline runs automatically during the image build, so the container is self-contained at startup. See the [Development Guide](./docs/09_Development_Guide.md#7-deployment-instructions) for details.

---

## Evaluation Results

Measured on an 80/20 per-user train/test split of the MovieLens ml-latest-small dataset:

| Strategy           | RMSE   | Precision@10 | Recall@10 |
| ------------------ | ------ | ------------ | --------- |
| Content-based only | —     | 0.5731       | 0.6371    |
| Collaborative only | 0.8895 | 0.6493       | 0.6834    |
| Hybrid             | —     | 0.6352       | 0.6772    |

Content-based catalog coverage: **58.98%** (2,086 / 3,537 content-eligible movies surfaced across sampled queries).

Details and interpretation: [Algorithms &amp; Scoring Logic §8](./docs/07_Algorithms_and_Scoring_Logic.md).

---

## License

MIT License

---

*Submitted as the final project for the CodRexa Virtual Internship in Machine Learning.*
