# Justification.md
## CineMatch — Final Project Submission Justification
### CodRexa Virtual Internship in Machine Learning

---

### 1. Purpose of This Document

This document serves as the formal justification for CineMatch as the final project submission for the **CodRexa Virtual Internship in Machine Learning**. It demonstrates that the project fulfils the scope, depth, and competency requirements expected of a final ML project by mapping every major deliverable to a corresponding ML domain requirement.

---

### 2. Project Overview

**Project Name:** CineMatch — A Hybrid Movie Recommendation Engine  
**Domain:** Machine Learning — Recommender Systems  
**Techniques Used:** Content-Based Filtering (CountVectorizer + Cosine Similarity), Collaborative Filtering (SVD-based Matrix Factorisation), Hybrid Scoring with Cold-Start Handling  
**Datasets:** MovieLens ml-latest-small (GroupLens Research), TMDB 5000 Movie Dataset (Kaggle/TMDB)  
**Deliverables:** Trained ML models, REST API, Streamlit web application, full project documentation

CineMatch is not a toy example or a tutorial reproduction. It is a fully functional, end-to-end recommendation system built from scratch, addressing a real-world problem (content discovery at scale) using multiple machine learning techniques, evaluated with domain-appropriate metrics, and deployed as a usable product.

---

### 3. Fulfilment of Machine Learning Final Project Requirements

#### 3.1 Real-World Problem Statement ✓

CineMatch solves a well-defined, industrially relevant problem: given a large catalog of movies and a user's interaction history, surface the most relevant titles that the user is likely to enjoy. This is the same class of problem solved by Netflix, Amazon, and Spotify recommendation engines. The problem is formally documented in `01_Vision_and_Product_Strategy.md`, including a problem statement, target users, and measurable objectives.

#### 3.2 Dataset Selection and Preparation ✓

Two publicly available, well-documented datasets are used:
- **MovieLens ml-latest-small** (GroupLens Research) — 100,836 ratings across 9,742 movies by 610 users
- **TMDB 5000 Movie Dataset** (Kaggle/TMDB) — rich metadata (overview, keywords, cast, crew) for ~4,800 movies

Both datasets are documented in `00_Dataset_Sources.md` with provenance, licensing terms, file schemas, and the join strategy used to combine them. The preprocessing pipeline (`src/data_preprocessing.py`) performs data cleaning, feature extraction from JSON-stringified columns, deduplication, and dataset merging — producing clean, analysis-ready outputs.

#### 3.3 Feature Engineering ✓

Feature engineering is performed at two levels:
- **Content-based features:** A "metadata soup" is constructed per movie by combining genres, keywords, top-3 cast members, and director into a single text representation, then vectorised using CountVectorizer into a sparse term-count matrix
- **Collaborative features:** Latent user and item factor vectors are learned automatically during SVD matrix factorisation, capturing implicit preference dimensions beyond explicit metadata

Full feature documentation is in `02_Feature_Engineering.md`.

#### 3.4 Machine Learning Model Implementation ✓

Two distinct ML models are implemented from core libraries (not black-box wrappers):

**Model 1 — Content-Based Filtering (`src/content_based.py`)**
- Algorithm: Cosine similarity over CountVectorizer-produced sparse vectors
- Type: Unsupervised similarity retrieval
- Implementation: scikit-learn CountVectorizer + scipy cosine_similarity

**Model 2 — Collaborative Filtering (`src/collaborative_filtering.py`)**
- Algorithm: Truncated SVD (matrix factorisation) with user and item bias correction
- Type: Latent variable model trained on user-item rating interactions
- Implementation: scipy.sparse.linalg.svds on a bias-adjusted sparse residual matrix

**Model 3 — Hybrid Scoring (`src/hybrid_model.py`)**
- Combines both models using a principled α-weighted blend
- α is determined by the user's rating history depth, implementing explicit cold-start handling
- Ensures a recommendation is always returned — never an error or empty state

#### 3.5 Model Training and Validation ✓

- The collaborative filtering model is trained on an **80/20 per-user train/test split** (random_state=42 for reproducibility)
- Users with fewer than 5 ratings are excluded from the split to avoid data leakage
- Training and evaluation are fully automated and reproducible via `src/collaborative_filtering.py`

#### 3.6 Model Evaluation with Appropriate Metrics ✓

Three strategies are evaluated on the same held-out test split:

| Strategy | RMSE | Precision@10 | Recall@10 |
|---|---|---|---|
| Content-based only | — | 0.5731 | 0.6371 |
| Collaborative only | 0.8895 | 0.6493 | 0.6834 |
| Hybrid | — | 0.6352 | 0.6772 |
| Content-based coverage | — | 58.98% of eligible catalog | — |

**RMSE** (Root Mean Squared Error) is used for rating prediction quality — standard for collaborative filtering evaluation.  
**Precision@K and Recall@K** are used for ranking quality — the correct metrics for recommendation tasks, more informative than plain accuracy since the goal is ranking, not classification.

The evaluation logic is in `src/evaluate.py` and results are saved to `models/evaluation_report.json`.

#### 3.7 Handling Real-World ML Challenges ✓

Two classic, non-trivial challenges in recommender systems are explicitly identified and addressed:

1. **Cold-Start Problem** — New users with no rating history cannot be served by collaborative filtering alone. CineMatch detects this condition and falls back to content-based similarity or popularity-weighted trending, transparently communicating the strategy to the user via the UI.

2. **Data Sparsity** — The user-item matrix is 98.3% sparse. SVD-based matrix factorisation is used specifically because it handles sparse matrices well — outperforming neighbourhood-based collaborative filtering (kNN) under high sparsity.

#### 3.8 End-to-End Deployment ✓

The project is not limited to a training notebook. It is delivered as a complete, running system:

- **`src/`** — ML pipeline (preprocessing → training → evaluation)
- **`api/`** — FastAPI REST API serving all three recommendation strategies across 5 documented endpoints
- **`app/`** — Streamlit web application with a dark cinematic UI, three navigation modes, match-reason tags on every card, and a cold-start fallback banner

The system is runnable locally with `pip install -r requirements.txt` and two commands (`uvicorn` + `streamlit`), and is containerisable via the included `Dockerfile`.

#### 3.9 Documentation ✓

The `docs/` folder contains 10 professional planning and design documents written prior to implementation:

| Document | Coverage |
|---|---|
| Vision & Product Strategy | Problem statement, objectives, USP |
| Feature Engineering | Every feature across both models |
| UI/UX Specification | Screens, navigation, design system |
| Software Requirements Specification (IEEE) | Functional & non-functional requirements, user stories, use cases |
| Database Design & ER Diagram | Schema, entities, relationships |
| System Architecture | Layered architecture, tech stack, deployment |
| Algorithms & Scoring Logic | Full algorithm documentation with empirical results |
| API Specification | All endpoints, request/response schemas, error handling |
| Development Guide | Setup, structure, workflow, testing, deployment |
| Dataset Sources | Provenance, licensing, schema, join strategy |

---

### 4. Machine Learning Concepts Covered

| Concept | Where Demonstrated |
|---|---|
| Unsupervised learning | Content-based cosine similarity (no labelled training signal) |
| Supervised/latent variable modelling | SVD matrix factorisation on user-item ratings |
| Feature engineering from text | Metadata soup construction and CountVectorizer pipeline |
| Train/test splitting | Per-user 80/20 split in `src/collaborative_filtering.py` |
| Bias correction | Global mean + user bias + item bias baseline predictors |
| Evaluation metrics | RMSE, Precision@K, Recall@K, Coverage |
| Model comparison | Three-strategy ablation on the same held-out test set |
| Cold-start handling | α-weighted hybrid with graceful fallback logic |
| Model serialisation | pickle-based artifact storage and API-time loading |
| End-to-end deployment | REST API + web UI serving trained model predictions |

---

### 5. Summary

CineMatch fulfils all the requirements of a final project for the CodRexa Virtual Internship in Machine Learning. It demonstrates applied ML across multiple paradigms, addresses real-world data challenges with documented solutions, is evaluated with appropriate domain-specific metrics, and is delivered as a fully operational system rather than a standalone notebook. The project reflects both machine learning competency and the engineering discipline required to take a model from experimentation to a deployable, documented product.
