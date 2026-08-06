# 06. System Architecture
## CineMatch — Architecture Overview

---

### 1. Architecture Style

CineMatch follows a **layered, service-oriented architecture**: a clear separation between data/model layer, API layer, and presentation layer. This mirrors how recommendation systems are structured in production (e.g., a model-serving layer behind a REST API, consumed by one or more frontends).

---

### 2. High-Level Architecture Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                        Presentation Layer                       │
│                    (Streamlit Web Application)                  │
│   - Home / Mode tabs   - Search & Select   - Results Grid        │
└───────────────────────────┬─────────────────────────────────────┘
                             │ HTTP (REST calls)
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                          API Layer                              │
│                     (FastAPI + Uvicorn)                          │
│  Endpoints:                                                      │
│   /recommend/content     /recommend/collaborative                │
│   /recommend/hybrid      /trending      /health                 │
└───────────────────────────┬─────────────────────────────────────┘
                             │ function calls
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                       Model / Service Layer                      │
│  ┌─────────────────────┐   ┌────────────────────────┐           │
│  │ Content-Based Engine  │   │ Collaborative Engine    │          │
│  │ (TF-IDF + Cosine Sim)│   │ (SVD Matrix Factorization)│         │
│  └───────────┬──────────┘   └────────────┬────────────┘          │
│              │                            │                       │
│              └─────────────┬──────────────┘                       │
│                             ▼                                     │
│                    Hybrid Scoring Module                          │
│                 (weighted blend + cold-start logic)                │
└───────────────────────────┬─────────────────────────────────────┘
                             │ load/read
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                        Data / Storage Layer                      │
│  - MovieLens ml-latest-small CSVs + TMDB 5000 CSVs (raw)          │
│  - Precomputed artifacts: tfidf_matrix.pkl, cosine_sim.pkl,       │
│    svd_model.pkl, movie_index_map.pkl                             │
└───────────────────────────────────────────────────────────────┘
```

---

### 3. Component Responsibilities

| Component | Responsibility |
|---|---|
| **Streamlit UI** | Collects user input, calls API endpoints, renders results |
| **FastAPI service** | Validates requests, routes to correct model logic, formats responses, handles errors gracefully |
| **Content-Based Engine** | Computes/serves item-item similarity from precomputed TF-IDF + cosine similarity matrix |
| **Collaborative Engine** | Serves rating predictions from a precomputed SVD model |
| **Hybrid Scoring Module** | Merges outputs of both engines using the α-weighted formula, applies cold-start detection |
| **Data/Storage Layer** | Holds raw datasets and precomputed model artifacts loaded at service startup |

---

### 4. Frontend–Backend Communication

- Communication is via synchronous HTTP requests (JSON payloads) from the Streamlit app to the FastAPI backend.
- Each API call is stateless — the frontend passes all required identifiers (movie title / user ID / K) on every request; no server-side session state is required.
- Response schema is consistent across all three recommendation endpoints (see API Specification doc) so the frontend can render results uniformly regardless of which strategy was used.

---

### 5. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML/Data | pandas, numpy, scikit-learn, scipy (SVD via `scipy.sparse.linalg.svds`) |
| API Framework | FastAPI, Uvicorn (ASGI server) |
| Frontend | Streamlit |
| Model Serialization | pickle / joblib |
| Containerization | Docker |
| Version Control | Git |

---

### 6. Deployment Structure

```
                ┌─────────────────────┐
                │   Docker Container    │
                │  ┌─────────────────┐  │
                │  │  FastAPI Backend  │  │
                │  │  (Uvicorn server) │  │
                │  └─────────────────┘  │
                └───────────┬───────────┘
                             │
                Deployed to: Render / Railway
                             │
                ┌───────────┴───────────┐
                │   Streamlit Frontend    │
                │ (Streamlit Community    │
                │       Cloud)             │
                └─────────────────────────┘
```

- Backend and frontend are containerized/deployed independently, communicating over a public API URL — reflecting a realistic microservice deployment pattern even at small scale.
- Model artifacts are baked into the backend image at build time (no external model registry needed for this project's scope).

---

### 7. Design Rationale

- **Why separate API and UI processes instead of one Streamlit app doing everything?** It establishes a proper service boundary and makes the ML logic reusable by any future client (mobile app, another UI), not just this one Streamlit frontend — a key trait of well-designed system architecture.
- **Why precompute similarity/model artifacts instead of computing on request?** Similarity matrix computation over the full catalog is expensive; precomputing keeps API latency low and predictable (NFR-1 in the SRS).
