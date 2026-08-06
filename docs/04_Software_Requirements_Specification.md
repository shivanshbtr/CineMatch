# 04. Software Requirements Specification (IEEE Style)
## CineMatch — A Hybrid Movie Recommendation Engine

---

### 1. Introduction

#### 1.1 Purpose
This document specifies the functional and non-functional requirements for CineMatch, a hybrid movie recommendation system, following an IEEE-inspired SRS structure.

#### 1.2 Scope
CineMatch recommends movies to users using content-based filtering, collaborative filtering, and a hybrid combination of both, served via a REST API and consumed by a web-based frontend.

#### 1.3 Definitions, Acronyms, Abbreviations
- **CF** — Collaborative Filtering
- **CBF** — Content-Based Filtering
- **SVD** — Singular Value Decomposition
- **Cold Start** — The problem of generating recommendations for new users/items with no historical interaction data
- **Top-K** — The top K ranked recommendation results returned to a user

#### 1.4 References
- MovieLens ml-latest-small Dataset Documentation (GroupLens Research) — see `00_Dataset_Sources.md`
- TMDB 5000 Movie Dataset Documentation (Kaggle/TMDB) — see `00_Dataset_Sources.md`
- Scikit-learn, Surprise library documentation
- FastAPI official documentation

---

### 2. Overall Description

#### 2.1 Product Perspective
CineMatch is a standalone academic/portfolio project, architected in the style of a production recommendation microservice: model layer, API layer, and presentation layer are separated.

#### 2.2 Product Functions (Summary)
- Generate content-based recommendations from a reference movie
- Generate collaborative-filtering-based recommendations for a known user
- Generate hybrid recommendations blending both approaches
- Serve trending/popular movies as a cold-start fallback
- Expose all of the above via REST API endpoints
- Provide a web UI to interact with the above

#### 2.3 User Classes
- **End User** — interacts via the web UI to get recommendations
- **Developer/Evaluator** — interacts via API directly or reviews code/docs

#### 2.4 Operating Environment
- Backend: Python 3.10+, FastAPI, Uvicorn
- ML Libraries: scikit-learn, pandas, numpy, scipy (SVD via `scipy.sparse.linalg.svds` — see `07_Algorithms_and_Scoring_Logic.md` §3)
- Frontend: Streamlit (or lightweight HTML/JS client consuming the API)
- Deployment target: Local / Streamlit Community Cloud / Render (containerized via Docker)

#### 2.5 Assumptions & Dependencies
- Datasets (MovieLens ml-latest-small + TMDB 5000 Movie Dataset) are static and pre-downloaded; no live data ingestion pipeline is in scope
- User authentication is simulated via a User ID selector, not a full auth system
- Model retraining is a manual/offline process, not real-time/online learning

---

### 3. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | The system shall accept a movie title and return the Top-K most similar movies using content-based filtering | High |
| FR-2 | The system shall accept a user ID and return Top-K personalized recommendations using collaborative filtering | High |
| FR-3 | The system shall combine content-based and collaborative scores into a single hybrid ranked list | High |
| FR-4 | The system shall detect cold-start users (< minimum ratings threshold) and fall back to content-based/popularity recommendations | High |
| FR-5 | The system shall expose recommendation functions via documented REST API endpoints | High |
| FR-6 | The system shall return a human-readable "reason" tag alongside each recommendation | Medium |
| FR-7 | The system shall provide a fallback "Trending" list when no valid input is available | Medium |
| FR-8 | The web UI shall allow searching/selecting a movie via autocomplete | Medium |
| FR-9 | The system shall log basic request/response metadata for debugging purposes | Low |

---

### 4. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | **Performance** — API shall return recommendations within 2 seconds for the demo dataset size |
| NFR-2 | **Scalability** — Model artifacts are precomputed/cached to avoid recomputation at request time |
| NFR-3 | **Usability** — UI must be understandable without a manual, per the UI/UX spec |
| NFR-4 | **Maintainability** — Codebase separated into clear modules (data, models, API, UI) per the Development Guide |
| NFR-5 | **Reliability** — System must never return a hard error to the end user; must degrade gracefully to a fallback recommendation |
| NFR-6 | **Portability** — Application shall be containerizable via Docker for platform-independent deployment |

---

### 5. User Stories

1. *As a new user with no rating history, I want to get relevant movie suggestions immediately, so that I don't hit a dead end on my first visit.*
2. *As a returning user, I want recommendations that improve based on my rating history, so that suggestions feel personalized over time.*
3. *As a user watching a specific movie, I want to see similar movies, so that I can find my next watch easily.*
4. *As a reviewer, I want to understand why a recommendation was made, so that I can trust the system is not a black box.*

---

### 6. Use Cases

**UC-1: Get Similar Movies**
- **Actor:** End User
- **Precondition:** A valid movie title exists in the dataset
- **Flow:** User selects a movie → system computes/retrieves cosine similarity → returns Top-K similar movies with reason tags
- **Postcondition:** User sees a ranked list of similar movies

**UC-2: Get Personalized Recommendations**
- **Actor:** End User
- **Precondition:** A valid user ID exists
- **Flow:** User selects ID → system checks rating count → if sufficient, runs CF/hybrid scoring; if not, falls back to CBF/popularity → returns Top-K recommendations
- **Postcondition:** User sees personalized (or fallback) recommendations

---

### 7. Business Rules & Validation Rules

- A movie title input must exactly match (or be selected from) an existing entry in the dataset — free-text typos are handled via autocomplete, not fuzzy backend matching, in this version.
- Top-K is bounded between 5 and 20 to keep the UI and API predictable.
- Users with fewer than 5 ratings are treated as cold-start and routed to the fallback path.
- Hybrid weight α is bounded between 0 and 1 and is deterministic given a user's rating count (not randomized).

---

### 8. Acceptance Criteria

- Given a valid movie title, the API returns exactly K similar movies with non-null reason tags.
- Given a valid user ID with sufficient history, the API returns personalized recommendations distinct from the pure popularity list.
- Given a new/unknown user ID, the API returns a non-empty fallback recommendation list rather than an error.
- All three recommendation modes are reachable and functional from the UI.

---

### 9. Future Scope

- Real-time online learning from live user feedback (click-through, watch time)
- Deep learning-based sequential/session-aware recommendations
- Full user authentication and persistent rating storage
- A/B testing framework comparing recommendation strategies in production
