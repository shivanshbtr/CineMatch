# 02. Feature Engineering
## CineMatch — Feature Catalog

---

### 1. Purpose

This document describes every feature used across CineMatch's models and application layer — what it represents, where it comes from, how it is transformed, and how it is consumed downstream.

---

### 2. Feature Categories

CineMatch uses two distinct feature sets, one per recommendation strategy, plus a small set of application-level features.

---

### 3. Content-Based Filtering Features

| Feature | Source Column(s) | Type | Transformation | Purpose |
|---|---|---|---|---|
| Genre vector | `genres` | Categorical (multi-label) | Multi-hot encoding / TF-IDF over genre tokens | Captures thematic similarity between movies |
| Keyword/tag vector | `keywords` | Text | TF-IDF vectorization | Captures fine-grained plot/theme similarity beyond broad genre |
| Cast & crew | `cast`, `director` | Text | Top-N cast members + director concatenated into a "soup", then vectorized | Captures similarity driven by talent (users who like a director/actor's other work) |
| Overview embedding | `overview` | Free text | TF-IDF (or CountVectorizer) on cleaned synopsis text | Captures narrative/topical similarity |
| Combined "metadata soup" | genres + keywords + cast + director | Text | Concatenated string → single TF-IDF/CountVectorizer matrix | Single unified representation of "what a movie is" used to compute cosine similarity |

**Output:** An item-item cosine similarity matrix, where each movie has a ranked list of most-similar movies.

---

### 4. Collaborative Filtering Features

| Feature | Source Column(s) | Type | Transformation | Purpose |
|---|---|---|---|---|
| User ID | `userId` | Categorical (ID) | Encoded to matrix index | Identifies the user dimension of the interaction matrix |
| Movie ID | `movieId` | Categorical (ID) | Encoded to matrix index | Identifies the item dimension of the interaction matrix |
| Rating | `rating` | Numeric (0.5–5.0) | Used directly as the matrix value; centered/normalized per user during SVD | Signal of user preference strength — the core learning target |
| Latent user factors | Derived | Numeric vector (learned) | Output of SVD/matrix factorization | Compact representation of a user's taste |
| Latent item factors | Derived | Numeric vector (learned) | Output of SVD/matrix factorization | Compact representation of a movie's "appeal profile" |

**Output:** A predicted rating for any (user, movie) pair, used to rank unseen movies for that user.

---

### 5. Hybrid Layer Features

| Feature | Description | Purpose |
|---|---|---|
| Content similarity score | Normalized cosine similarity (0–1) | Contribution from content-based model |
| Predicted rating score | Normalized predicted rating (0–1) | Contribution from collaborative model |
| Confidence weight (α) | Tunable weight, adjusted by data availability (e.g., number of ratings a user has) | Determines the blend ratio between the two models — higher α toward collaborative filtering as user history grows |

**Hybrid score formula:**

```
final_score = α * collaborative_score + (1 - α) * content_score
```

Where α is small (or 0) for new users (cold-start → content-based dominates) and increases as the user accumulates more ratings.

---

### 6. Application-Level Features (non-ML)

| Feature | Description |
|---|---|
| Search/select movie | User selects a reference movie to get "more like this" |
| User ID input | User selects/enters an ID to get personalized picks (demo dataset users) |
| Top-N selector | Number of recommendations to return (default 10) |
| Explanation tag | UI label indicating why a movie was recommended (e.g., "Because you liked...", "Popular with similar users") |

---

### 7. Data Cleaning Notes

- Missing `overview`, `genres`, or `keywords` values are replaced with empty strings prior to vectorization to avoid pipeline failures.
- Duplicate movie titles (common in raw datasets) are deduplicated by `movieId`.
- Ratings dataset is filtered to remove users/movies with fewer than a minimum interaction threshold to reduce sparsity noise (configurable, default: users with < 5 ratings excluded from CF training, still served via content-based fallback).

---

### 8. Feature Store (Conceptual)

For this project's scope, engineered features are precomputed and cached as serialized artifacts rather than a full feature store:

- `tfidf_matrix.pkl` — content-based vector space
- `count_matrix.pkl` — sparse term-count matrix over the metadata soup (item-item similarity is computed on-demand from this at query time, not precomputed as a dense matrix — a full 9,742×9,742 dense similarity matrix would be ~725MB; see `07_Algorithms_and_Scoring_Logic.md` §2 for the rationale)
- `svd_model.pkl` — trained collaborative filtering model
- `movie_index_map.pkl` — mapping between movie titles/IDs and matrix indices

This precomputation keeps API response times low since expensive similarity computation is done offline, not at request time.
