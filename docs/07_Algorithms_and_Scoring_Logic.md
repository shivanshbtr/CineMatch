# 07. Algorithms & Scoring Logic
## CineMatch — Core ML Algorithms

---

### 1. Overview

This document details the three core algorithmic components of CineMatch: the Content-Based Engine, the Collaborative Filtering Engine, and the Hybrid Scoring Logic that combines them, including the cold-start decision rule.

---

### 2. Content-Based Filtering Algorithm

**Goal:** Given a movie, find other movies that are most similar in content.

**Steps:**
1. Build a "metadata soup" per movie: concatenate genres + keywords + top-3 cast + director into a single text string.
2. Vectorize the corpus of metadata soups using `CountVectorizer` (or `TfidfVectorizer`) to produce a document-term matrix.
3. Compute pairwise **cosine similarity** across all movies:

```
similarity(A, B) = (A · B) / (‖A‖ * ‖B‖)
```

4. For a given input movie, sort all other movies by similarity score descending, and return the Top-K.

**Why cosine similarity:** It measures the *angle* between feature vectors rather than magnitude, which is appropriate here because vector length (number of shared words/tags) shouldn't dominate — two sparse-but-well-matched movies should score as similar as two verbose-but-matched ones.

**Implementation note — on-demand computation, not a precomputed dense matrix:** A full n×n similarity matrix for ~9,742 movies would require storing ~95 million float64 values (~725MB), which is impractical to persist or version. The sparse term-count matrix (a few hundred KB) is cached as a model artifact instead; the similarity row for a specific query movie is computed on demand — a single sparse-row multiply against the full matrix, sub-millisecond and well within the 2-second budget in NFR-1 (`04_Software_Requirements_Specification.md`). This approach still satisfies the offline precomputation principle in `06_System_Architecture.md` §7: the expensive operation (vectorising ~9,742 documents) is performed once during the build phase; only the lightweight O(n) similarity lookup occurs per request.

---

### 3. Collaborative Filtering Algorithm

**Goal:** Given a user's rating history, predict how they would rate movies they haven't seen, using patterns from *all* users.

**Approach: Matrix Factorization via SVD**

> **Implementation note:** This is implemented using `scipy.sparse.linalg.svds` directly (truncated SVD on the bias-adjusted, sparse user-item matrix), rather than a higher-level library wrapper such as `surprise`. This is the same underlying technique popularized as "Funk SVD" during the Netflix Prize — the conceptual algorithm below is unchanged; only the implementation library differs. See `09_Development_Guide.md` for the exact dependency list.

1. Construct the user-item ratings matrix `R` (rows = users, columns = movies, cell = rating or missing).
2. Compute baseline predictors to remove systematic bias before factorization:

```
baseline(u, i) = global_mean + user_bias(u) + item_bias(i)
```

3. Build the **residual matrix** `R' = R - baseline` (only at observed entries; missing entries treated as 0 residual, i.e. "no signal beyond baseline").
4. Decompose the sparse residual matrix into latent factor matrices via truncated SVD:

```
R' ≈ U × Σ × Vᵀ
```

Where:
- `U` = user latent factor matrix (each user represented as a small vector of "taste dimensions")
- `V` = item latent factor matrix (each movie represented as a small vector of "appeal dimensions")
- `Σ` = diagonal matrix of singular values (relative importance of each latent dimension)

5. Predicted rating for user *u* and movie *i* reconstructs the baseline plus the learned residual:

```
r̂(u, i) = baseline(u, i) + (U_u · Σ · V_iᵀ)
```

Predictions are clipped to the valid rating range [0.5, 5.0].

6. For a target user, predict scores for all unseen movies, sort descending, return the Top-K.

**Why SVD over simple user-based/item-based kNN CF:** Matrix factorization scales better, generalizes better under sparsity, and captures latent taste dimensions that aren't explicit in raw metadata (e.g., "quirky indie humor") — the classic strength that made this approach famous from the Netflix Prize era.

---

### 4. Hybrid Scoring Logic

**Goal:** Combine both signals into one ranked list, adapting automatically to how much is known about the user.

**Step 1 — Normalize both scores to a common [0, 1] range:**

```
content_score_norm  = (content_score - min) / (max - min)
predicted_rating_norm = predicted_rating / 5.0
```

**Step 2 — Determine α (collaborative weight) based on user rating count:**

| User rating count | α (CF weight) | (1 - α) (CBF weight) |
|---|---|---|
| 0 (new user) | 0.0 | 1.0 |
| 1–4 (very sparse) | 0.2 | 0.8 |
| 5–19 | 0.5 | 0.5 |
| 20+ (established) | 0.8 | 0.2 |

**Step 3 — Compute final hybrid score:**

```
final_score = α * predicted_rating_norm + (1 - α) * content_score_norm
```

**Handling a missing collaborative signal for an individual candidate:** A candidate movie may have too few ratings in the dataset to appear in the trained collaborative model at all (e.g., 1-2 total ratings). In this case `predicted_rating_norm` cannot be computed from the model. It is set to a **neutral prior** — the global mean rating normalized to [0, 1] — rather than substituting the content score. Reusing the content score here would double-count content similarity into both terms of the blend and let obscure, near-unrated movies rank above movies with a genuine strong collaborative prediction, which defeats the purpose of a high α for established users.

**Step 4 — Rank and return Top-K movies by `final_score`, attaching a reason tag:**
- If α ≥ 0.5 → tag as *"Recommended based on similar users' ratings"*
- If α < 0.5 → tag as *"Recommended based on movies you've liked"*

---

### 5. Cold-Start Handling (Decision Rule)

```
IF user has 0 ratings AND no reference movie provided:
    → return Trending/Popularity-based list (avg rating × log(rating count))

ELSE IF user has 0 ratings BUT a reference movie is provided:
    → return pure Content-Based results (α = 0)

ELSE IF user has ratings:
    → run Hybrid Scoring with α determined by rating count table above
```

This decision tree ensures the system **never returns an empty or failed response**, directly satisfying NFR-5 (Reliability) from the SRS.

---

### 6. Popularity Fallback Score (Trending)

Used when there is no personalization signal at all:

```
popularity_score = weighted_rating = (v / (v + m)) * R + (m / (v + m)) * C
```

Where:
- `R` = average rating of the movie
- `v` = number of ratings for the movie
- `m` = minimum ratings threshold required to be considered (e.g., 50th percentile of rating counts)
- `C` = mean rating across the entire dataset

This is the **IMDB weighted rating formula**, which prevents movies with very few ratings (but a lucky 5.0 average) from dominating the trending list.

---

### 7. Evaluation Metrics for These Algorithms

| Metric | Applies To | Purpose |
|---|---|---|
| RMSE | Collaborative Filtering | Measures rating prediction error on held-out test ratings |
| Precision@K | Both / Hybrid | Of the K recommended movies, how many were actually relevant (highly rated) in held-out data |
| Recall@K | Both / Hybrid | Of all relevant movies for a user, how many appeared in the Top-K recommendations |
| Coverage | Content-Based | What fraction of the catalog is ever recommended, to check for over-concentration on popular items |

Full results and comparison tables are captured in the model training notebook and referenced in the Development Guide.

---

### 8. Empirical Results (measured on ml-latest-small)

Computed by `src/evaluate.py` on an 80/20 per-user train/test split (random_state=42; users with <5 ratings excluded from the split, matching `MIN_RATINGS_THRESHOLD`). Precision@10/Recall@10 use a candidate-restricted leave-out methodology: for each test user, only their own held-out test movies are ranked (a standard simplified offline evaluation), with "relevant" defined as an actual rating ≥ 4.0.

**Reproducibility note:** The content-based and hybrid seed movie for each user (their highest-rated training movie) is selected using a stable sort on `(rating descending, movie_id ascending)`. This guarantees deterministic tie-breaking when a user has multiple movies tied at their maximum rating, so these figures reproduce identically across machines, operating systems, and pandas/numpy versions.

| Strategy | RMSE | Precision@10 | Recall@10 | Coverage |
|---|---|---|---|---|
| Content-based only | n/a | 0.5731 | 0.6371 | 0.5898 (2,086 / 3,537 content-eligible movies) |
| Collaborative only | 0.8895 | 0.6493 | 0.6834 | n/a |
| **Hybrid** | n/a | 0.6352 | 0.6772 | n/a |

**Interpretation:** Hybrid does **not** beat pure collaborative filtering on this metric — it performs slightly below it, and above content-based alone. This is expected behaviour, not a flaw in the hybrid design: Precision@K/Recall@K here can only be computed for users who have a held-out test set, i.e. users with sufficient rating history — precisely the population where collaborative filtering is already at its strongest. The metric structurally cannot capture hybrid's actual value proposition, which is **coverage across the user lifecycle**: collaborative filtering returns nothing for a zero-rating user (no row exists in the trained matrix), while the hybrid system always returns a ranked, explainable recommendation list by falling back to content similarity or popularity (see §5, Cold-Start Handling). That robustness property is a coverage and availability guarantee, not a ranking-quality metric, and is not reflected in the table above.

RMSE of 0.8895 on a 0.5–5.0 scale is consistent with published MovieLens SVD benchmarks (typically ~0.85–0.95), indicating the collaborative model is well-calibrated.
