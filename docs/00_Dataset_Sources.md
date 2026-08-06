# 00. Dataset Sources
## CineMatch — Data Provenance & Licensing

---

### 1. Purpose

This document records exactly which datasets CineMatch uses, where they come from, how they're documented upstream, and how they map to each part of the system. This exists so data provenance is never ambiguous during review.

---

### 2. Dataset 1 — MovieLens (ml-latest-small)

| Field | Detail |
|---|---|
| **Publisher** | GroupLens Research, University of Minnesota |
| **Official page** | https://grouplens.org/datasets/movielens/latest/ |
| **Direct download** | https://files.grouplens.org/datasets/movielens/ml-latest-small.zip |
| **Official README** | https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html |
| **Size** | ~1 MB zipped |
| **Contents** | 100,000 ratings, 3,600 tag applications, 9,000 movies, 600 users |
| **Last updated (upstream)** | September 2018 |
| **License / usage terms** | Free for research and education under GroupLens' usage terms (non-commercial, attribution required — see README) |
| **Used for** | Collaborative Filtering (user-item ratings matrix, SVD training/evaluation) |

**Files used:**

| File | Columns | Used In |
|---|---|---|
| `ratings.csv` | `userId`, `movieId`, `rating`, `timestamp` | Collaborative filtering training/evaluation |
| `movies.csv` | `movieId`, `title`, `genres` | Joined into the master `Movies` table; genre feature for content-based model |
| `links.csv` | `movieId`, `imdbId`, `tmdbId` | Used to join MovieLens movies to their corresponding TMDB records |
| `tags.csv` | `userId`, `movieId`, `tag`, `timestamp` | Optional supplementary keyword signal (not required for baseline model) |

**Note on dataset category:** This is a "Latest" (rolling) MovieLens dataset rather than a numbered "stable benchmark" release (e.g., ml-100k, ml-1m, ml-32m). GroupLens explicitly states the "Latest" datasets are not intended for citation in published research results, since their contents can be regenerated over time. This does not affect CineMatch's use case — it is an applied system project, not a research benchmark study, so a stable citation snapshot is not required. The file schema is identical to the numbered releases.

---

### 3. Dataset 2 — TMDB 5000 Movie Dataset

| Field | Detail |
|---|---|
| **Publisher** | The Movie Database (TMDB), redistributed via Kaggle |
| **Kaggle page** | https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata |
| **Contents** | ~5,000 movies with metadata + credits |
| **License / usage terms** | Kaggle dataset license (see Kaggle page); underlying data sourced from TMDB's public API/database |
| **Used for** | Content-Based Filtering (overview, keywords, cast, crew → "metadata soup") |

**Files used:**

| File | Columns | Used In |
|---|---|---|
| `tmdb_5000_movies.csv` | `id`, `title`, `overview`, `genres`, `keywords`, `release_date`, `vote_average`, `vote_count` | Overview/keyword text features; popularity/trending fallback scoring |
| `tmdb_5000_credits.csv` | `movie_id`, `title`, `cast`, `crew` | Cast (top-N actors) and director extraction for the metadata soup |

---

### 4. Why Two Datasets Instead of One

MovieLens provides high-quality **user-rating behavior** but minimal item metadata (title + genre only — no synopsis, cast, or keywords). TMDB provides rich **item metadata** but no user-rating history in the form needed for collaborative filtering. CineMatch's hybrid architecture (see `07_Algorithms_and_Scoring_Logic.md`) requires both:

- **MovieLens → Collaborative Filtering** (needs many users × many ratings)
- **TMDB → Content-Based Filtering** (needs rich per-movie metadata)

---

### 5. Joining Strategy

1. Load `movies.csv` and `links.csv` from MovieLens; extract `tmdbId` for each `movieId`.
2. Load `tmdb_5000_movies.csv`; join on `tmdbId == id`.
3. Resulting merged table becomes the master `Movies` entity (see `05_Database_Design_and_ER_Diagram.md`) — MovieLens supplies `movieId`/genre/rating linkage, TMDB supplies overview/keywords/cast.
4. Movies present in one dataset but not the other (unmatched joins) are logged and excluded from the content-based corpus, but retained for collaborative filtering if ratings exist — this is the practical reason `010_Data_Preprocessing` explicitly handles partial-metadata rows rather than silently dropping them (see Development Guide).

---

### 6. Data Placement in the Repository

```
data/raw/
├── ml-latest-small/
│   ├── ratings.csv
│   ├── movies.csv
│   ├── links.csv
│   └── tags.csv
└── tmdb-5000/
    ├── tmdb_5000_movies.csv
    └── tmdb_5000_credits.csv
```

Raw files are never committed with large binary changes tracked; `data/raw/` is included in `.gitignore` in practice, with this document serving as the reproducible source of truth for what to re-download.

---

### 7. Attribution

Per GroupLens' citation request, any report/paper using this dataset should cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.

TMDB data is used under Kaggle's redistribution terms and is credited to The Movie Database (TMDB) as the original data source.
