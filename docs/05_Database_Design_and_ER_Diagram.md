# 05. Database Design & ER Diagram
## CineMatch — Data Model

---

### 1. Overview

CineMatch's core data comes from two sources — the **MovieLens ml-latest-small** dataset (ratings, users, genres) and the **TMDB 5000 Movie Dataset** (overview, keywords, cast/crew) — joined via `tmdbId` and modeled here as if it were a relational schema (CSV-based for this project's scope). Full dataset provenance is documented in `00_Dataset_Sources.md`. This schema is the structure that would back a production version of the system, and it is also how the data is logically organized for the ML pipeline.

---

### 2. Entities

#### 2.1 `Movies`
| Column | Type | Constraint | Description |
|---|---|---|---|
| movie_id | INTEGER | PRIMARY KEY | Internal surrogate key = MovieLens `movieId` (used as the system-wide identifier in the API and all other tables) |
| tmdb_id | INTEGER | NULLABLE, UNIQUE | TMDB `id`, sourced from MovieLens `links.csv`; NULL if no TMDB match exists (see `00_Dataset_Sources.md` §5) |
| imdb_id | VARCHAR(20) | NULLABLE | IMDB identifier, sourced from MovieLens `links.csv` |
| title | VARCHAR(255) | NOT NULL | Movie title |
| genres | TEXT | NOT NULL | Pipe/comma-separated genre list |
| overview | TEXT | NULLABLE | Plot synopsis (from TMDB; NULL for unmatched movies) |
| keywords | TEXT | NULLABLE | Associated theme/plot keywords (from TMDB; NULL for unmatched movies) |
| release_year | INTEGER | NULLABLE | Year of release |
| has_content_features | BOOLEAN | NOT NULL, DEFAULT FALSE | TRUE if this movie successfully joined to TMDB metadata; FALSE means it is collaborative-filtering-only (see Joining Strategy, `00_Dataset_Sources.md`) |

#### 2.2 `Cast_Crew`
| Column | Type | Constraint | Description |
|---|---|---|---|
| credit_id | INTEGER | PRIMARY KEY | Unique credit record ID |
| movie_id | INTEGER | FOREIGN KEY → Movies.movie_id | Associated movie |
| person_name | VARCHAR(255) | NOT NULL | Actor/director name |
| role_type | VARCHAR(50) | NOT NULL | "Actor" / "Director" |

#### 2.3 `Users`
| Column | Type | Constraint | Description |
|---|---|---|---|
| user_id | INTEGER | PRIMARY KEY | Unique user identifier |
| signup_date | DATE | NULLABLE | Demo/synthetic field |

#### 2.4 `Ratings`
| Column | Type | Constraint | Description |
|---|---|---|---|
| rating_id | INTEGER | PRIMARY KEY | Unique rating record ID |
| user_id | INTEGER | FOREIGN KEY → Users.user_id | Rating author |
| movie_id | INTEGER | FOREIGN KEY → Movies.movie_id | Rated movie |
| rating | DECIMAL(2,1) | NOT NULL, CHECK (0.5–5.0) | Rating value |
| timestamp | DATETIME | NOT NULL | When the rating was submitted |

#### 2.5 `Recommendation_Log` (application-level, for traceability)
| Column | Type | Constraint | Description |
|---|---|---|---|
| log_id | INTEGER | PRIMARY KEY | Unique log entry |
| user_id | INTEGER | FOREIGN KEY → Users.user_id, NULLABLE | Null if anonymous/content-based only |
| requested_movie_id | INTEGER | FOREIGN KEY → Movies.movie_id, NULLABLE | Reference movie for CBF requests |
| strategy_used | VARCHAR(20) | NOT NULL | One of: "content", "collaborative", "hybrid", "fallback_popularity", "fallback_content" (matches `strategy_used` values returned by the API — see `08_API_Specification.md` §5) |
| returned_movie_ids | TEXT | NOT NULL | Comma-separated list of recommended movie IDs |
| created_at | DATETIME | NOT NULL | Request timestamp |

---

### 3. Relationships

- **Users (1) → (M) Ratings** — a user can submit many ratings
- **Movies (1) → (M) Ratings** — a movie can receive many ratings
- **Movies (1) → (M) Cast_Crew** — a movie has many cast/crew records
- **Users (1) → (M) Recommendation_Log** — a user can generate many recommendation requests
- **Movies (1) → (M) Recommendation_Log** *(as requested_movie_id)* — a movie can be the reference point for many content-based requests

---

### 4. Entity-Relationship Diagram (Textual)

```
┌─────────────┐        ┌───────────────┐        ┌─────────────┐
│    Users     │        │    Ratings     │        │    Movies    │
├─────────────┤ 1    M ├───────────────┤ M    1 ├─────────────┤
│ user_id (PK) │────────│ rating_id (PK) │────────│ movie_id(PK) │
│ signup_date  │        │ user_id (FK)   │        │ tmdb_id      │
└──────┬──────┘        │ movie_id (FK)  │        │ imdb_id      │
       │                │ rating         │        │ title        │
       │ 1              │ timestamp      │        │ genres       │
       │                └───────────────┘        │ overview     │
       │                                          │ keywords     │
       │                                          │ release_year │
       │                                          │ has_content_features │
       │                                          └──────┬──────┘
       │ M                                                │ 1
┌──────┴────────────┐                                    │ M
│ Recommendation_Log │                            ┌───────┴──────┐
├────────────────────┤                            │  Cast_Crew    │
│ log_id (PK)         │                            ├──────────────┤
│ user_id (FK)         │                           │ credit_id(PK) │
│ requested_movie_id(FK)│                          │ movie_id (FK) │
│ strategy_used         │                          │ person_name   │
│ returned_movie_ids    │                          │ role_type     │
│ created_at            │                          └──────────────┘
└────────────────────┘
```

---

### 5. Normalization Notes

- Schema is normalized to 3NF: genre/keyword lists are kept denormalized as text fields *only* for simplicity in this project's CSV-based implementation; a full production schema would split these into `Genres` and `Movie_Genres` junction tables.
- `Recommendation_Log` is intentionally denormalized (`returned_movie_ids` as a delimited string) to keep logging lightweight — acceptable since this table is for traceability/debugging, not transactional integrity.

---

### 6. Implementation Note for This Project

For the scope of this project, `Movies`, `Users`, and `Ratings` are implemented as in-memory pandas DataFrames loaded from the MovieLens and TMDB CSV files (joined on `tmdbId`) rather than a live relational database, since the primary goal is to demonstrate the ML pipeline. This schema documents how the same data would be structured if persisted in a relational database (e.g., PostgreSQL) in a production deployment.
