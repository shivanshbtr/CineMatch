# 08. API Specification
## CineMatch — REST API Reference

---

### 1. Base URL

```
Local:      http://localhost:8000
Production: https://cinematch-api.<deployment-domain>.com
```

### 2. Authentication

This version of the API is unauthenticated (public demo scope). All requests are open. Future scope (see SRS §9) includes API-key based access for production use.

---

### 3. Common Response Envelope

All endpoints return a consistent JSON structure. `movie_id` is always the MovieLens `movieId` (the system-wide surrogate key defined in `05_Database_Design_and_ER_Diagram.md` §2.1) — not the TMDB `id`:

```json
{
  "status": "success",
  "strategy_used": "hybrid",
  "count": 10,
  "results": [
    {
      "movie_id": 1721,
      "title": "Titanic",
      "genres": ["Drama", "Romance"],
      "score": 0.87,
      "reason": "Recommended based on similar users' ratings"
    }
  ]
}
```

Error responses:

```json
{
  "status": "error",
  "error_code": "MOVIE_NOT_FOUND",
  "message": "No movie found matching the given title."
}
```

---

### 4. Endpoints

#### 4.1 `GET /recommend/content`

Returns content-based similar movies for a given movie title.

**Query Parameters**

| Param | Type | Required | Description |
|---|---|---|---|
| `title` | string | Yes | Exact/selected movie title |
| `top_k` | integer | No (default 10) | Number of results (5–20) |

**Sample Request**
```
GET /recommend/content?title=Inception&top_k=10
```

**Sample Response** — see §3 envelope, `strategy_used: "content"`

**Error Codes:** `MOVIE_NOT_FOUND` (404), `INVALID_TOP_K` (400)

---

#### 4.2 `GET /recommend/collaborative`

Returns collaborative-filtering-based recommendations for a given user.

**Query Parameters**

| Param | Type | Required | Description |
|---|---|---|---|
| `user_id` | integer | Yes | Valid user ID from the dataset |
| `top_k` | integer | No (default 10) | Number of results (5–20) |

**Sample Request**
```
GET /recommend/collaborative?user_id=42&top_k=10
```

**Error Codes:** `USER_NOT_FOUND` (404), `INSUFFICIENT_HISTORY` (returns fallback, not an error — see §5)

---

#### 4.3 `GET /recommend/hybrid`

Returns hybrid recommendations, automatically applying cold-start logic.

**Query Parameters**

| Param | Type | Required | Description |
|---|---|---|---|
| `user_id` | integer | No | User ID, if known |
| `reference_movie` | string | No | Movie title, used for CBF component / cold-start fallback |
| `top_k` | integer | No (default 10) | Number of results (5–20) |

*At least one of `user_id` or `reference_movie` must be provided.*

**Sample Request**
```
GET /recommend/hybrid?user_id=7&reference_movie=Interstellar&top_k=10
```

**Error Codes:** `MISSING_INPUT` (400) — neither `user_id` nor `reference_movie` supplied

---

#### 4.4 `GET /trending`

Returns the current popularity-based fallback list. No parameters required.

**Sample Request**
```
GET /trending?top_k=10
```

---

#### 4.5 `GET /health`

Simple liveness/readiness check for deployment monitoring.

**Sample Response**
```json
{ "status": "ok", "models_loaded": true }
```

---

### 5. Business Logic Notes Reflected in the API

- `/recommend/collaborative` never returns a hard error for a valid-but-sparse user — it silently degrades to the fallback path described in the Algorithms doc (§5, Cold-Start Handling) and reports `strategy_used: "fallback_popularity"` or `strategy_used: "fallback_content"` accordingly, so the client always knows what was actually served.
- `top_k` is server-side clamped to [5, 20] even if a client requests outside that range, per SRS Business Rules.

---

### 6. Validation Rules

| Rule | Enforcement |
|---|---|
| `title` must match an existing movie (case-insensitive) | 404 `MOVIE_NOT_FOUND` if not |
| `user_id` must be a positive integer present in the dataset | 404 `USER_NOT_FOUND` if not |
| `top_k` must be an integer between 5 and 20 | Clamped, not rejected, for a smoother UX |
| At least one identifying parameter required on `/recommend/hybrid` | 400 `MISSING_INPUT` if neither given |

---

### 7. Rate Limiting & Error Handling

- All endpoints wrapped in centralized FastAPI exception handlers to guarantee the standard error envelope (§3) is always returned, never a raw stack trace.
- Rate limiting is out of scope for this project's demo deployment but is noted as future scope in the SRS.
