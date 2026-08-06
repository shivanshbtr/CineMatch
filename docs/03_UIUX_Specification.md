# 03. UI/UX Specification
## CineMatch — User Experience Design

---

### 1. Design Philosophy

CineMatch's interface follows three principles:

1. **Instant clarity** — a first-time visitor should understand what to do within 5 seconds, without instructions.
2. **Minimal input, maximum output** — one search/select action should produce a full, useful set of recommendations.
3. **Explainable results** — every recommendation card shows *why* it was suggested, not just a bare title.

---

### 2. Navigation Flow

```
┌────────────────────┐
│   Landing / Home    │
│  "Find a Movie"     │
└─────────┬───────────┘
          │
          ▼
┌────────────────────────────┐
│  Mode Selector (tabs)       │
│  1. Similar Movies          │
│  2. Recommended For Me      │
│  3. Trending / Popular      │
└─────────┬────────────────────┘
          │
          ▼
┌────────────────────────────┐
│  Input Panel                 │
│  - Movie search/select       │
│    OR                        │
│  - User ID select             │
└─────────┬────────────────────┘
          │
          ▼
┌────────────────────────────┐
│  Results Grid                │
│  - Poster/title cards        │
│  - Match reason tag          │
│  - Score/confidence badge    │
└─────────┬────────────────────┘
          │
          ▼
┌────────────────────────────┐
│  Movie Detail (optional)      │
│  - Overview, genres, cast     │
│  - "More like this" button    │
└────────────────────────────┘
```

---

### 3. Screens

#### 3.1 Home / Landing Screen
- App name + one-line tagline ("Find your next favorite movie")
- Three mode tabs: **Similar Movies**, **Recommended For Me**, **Trending**
- Clean search bar with autocomplete for movie titles

#### 3.2 Similar Movies Screen (Content-Based)
- Input: dropdown/search-select of a movie title
- Output: grid of 10 similar movies
- Each card tagged: *"Because it shares genre/theme with [selected movie]"*

#### 3.3 Recommended For Me Screen (Collaborative / Hybrid)
- Input: User ID selector (dropdown, since this is a demo dataset without full auth)
- Output: grid of personalized picks
- Each card tagged: *"Users with similar taste also enjoyed this"*
- If user has < 5 ratings: banner shown — *"New user detected — showing popular & similar picks"* (cold-start fallback, transparently communicated)

#### 3.4 Trending Screen
- Shows top-rated / most-popular movies from the dataset as a static fallback, ensuring the app never shows an empty state

#### 3.5 Movie Detail View
- Poster, title, genre tags, overview, cast
- "More like this" button routes back into the Similar Movies flow

---

### 4. Component Library

| Component | Description |
|---|---|
| `MovieCard` | Poster thumbnail, title, year, star rating, match-reason tag |
| `SearchSelect` | Debounced, searchable dropdown for movie titles |
| `UserSelect` | Dropdown to pick a demo user ID |
| `ScoreBadge` | Small pill showing match confidence (%) |
| `ModeTabs` | Top-level tab navigation between the three modes |
| `EmptyState` | Friendly fallback message + trending suggestions when no result is found |
| `LoadingSkeleton` | Placeholder cards shown while the API call resolves |

---

### 5. Design System

- **Typography:** Clean sans-serif (e.g., Inter/Poppins), large title weight for movie names, muted secondary text for metadata
- **Color palette:** Dark cinematic theme — deep charcoal background, warm accent (amber/gold) for CTAs and score badges, to evoke a "movie theater" feel
- **Grid:** Responsive card grid — 4 columns desktop, 2 columns tablet, 1 column mobile
- **Iconography:** Simple line icons for genre tags and score indicators

---

### 6. Interaction Guidelines

- Every action (search, select) has a loading state — never a frozen UI
- Recommendation results always load in under 2 seconds on the demo dataset
- Error/empty states never dead-end the user — they always route back to "Trending" as a safe fallback
- Match-reason tags are mandatory on every card — this is a core trust/explainability requirement, not optional styling

---

### 7. Accessibility Notes

- All interactive elements are keyboard-navigable
- Sufficient color contrast maintained between dark background and text/badges
- Alt text provided for poster images
