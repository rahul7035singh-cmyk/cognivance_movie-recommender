"""A simple hybrid movie recommender (pure Python, no dependencies).

Two signals are combined:

1. Content-based  - movies are vectors of IDF-weighted genres. A user profile is
   built from the genres they like (stated interests + rated movies) and
   candidates are ranked by cosine similarity to that profile.
2. Collaborative  - user-based k-NN: find existing users whose (mean-centred)
   ratings correlate with yours and predict what they'd say about unseen movies.

final_score = alpha * content + (1 - alpha) * collaborative
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
NEUTRAL = 3.0  # a rating of 3/5 means "neutral"; above = like, below = dislike


@dataclass
class Movie:
    movie_id: int
    title: str
    year: int
    genres: list[str]


def load_movies(path: Path = DATA_DIR / "movies.csv") -> dict[int, Movie]:
    with open(path, newline="", encoding="utf-8") as f:
        return {
            int(r["movie_id"]): Movie(int(r["movie_id"]), r["title"], int(r["year"]), r["genres"].split("|"))
            for r in csv.DictReader(f)
        }


def load_ratings(path: Path = DATA_DIR / "ratings.csv") -> dict[int, dict[int, float]]:
    ratings: dict[int, dict[int, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ratings.setdefault(int(r["user_id"]), {})[int(r["movie_id"])] = float(r["rating"])
    return ratings


def _cosine(a: dict, b: dict) -> float:
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else NEUTRAL


class Recommender:
    def __init__(self, movies: dict[int, Movie], ratings: dict[int, dict[int, float]]):
        self.movies = movies
        self.ratings = ratings
        self.genres = sorted({g for m in movies.values() for g in m.genres})
        self._build_movie_vectors()

    # ---------- content-based ----------
    def _build_movie_vectors(self) -> None:
        n = len(self.movies)
        df = {g: sum(g in m.genres for m in self.movies.values()) for g in self.genres}
        self.idf = {g: math.log(n / df[g]) + 1 for g in self.genres}  # rare genres count more
        self.vectors = {
            mid: {g: self.idf[g] for g in m.genres} for mid, m in self.movies.items()
        }

    def build_profile(self, interests: list[str], my_ratings: dict[int, float],
                      interest_weight: float = 2.0) -> dict[str, float]:
        """Genre-preference vector: +weight for interests, +/- for liked/disliked movies."""
        profile: dict[str, float] = {}
        for g in interests:
            profile[g] = profile.get(g, 0) + interest_weight * self.idf.get(g, 1)
        for mid, rating in my_ratings.items():
            weight = rating - NEUTRAL
            for g, v in self.vectors[mid].items():
                profile[g] = profile.get(g, 0) + weight * v / len(self.vectors[mid])
        return profile

    def content_scores(self, profile: dict[str, float]) -> dict[int, float]:
        return {mid: _cosine(profile, vec) for mid, vec in self.vectors.items()}

    # ---------- collaborative ----------
    def _similarity(self, mine: dict[int, float], other: dict[int, float]) -> float:
        common = mine.keys() & other.keys()
        if len(common) < 2:
            return 0.0
        m_mine, m_other = _mean(mine.values()), _mean(other.values())
        a = {k: mine[k] - m_mine for k in common}
        b = {k: other[k] - m_other for k in common}
        # shrink similarity when there are few co-rated movies
        return _cosine(a, b) * min(1.0, len(common) / 5)

    def predict_rating(self, mine: dict[int, float], movie_id: int, k: int = 5,
                       exclude_user: int | None = None, damping: float = 1.0) -> float | None:
        """Predicted 1-5 rating from the k most similar users, or None if not enough data."""
        m_mine = _mean(mine.values())
        neighbours = []
        for uid, other in self.ratings.items():
            if uid == exclude_user or movie_id not in other:
                continue
            sim = self._similarity(mine, other)
            if sim > 0:
                neighbours.append((sim, other[movie_id] - _mean(other.values())))
        neighbours = sorted(neighbours, reverse=True)[:k]
        if not neighbours:
            return None
        # damping pulls weak-evidence predictions toward the user's average
        pred = m_mine + sum(s * d for s, d in neighbours) / (sum(s for s, _ in neighbours) + damping)
        return min(5.0, max(1.0, pred))

    # ---------- hybrid ----------
    def recommend(self, interests: list[str] | None = None, my_ratings: dict[int, float] | None = None,
                  top_n: int = 5, alpha: float = 0.5, genre_filter: str | None = None):
        """Return [(Movie, score 0-100, explanation)] for movies the user hasn't rated."""
        interests = interests or []
        my_ratings = my_ratings or {}
        profile = self.build_profile(interests, my_ratings)
        content = self.content_scores(profile) if profile else {}
        use_cf = len(my_ratings) >= 3

        results = []
        for mid, movie in self.movies.items():
            if mid in my_ratings:
                continue
            if genre_filter and genre_filter not in movie.genres:
                continue
            c = content.get(mid, 0.0)                      # roughly 0..1
            cf_pred = self.predict_rating(my_ratings, mid) if use_cf else None
            if cf_pred is not None:
                cf = (cf_pred - 1) / 4                     # map 1..5 -> 0..1
                score = alpha * max(c, 0) + (1 - alpha) * cf
            else:
                score = max(c, 0)
            results.append((movie, round(score * 100, 1), self._explain(movie, profile, cf_pred)))
        results.sort(key=lambda r: r[1], reverse=True)
        return results[:top_n]

    def _explain(self, movie: Movie, profile: dict[str, float], cf_pred: float | None) -> str:
        liked = [g for g in movie.genres if profile.get(g, 0) > 0]
        parts = []
        if liked:
            parts.append("matches your taste in " + ", ".join(liked))
        if cf_pred is not None:
            parts.append(f"similar users would rate it ~{cf_pred:.1f}/5")
        return "; ".join(parts) or "popular pick"

    # ---------- evaluation ----------
    def evaluate(self, holdout: float = 0.2, seed: int = 0) -> tuple[float, float]:
        """RMSE of the collaborative predictor vs. a global-mean baseline on held-out ratings."""
        import random
        rng = random.Random(seed)
        global_mean = _mean(r for u in self.ratings.values() for r in u.values())
        se_model = se_base = n = 0
        for uid, user in self.ratings.items():
            items = list(user.items())
            rng.shuffle(items)
            cut = max(1, int(len(items) * holdout))
            test, train = items[:cut], dict(items[cut:])
            for mid, actual in test:
                pred = self.predict_rating(train, mid, exclude_user=uid)
                pred = global_mean if pred is None else pred
                se_model += (pred - actual) ** 2
                se_base += (global_mean - actual) ** 2
                n += 1
        return math.sqrt(se_model / n), math.sqrt(se_base / n)
