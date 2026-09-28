"""Generate a small synthetic ratings file (data/ratings.csv).

Each fake user has a favourite-genre profile; ratings are derived from how well
a movie matches that profile, plus some noise. Fixed seed => reproducible.
"""
import csv
import random
from pathlib import Path

DATA = Path(__file__).parent / "data"

PROFILES = {
    "scifi_fan":   {"Sci-Fi": 2, "Action": 1, "Thriller": 1, "Mystery": 1},
    "romcom_fan":  {"Romance": 2, "Comedy": 2, "Musical": 1, "Drama": 0.5},
    "family_fan":  {"Animation": 2, "Family": 2, "Adventure": 1, "Musical": 1, "Fantasy": 1},
    "crime_buff":  {"Crime": 2, "Thriller": 2, "Mystery": 1.5, "Drama": 1},
    "horror_fan":  {"Horror": 2.5, "Thriller": 1.5, "Mystery": 1},
    "action_fan":  {"Action": 2.5, "Adventure": 1.5, "Sci-Fi": 1, "Crime": 0.5},
    "drama_lover": {"Drama": 2, "Biography": 1.5, "History": 1.5, "Romance": 1},
    "fantasy_fan": {"Fantasy": 2.5, "Adventure": 2, "Family": 1, "Action": 0.5},
    "cinephile":   {"Drama": 1.5, "Mystery": 1.5, "Crime": 1, "Sci-Fi": 1, "Biography": 1},
    "comedy_fan":  {"Comedy": 2.5, "Romance": 1, "Family": 0.5},
    "thrill_seeker": {"Thriller": 2, "Action": 1.5, "Horror": 1, "Crime": 1},
    "musical_fan": {"Musical": 3, "Romance": 1, "Animation": 1, "Comedy": 0.5},
}


def main():
    rng = random.Random(42)
    with open(DATA / "movies.csv", newline="", encoding="utf-8") as f:
        movies = [(int(r["movie_id"]), r["genres"].split("|")) for r in csv.DictReader(f)]

    rows = []
    for user_id, (_, profile) in enumerate(PROFILES.items(), start=1):
        seen = rng.sample(movies, k=rng.randint(22, 32))  # each user rated a subset
        for movie_id, genres in seen:
            affinity = sum(profile.get(g, 0) for g in genres) / len(genres)
            rating = 2.2 + 1.6 * affinity + rng.gauss(0, 0.5)
            rating = min(5, max(1, round(rating)))
            rows.append((user_id, movie_id, rating))

    with open(DATA / "ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "movie_id", "rating"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} ratings for {len(PROFILES)} users.")


if __name__ == "__main__":
    main()
