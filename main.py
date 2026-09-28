"""Movie recommender CLI.

Interactive:   python main.py
One-shot:      python main.py --interests Sci-Fi,Thriller --rate "Inception=5" "Toy Story=2" --top 5
Evaluate:      python main.py --evaluate
"""
import argparse
import sys

from recommender import Recommender, load_movies, load_ratings


def find_movie(movies, query: str):
    q = query.strip().lower()
    exact = [m for m in movies.values() if m.title.lower() == q]
    if exact:
        return exact[0]
    partial = [m for m in movies.values() if q in m.title.lower()]
    return partial[0] if len(partial) == 1 else None


def parse_interests(text: str, rec: Recommender) -> list[str]:
    lookup = {g.lower(): g for g in rec.genres}
    picked = []
    for part in text.split(","):
        g = lookup.get(part.strip().lower())
        if g:
            picked.append(g)
        elif part.strip():
            print(f"  (ignoring unknown genre '{part.strip()}')")
    return picked


def show(results):
    if not results:
        print("No recommendations found.")
        return
    print("\nTop picks for you:\n")
    for i, (m, score, why) in enumerate(results, 1):
        print(f"{i}. {m.title} ({m.year})  -  match {score}%")
        print(f"   {', '.join(m.genres)}")
        print(f"   Why: {why}\n")


def interactive(rec: Recommender):
    print("=== Movie Recommender ===")
    print("Genres:", ", ".join(rec.genres))
    interests = parse_interests(input("\nWhat genres are you into? (comma-separated): "), rec)

    ratings = {}
    print("\nRate movies you've seen (1-5), or press Enter to skip. Type 'done' to finish.")
    sample = sorted(rec.movies.values(), key=lambda m: -sum(m.movie_id in u for u in rec.ratings.values()))[:12]
    for m in sample:
        ans = input(f"  {m.title} ({m.year}) [1-5/Enter]: ").strip().lower()
        if ans == "done":
            break
        if ans in {"1", "2", "3", "4", "5"}:
            ratings[m.movie_id] = float(ans)

    show(rec.recommend(interests, ratings, top_n=5))


def main():
    ap = argparse.ArgumentParser(description="Simple AI movie recommender")
    ap.add_argument("--interests", default="", help="comma-separated genres, e.g. Sci-Fi,Thriller")
    ap.add_argument("--rate", nargs="*", default=[], metavar="TITLE=RATING", help='e.g. "Inception=5"')
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of content-based score (0-1)")
    ap.add_argument("--genre", help="only recommend this genre")
    ap.add_argument("--evaluate", action="store_true", help="report offline accuracy (RMSE)")
    args = ap.parse_args()

    rec = Recommender(load_movies(), load_ratings())

    if args.evaluate:
        model, base = rec.evaluate()
        print(f"RMSE  collaborative model: {model:.3f}   global-mean baseline: {base:.3f}")
        return

    if not args.interests and not args.rate:
        return interactive(rec)

    ratings = {}
    for item in args.rate:
        title, _, val = item.rpartition("=")
        movie = find_movie(rec.movies, title)
        if not movie or not val.replace(".", "", 1).isdigit():
            print(f"Could not use '{item}' (use TITLE=1-5 with a known title)", file=sys.stderr)
            continue
        ratings[movie.movie_id] = float(val)

    show(rec.recommend(parse_interests(args.interests, rec), ratings,
                       top_n=args.top, alpha=args.alpha, genre_filter=args.genre))


if __name__ == "__main__":
    main()
