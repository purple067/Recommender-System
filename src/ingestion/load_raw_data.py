
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/ml-1m")

def load_ratings():
    return pd.read_csv(
        RAW_PATH / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

def load_movies():
    return pd.read_csv(
        RAW_PATH / "movies.dat",
        sep="::",
        engine="python",
encoding="latin-1",
        names=["item_id", "title", "genres"]
    )

if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()

    print(ratings.head())
    print(movies.head())
