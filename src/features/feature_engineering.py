import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import time

ENGINE = create_engine(
    "postgresql://postgres:postgres@localhost:5432/recsys"
)

def load_interactions():
    query = "SELECT user_id, item_id, rating, timestamp FROM interactions"
    return pd.read_sql(query, ENGINE)

def load_items():
    query = "SELECT item_id, title, genres FROM items"
    return pd.read_sql(query, ENGINE)

def build_user_features(interactions: pd.DataFrame) -> pd.DataFrame:
    user_features = interactions.groupby("user_id").agg(
        interaction_count=("item_id", "count"),
        avg_rating=("rating", "mean"),
        last_interaction=("timestamp", "max")
    ).reset_index()

    return user_features

def build_item_popularity(interactions: pd.DataFrame) -> pd.DataFrame:
    popularity = interactions.groupby("item_id").size().reset_index(name="popularity")
    return popularity


def add_freshness(interactions: pd.DataFrame) -> pd.DataFrame:
    now = int(time.time())
    interactions["freshness"] = np.exp(-(now - interactions["timestamp"]) / (60 * 60 * 24 * 30))
    return interactions

def tokenize_genres(items: pd.DataFrame) -> pd.DataFrame:
    items["genres"] = items["genres"].str.replace("|", " ", regex=False)
    return items

if __name__ == "__main__":
    interactions = load_interactions()
    items = load_items()

    interactions = add_freshness(interactions)

    user_features = build_user_features(interactions)
    item_popularity = build_item_popularity(interactions)
    items = tokenize_genres(items)

    print(user_features.head())
    print(item_popularity.head())
    print(items.head())
