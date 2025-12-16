import pandas as pd
from sqlalchemy import create_engine
from src.ingestion.load_raw_data import load_ratings, load_movies

ENGINE = create_engine(
    "postgresql://postgres:postgres@localhost:5432/recsys"
)

def ingest():
    ratings = load_ratings()
    movies = load_movies()

    users = pd.DataFrame({"user_id": ratings["user_id"].unique()})

    users.to_sql("users", ENGINE, if_exists="replace", index=False)
    movies.to_sql("items", ENGINE, if_exists="replace", index=False)
    ratings.to_sql("interactions", ENGINE, if_exists="replace", index=False)

if __name__ == "__main__":
    ingest()
    print("Data successfully ingested into PostgreSQL")
