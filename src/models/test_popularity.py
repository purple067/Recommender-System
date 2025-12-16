from src.features.feature_engineering import load_interactions
from src.models.popularity_recommender import PopularityRecommender

if __name__ == "__main__":
    interactions = load_interactions()

    model = PopularityRecommender()
    model.fit(interactions)

    print("Top 10 Popular Items:")
    print(model.recommend(10))
