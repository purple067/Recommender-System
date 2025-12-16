from src.features.feature_engineering import load_interactions
from src.models.collaborative_filtering import CollaborativeFiltering

if __name__ == "__main__":
    interactions = load_interactions()

    model = CollaborativeFiltering(n_factors=20)
    model.fit(interactions)

    print("Recommendations for user 1:")
    print(model.recommend(user_id=1, k=10))
