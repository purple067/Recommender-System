from src.features.feature_engineering import load_interactions, load_items
from src.models.collaborative_filtering import CollaborativeFiltering
from src.models.content_based import ContentBasedRecommender
from src.models.popularity_recommender import PopularityRecommender
from src.models.hybrid_recommender import HybridRecommender

if __name__ == "__main__":
    interactions = load_interactions()
    items = load_items()

    # Train models
    cf = CollaborativeFiltering(n_factors=20)
    cf.fit(interactions)

    content = ContentBasedRecommender()
    content.fit(items)

    pop = PopularityRecommender()
    pop.fit(interactions)

    hybrid = HybridRecommender(cf, content, pop)

    print("Hybrid recommendations for known user:")
    print(hybrid.recommend(user_id=1, liked_item_ids=[1, 2], k=10))

    print("\nHybrid recommendations for cold user:")
    print(hybrid.recommend(liked_item_ids=[1, 2], k=10))

    print("\nHybrid recommendations for new user:")
    print(hybrid.recommend(k=10))
