from src.features.feature_engineering import load_items
from src.models.content_based import ContentBasedRecommender

if __name__ == "__main__":
    items = load_items()

    model = ContentBasedRecommender()
    model.fit(items)

    # Simulate a new user who liked item 1 and 2
    print("Content-based recommendations:")
    print(model.recommend(liked_item_ids=[1, 2], k=10))
