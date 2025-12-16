import pandas as pd
from src.features.feature_engineering import load_interactions, load_items
from src.models.hybrid_recommender import HybridRecommender
from src.models.collaborative_filtering import CollaborativeFiltering
from src.models.content_based import ContentBasedRecommender
from src.models.popularity_recommender import PopularityRecommender
from src.evaluation.ranking_metrics import precision_at_k, recall_at_k


def train_test_split(interactions, test_ratio=0.2):
    interactions = interactions.sort_values("timestamp")
    split = int(len(interactions) * (1 - test_ratio))
    return interactions.iloc[:split], interactions.iloc[split:]


if __name__ == "__main__":
    interactions = load_interactions()
    items = load_items()

    train, test = train_test_split(interactions)

    # Train models
    cf = CollaborativeFiltering(n_factors=20)
    cf.fit(train)

    content = ContentBasedRecommender()
    content.fit(items)

    pop = PopularityRecommender()
    pop.fit(train)

    hybrid = HybridRecommender(cf, content, pop)

    precisions, recalls = [], []

    for user_id, group in test.groupby("user_id"):
        relevant_items = group["item_id"].tolist()

        try:
            recommended = hybrid.recommend(user_id=user_id, k=10)
        except:
            continue

        precisions.append(precision_at_k(recommended, relevant_items, 10))
        recalls.append(recall_at_k(recommended, relevant_items, 10))

    print("Precision@10:", sum(precisions) / len(precisions))
    print("Recall@10:", sum(recalls) / len(recalls))
