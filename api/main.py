from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from src.features.feature_engineering import load_interactions, load_items
from src.models.collaborative_filtering import CollaborativeFiltering
from src.models.content_based import ContentBasedRecommender
from src.models.popularity_recommender import PopularityRecommender
from src.models.hybrid_recommender import HybridRecommender

app = FastAPI(title="Hybrid Recommendation API")

# Load data once (startup)
interactions = load_interactions()
items = load_items()

# Train models once
cf = CollaborativeFiltering(n_factors=20)
cf.fit(interactions)

content = ContentBasedRecommender()
content.fit(items)

pop = PopularityRecommender()
pop.fit(interactions)

hybrid = HybridRecommender(cf, content, pop)

class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    liked_item_ids: Optional[List[int]] = None
    k: int = 10

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    recommendations = hybrid.recommend(
        user_id=req.user_id,
        liked_item_ids=req.liked_item_ids,
        k=req.k
    )
    return {"recommended_items": recommendations}
