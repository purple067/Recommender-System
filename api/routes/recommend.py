from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class RecommendRequest(BaseModel):
    user_id: int
    liked_item_ids: List[int]
    k: int = 10

@router.post("/recommend")
def recommend(req: RecommendRequest):
    recommendations = recommender.recommend(
        user_id=req.user_id,
        liked_item_ids=req.liked_item_ids,
        k=req.k
    )
    return {"recommended_items": recommendations}