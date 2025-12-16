import numpy as np

class HybridRecommender:
    def __init__(self, cf_model, content_model, popularity_model,
                 w_cf=0.6, w_content=0.3, w_pop=0.1):
        self.cf = cf_model
        self.content = content_model
        self.popularity = popularity_model
        self.w_cf = w_cf
        self.w_content = w_content
        self.w_pop = w_pop

    def recommend(self, user_id=None, liked_item_ids=None, k=10):
        scores = {}

        # 1️⃣ Collaborative Filtering
        if user_id is not None:
            try:
                cf_items = self.cf.recommend(user_id, k=50)
                for rank, item in enumerate(cf_items):
                    scores[item] = scores.get(item, 0) + self.w_cf * (1 / (rank + 1))
            except:
                pass

        # 2️⃣ Content-Based
        if liked_item_ids:
            try:
                content_items = self.content.recommend(liked_item_ids, k=50)
                for rank, item in enumerate(content_items):
                    scores[item] = scores.get(item, 0) + self.w_content * (1 / (rank + 1))
            except:
                pass

        # 3️⃣ Popularity fallback
        pop_items = self.popularity.recommend(k=50)
        for rank, item in enumerate(pop_items):
            scores[item] = scores.get(item, 0) + self.w_pop * (1 / (rank + 1))

        # Sort & return
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked_items[:k]]
