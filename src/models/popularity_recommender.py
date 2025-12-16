import pandas as pd

class PopularityRecommender:
    def __init__(self):
        self.popularity_rank = None

    def fit(self, interactions: pd.DataFrame):
        """
        Compute item popularity based on interaction count.
        """
        self.popularity_rank = (
            interactions.groupby("item_id")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="score")
        )

    def recommend(self, k: int = 10):
        """
        Return top-k popular items.
        """
        return self.popularity_rank.head(k)["item_id"].tolist()
