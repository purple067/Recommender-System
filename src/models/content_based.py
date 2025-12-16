import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack


class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.item_vectors = None
        self.item_ids = None
        self.item_index = {}

    def fit(self, items: pd.DataFrame):
        self.item_ids = items["item_id"].tolist()
        self.item_index = {i: idx for idx, i in enumerate(self.item_ids)}
        self.item_vectors = self.vectorizer.fit_transform(items["genres"])

    def _build_user_profile(self, liked_item_ids):
        vectors = []

        for item_id in liked_item_ids:
            if item_id in self.item_index:
                vectors.append(self.item_vectors[self.item_index[item_id]])

        if not vectors:
            return None

        return vstack(vectors).mean(axis=0).A

    def recommend(self, liked_item_ids, k=10):
        user_profile = self._build_user_profile(liked_item_ids)

        if user_profile is None:
            raise ValueError("No valid liked items provided")

        similarities = cosine_similarity(user_profile, self.item_vectors)[0]
        top_indices = similarities.argsort()[::-1][:k]

        return [self.item_ids[i] for i in top_indices]
