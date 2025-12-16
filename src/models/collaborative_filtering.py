import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

class CollaborativeFiltering:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.model = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self.user_factors = None
        self.item_factors = None

    def _create_mappings(self, df):
        users = df["user_id"].unique()
        items = df["item_id"].unique()

        self.user_mapping = {u: i for i, u in enumerate(users)}
        self.item_mapping = {i: j for j, i in enumerate(items)}
        self.reverse_item_mapping = {j: i for i, j in self.item_mapping.items()}

    def _build_matrix(self, df):
        matrix = np.zeros((len(self.user_mapping), len(self.item_mapping)))

        for _, row in df.iterrows():
            u = self.user_mapping[row.user_id]
            i = self.item_mapping[row.item_id]
            matrix[u, i] = row.rating

        return matrix

    def fit(self, interactions: pd.DataFrame):
        self._create_mappings(interactions)
        matrix = self._build_matrix(interactions)

        self.user_factors = self.model.fit_transform(matrix)
        self.item_factors = self.model.components_.T

    def recommend(self, user_id, k=10):
        if user_id not in self.user_mapping:
            raise ValueError("User not found")

        user_idx = self.user_mapping[user_id]
        scores = np.dot(self.user_factors[user_idx], self.item_factors.T)

        top_items = np.argsort(scores)[::-1][:k]
        return [self.reverse_item_mapping[i] for i in top_items]
