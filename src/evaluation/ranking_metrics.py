def precision_at_k(recommended, relevant, k):
    if not recommended:
        return 0.0
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    return len(set(recommended_k) & relevant_set) / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    return len(set(recommended_k) & relevant_set) / len(relevant_set)
