
import pandas as pd
import math


def hit_rate(recommended, hidden, k=10):
    """
    Calculate the hit rate in the top k recommendations.
    Returns 1 if any of the hidden movies are in the top, otherwise returns 0.
    """
    top_k = recommended[:k]

    for movie_id in top_k:
        if movie_id in hidden:
            return 1
    return 0


def recall(recommended, hidden, k=10):
    """
    Calculate the recall in the top k recommendations.
    Returns the fraction of hidden movies that are in the top k recommendations.
    """

    if len(hidden) == 0:
            return 0

    top_k = recommended[:k]
    hits = 0

    for movie_id in top_k:
        if movie_id in hidden:
            hits += 1

    return hits / len(hidden) 
            
def ndcg(recommended, hidden, k=10):
    """
    Calculate the normalized discounted cumulative gain (NDCG) in the top k recommendations.
    Returns a score between 0 and 1, where 1 is a perfect ranking.
    """
    if len(hidden) == 0:
        return 0

    top_k = recommended[:k]

    # Calculate DCG, hits higher in the ranking give a higher score
    dcg = 0 

    for i, movie_id in enumerate(top_k):
        if movie_id in hidden:
            dcg += 1 / math.log2(i + 2) 

    # Calculate the ideal DCG (IDCG), which is the best possible ranking
    idcg = 0

    for i in range(min(len(hidden), k)):
        idcg += 1 / math.log2(i + 2)

    return dcg / idcg
    



if __name__ == "__main__":
    """
    Test the evaluation metrics with example data.
    """

    # Test hit_rate
    print(hit_rate([1, 2, 3, 4, 7], [2, 5], k=5))   # Expected output: 1
    print(hit_rate([1, 2, 3, 4, 7], [2, 5], k=1))   # Expected output: 0

    # Test recall
    print(recall([1, 2, 3, 4, 7], [2, 5], k=5))  # Expected output: 0.5
    print(recall([1, 2, 3, 5, 7], [2, 5], k=5))  # Expected output: 1.0
    print(recall([1, 2, 3, 4, 7], [2, 5], k=1))  # Expected output: 0.0

    # Test ndcg 
    print(ndcg([1, 2, 3, 4, 7], [2, 5], k=5))    # Expected output: ~0.3868
    print(ndcg([2, 3, 4, 5, 7], [2, 5], k=5))    # Expected output: ~0.8772
    print(ndcg([2, 5, 3, 4, 7], [2, 5], k=5))    # Expected output: 1.0
    print(ndcg([1, 2, 3, 4, 7], [2, 5], k=1))    # Expected output: 0.0
