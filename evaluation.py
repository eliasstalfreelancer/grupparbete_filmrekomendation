
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
            


if __name__ == "__main__":
    print(hit_rate([1, 2, 3, 4, 7], [2, 5], k=5))   # Expected output: 1
    print(hit_rate([1, 2, 3, 4, 7], [2, 5], k=1))   # Expected output: 0


    print(recall([1, 2, 3, 4, 7], [2, 5], k=5))  # Expected output: 0.5
    print(recall([1, 2, 3, 5, 7], [2, 5], k=5))  # Expected output: 1.0
    print(recall([1, 2, 3, 4, 7], [2, 5], k=1))  # Expected output: 0.0
    