
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity_scores(df, user_profile, genre_columns):
    """
    Calculate the cosine similarity scores between the user profile and each movie's genre vector.
    """

    profile = user_profile.values.reshape(1, -1)    # Reshapes user profile to 2D array with one row
    movie_vectors = df[genre_columns].values     # Gets one vector per movie

    # Compare the user profile to each movie and return the similarity scores
    return cosine_similarity(profile, movie_vectors)[0]


def calculate_feature_similarity_scores(
    movie_matrix,
    user_profile
):
    """
    Calculate cosine similarity between the main model
    user profile and all movie feature vectors.

    Both movie_matrix and user_profile may be sparse matrices.

    movie_matrix:
        movies x features

    user_profile:
        1 x features

    Returns:
        One similarity score per movie.
    """

    scores = cosine_similarity(
        user_profile,
        movie_matrix
    )[0]

    return scores

def rank_movies(df, scores, n=None):
    """
    Rank movies based on similarity scores.
    Adds the scores to a copy of the DataFrame and sorts it in 
    descending order. If n is provided, only the top n movies are returned.
    """

    results = df.copy()     # Copy the DataFrame to avoid modifying the original
    results["cosine_score"] = scores     # Add one score per movie
    results = results.sort_values("cosine_score", ascending=False)     # Highest scores first

    if n is not None:
        results = results.head(n)     # Only keeps the top n movies

    return results


if __name__ == "__main__":
    from data_conversion import genre_multi_label_binary, create_user_profile

    # Example user input
    df = pd.DataFrame({
        "movie": ["The Matrix", "Inception", "Interstellar","Up","Jumanji","rocky"],
        "genre": ["Sci-Fi", "Sci-Fi", "Sci-Fi", "Animation", "Adventure", "Drama"]
    })
    # Convert genres to binary features
    df, genre_columns = genre_multi_label_binary(df)

    #Create a user profile from selected movies
    selected = ["The Matrix", "Inception"]
    user_profile = create_user_profile(df[df["movie"].isin(selected)], genre_columns)

    # Calculate similarity scores between the user profile and all movies
    scores = calculate_similarity_scores(df, user_profile, genre_columns)
    print(scores)

    # From the print statement, the output will be:
    """[1. 1. 1. 0. 0. 0.]"""

    # Rank the movies based on similarity scores
    ranked_movies = rank_movies(df, scores)
    print(ranked_movies[["movie", "cosine_score"]])

    # From the print statement, the output will be:
    """          
          movie  cosine_score
0    The Matrix           1.0
1     Inception           1.0
2  Interstellar           1.0
3            Up           0.0
4       Jumanji           0.0
5         rocky           0.0
"""

    # Test that n works, only returning the top 3 movies
    top_movies = rank_movies(df, scores, n=3)
    print(top_movies[["movie", "cosine_score"]])    

    # From the print statement, the output will be:
    """          
          movie  cosine_score
0    The Matrix           1.0
1     Inception           1.0
2  Interstellar           1.0
"""