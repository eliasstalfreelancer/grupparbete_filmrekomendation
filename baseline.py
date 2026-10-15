import pandas as pd

from data_conversion import genre_multi_label_binary, create_user_profile
from similarity import calculate_similarity_scores, rank_movies

def baseline_recommendation(df, selected_movies, n=10):
    """
    Generate movie recommendations based on the selected movies.
    
    df: DataFrame containing "id" and "genre" columns.
    selected_movies: List of movie ids that the user has selected.
    n: Number of recommendations to return.

    Returns:
        ranked_movies: DataFrame containing the top n recommended movies.
    """
    # Convert genres to binary features
    df, genre_columns = genre_multi_label_binary(df)

    # Create a user profile from the selected movies
    user_profile = create_user_profile(df[df["id"].isin(selected_movies)], genre_columns)

    # Calculate similarity scores between the user profile and all movies
    scores = calculate_similarity_scores(df, user_profile, genre_columns)

    # Rank the movies based on similarity scores
    ranked_movies = rank_movies(df, scores)
    ranked_movies = ranked_movies[~ranked_movies["id"].isin(selected_movies)]  # Exclude selected movies
    
    return ranked_movies.head(n)  # Return the top n recommendations

if __name__ == "__main__":

    # Example user input
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "title": ["The Matrix", "Inception", "Interstellar","Up","Jumanji","rocky"],
        "genre": ["Sci-Fi", "Sci-Fi", "Sci-Fi", "Animation", "Adventure", "Drama"]
    })

    # Generate recommendations based on selected movies
    recommended_movies = baseline_recommendation(df, [1, 3])    # Selected movies: "The Matrix" and "Interstellar"
    print(recommended_movies[["title", "cosine_score"]])

    # From the print statement, the output will be:
    """
        title  cosine_score
1  Inception           1.0
3         Up           0.0
4    Jumanji           0.0
5      rocky           0.0
"""
