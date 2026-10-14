import pandas as pd

from data_conversion import genre_multi_label_binary, create_user_profile
from similarity import calculate_similarity_scores, rank_movies

def baseline_recommendation(df, selected_movies, n=10):
    """
    Generate movie recommendations based on the selected movies.
    Returns:
        ranked_movies: DataFrame containing the top n recommended movies.
    """
    # Convert genres to binary features
    df, genre_columns = genre_multi_label_binary(df)

    # Create a user profile from the selected movies
    user_profile = create_user_profile(df[df["movie"].isin(selected_movies)], genre_columns)

    # Calculate similarity scores between the user profile and all movies
    scores = calculate_similarity_scores(df, user_profile, genre_columns)

    # Rank the movies based on similarity scores
    ranked_movies = rank_movies(df, scores)
    ranked_movies = ranked_movies[~ranked_movies["movie"].isin(selected_movies)]  # Exclude selected movies
    
    return ranked_movies.head(n)  # Return the top n recommendations

if __name__ == "__main__":

    # Example user input
    df = pd.DataFrame({
        "movie": ["The Matrix", "Inception", "Interstellar","Up","Jumanji","rocky"],
        "genre": ["Sci-Fi", "Sci-Fi", "Sci-Fi", "Animation", "Adventure", "Drama"]
    })
    
    # Generate recommendations based on selected movies
    recommended_movies = baseline_recommendation(df, ["The Matrix", "Interstellar"])
    print(recommended_movies[["movie", "cosine_score"]])

    # From the print statement, the output will be:
    """
        movie  cosine_score
1  Inception           1.0
3         Up           0.0
4    Jumanji           0.0
5      rocky           0.0
"""
