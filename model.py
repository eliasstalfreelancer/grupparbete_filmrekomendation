import pandas as pd

from data_conversion import (
    create_main_feature_matrix,
    create_feature_user_profile
)

from similarity import (
    calculate_feature_similarity_scores,
    rank_movies
)


def build_main_model(
    df,
    genre_weight=1.0,
    keyword_weight=1.0,
    min_keyword_df=1
):
    """
    Build the main content-based recommendation model.

    The model represents every movie using:

        Genre binary features
        +
        TF-IDF keyword features

    Parameters:
        df:
            DataFrame containing:
            - id
            - title
            - genre
            - keywords

        genre_weight:
            Weight applied to genre features.

        keyword_weight:
            Weight applied to keyword features.

        min_keyword_df:
            Minimum number of movies a keyword must
            appear in to be kept by TF-IDF.

    Returns:
        model:
            Dictionary containing everything required
            to generate recommendations.
    """

    required_columns = {
        "id",
        "title",
        "genre",
        "keywords"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Reset the index because matrix row numbers
    # must match DataFrame row numbers.
    movies = df.copy().reset_index(drop=True)

    # Each movie ID should represent exactly one movie.
    duplicated_ids = movies[
        movies["id"].duplicated(
            keep=False
        )
    ]

    if not duplicated_ids.empty:
        duplicate_values = (
            duplicated_ids["id"]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Duplicate movie IDs found. "
            "Clean the dataset before building the model. "
            f"Example duplicate IDs: "
            f"{duplicate_values[:10]}"
        )

    # Create:
    #
    # genre binary matrix
    #       +
    # keyword TF-IDF matrix
    #
    # and combine them.
    (
        movie_matrix,
        genre_columns,
        keyword_vectorizer
    ) = create_main_feature_matrix(
        movies,
        genre_weight=genre_weight,
        keyword_weight=keyword_weight,
        min_keyword_df=min_keyword_df
    )

    model = {
        "movies": movies,
        "movie_matrix": movie_matrix,

        "genre_columns": genre_columns,

        "keyword_vectorizer": keyword_vectorizer,

        "genre_weight": genre_weight,
        "keyword_weight": keyword_weight
    }

    return model


def get_selected_movie_indices(
    movies,
    selected_movies
):
    """
    Convert selected movie IDs into DataFrame/matrix
    row indices.

    Example:

        selected_movies = [1, 3, 7]

    becomes something like:

        [0, 2, 6]
    """

    if not selected_movies:
        raise ValueError(
            "At least one movie must be selected."
        )

    # Remove duplicated selections while keeping order.
    selected_movies = list(
        dict.fromkeys(selected_movies)
    )

    selected_mask = movies[
        "id"
    ].isin(selected_movies)

    found_movies = movies.loc[
        selected_mask,
        "id"
    ].tolist()

    found_movie_set = set(
        found_movies
    )

    missing_movies = [
        movie_id
        for movie_id in selected_movies
        if movie_id not in found_movie_set
    ]

    if missing_movies:
        raise ValueError(
            "The following selected movie IDs "
            "were not found in the model: "
            f"{missing_movies}"
        )

    selected_indices = movies.index[
        selected_mask
    ].to_numpy()

    return selected_indices


def recommend_movies(
    model,
    selected_movies,
    n=10
):
    """
    Generate recommendations using the main model.

    Steps:

        1. Find selected movie rows.
        2. Average their feature vectors.
        3. Create user profile.
        4. Calculate cosine similarity.
        5. Remove already selected movies.
        6. Return the top N movies.

    Parameters:
        model:
            Model created with build_main_model().

        selected_movies:
            List of movie IDs selected by the user.

        n:
            Number of recommendations.

    Returns:
        DataFrame containing the top recommendations.
    """

    if n <= 0:
        raise ValueError(
            "n must be greater than 0."
        )

    movies = model[
        "movies"
    ]

    movie_matrix = model[
        "movie_matrix"
    ]

    # --------------------------------------------------------
    # FIND THE USER'S SELECTED MOVIES
    # --------------------------------------------------------

    selected_indices = (
        get_selected_movie_indices(
            movies,
            selected_movies
        )
    )

    # --------------------------------------------------------
    # CREATE USER PROFILE
    # --------------------------------------------------------

    user_profile = (
        create_feature_user_profile(
            movie_matrix,
            selected_indices
        )
    )

    # --------------------------------------------------------
    # CALCULATE COSINE SIMILARITY
    # --------------------------------------------------------

    scores = (
        calculate_feature_similarity_scores(
            movie_matrix,
            user_profile
        )
    )

    # --------------------------------------------------------
    # RANK MOVIES
    # --------------------------------------------------------

    ranked_movies = rank_movies(
        movies,
        scores
    )

    # Do not recommend movies the user
    # already selected.
    ranked_movies = ranked_movies[
        ~ranked_movies[
            "id"
        ].isin(selected_movies)
    ]

    # Return top N recommendations.
    ranked_movies = (
        ranked_movies
        .head(n)
        .reset_index(drop=True)
    )

    return ranked_movies


def main_model_recommendation(
    df,
    selected_movies,
    n=10,
    genre_weight=1.0,
    keyword_weight=1.0,
    min_keyword_df=1
):
    """
    Convenience function.

    Builds the model and immediately generates
    recommendations.

    Useful for testing.

    For Streamlit, build_main_model() should normally
    be cached instead of rebuilding the TF-IDF matrix
    every time the user clicks Recommend.
    """

    model = build_main_model(
        df,
        genre_weight=genre_weight,
        keyword_weight=keyword_weight,
        min_keyword_df=min_keyword_df
    )

    recommendations = recommend_movies(
        model,
        selected_movies,
        n=n
    )

    return recommendations


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    # Example dataset.
    #
    # Keywords are intentionally written in approximately
    # the same format as keywords.csv from The Movies Dataset.
    df = pd.DataFrame(
        {
            "id": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10
            ],

            "title": [
                "The Matrix",
                "Inception",
                "Interstellar",
                "Up",
                "Jumanji",
                "Rocky",
                "Blade Runner 2049",
                "Arrival",
                "Ex Machina",
                "Toy Story"
            ],

            "genre": [
                "Sci-Fi, Action",
                "Sci-Fi, Thriller",
                "Sci-Fi, Drama",
                "Animation, Family",
                "Adventure, Fantasy",
                "Drama, Sport",
                "Sci-Fi, Drama",
                "Sci-Fi, Drama",
                "Sci-Fi, Drama",
                "Animation, Family"
            ],

            "keywords": [

                # The Matrix
                """
                [
                    {'id': 1, 'name': 'artificial intelligence'},
                    {'id': 2, 'name': 'virtual reality'},
                    {'id': 3, 'name': 'future'}
                ]
                """,

                # Inception
                """
                [
                    {'id': 4, 'name': 'dream'},
                    {'id': 5, 'name': 'subconscious'},
                    {'id': 3, 'name': 'future'}
                ]
                """,

                # Interstellar
                """
                [
                    {'id': 6, 'name': 'space'},
                    {'id': 7, 'name': 'time travel'},
                    {'id': 8, 'name': 'astronaut'}
                ]
                """,

                # Up
                """
                [
                    {'id': 9, 'name': 'old man'},
                    {'id': 10, 'name': 'balloon'},
                    {'id': 11, 'name': 'friendship'}
                ]
                """,

                # Jumanji
                """
                [
                    {'id': 12, 'name': 'game'},
                    {'id': 13, 'name': 'jungle'},
                    {'id': 14, 'name': 'adventure'}
                ]
                """,

                # Rocky
                """
                [
                    {'id': 15, 'name': 'boxing'},
                    {'id': 16, 'name': 'sports'},
                    {'id': 17, 'name': 'underdog'}
                ]
                """,

                # Blade Runner 2049
                """
                [
                    {'id': 1, 'name': 'artificial intelligence'},
                    {'id': 3, 'name': 'future'},
                    {'id': 18, 'name': 'dystopia'}
                ]
                """,

                # Arrival
                """
                [
                    {'id': 19, 'name': 'alien'},
                    {'id': 20, 'name': 'language'},
                    {'id': 6, 'name': 'space'}
                ]
                """,

                # Ex Machina
                """
                [
                    {'id': 1, 'name': 'artificial intelligence'},
                    {'id': 21, 'name': 'robot'},
                    {'id': 22, 'name': 'technology'}
                ]
                """,

                # Toy Story
                """
                [
                    {'id': 23, 'name': 'toy'},
                    {'id': 11, 'name': 'friendship'},
                    {'id': 24, 'name': 'child'}
                ]
                """
            ]
        }
    )

    # --------------------------------------------------------
    # BUILD MODEL ONCE
    # --------------------------------------------------------

    model = build_main_model(
        df
    )
    print(df)
    print(
        "Movie matrix shape:",
        model["movie_matrix"].shape
    )

    print(
        "Number of genre features:",
        len(
            model["genre_columns"]
        )
    )

    print(
        "Number of keyword features:",
        len(
            model[
                "keyword_vectorizer"
            ].get_feature_names_out()
        )
    )

    print(
        "\nKeyword features:"
    )

    print(
        model[
            "keyword_vectorizer"
        ].get_feature_names_out()
    )

    # --------------------------------------------------------
    # EXAMPLE USER INPUT
    # --------------------------------------------------------

    selected_movies = [
        1,  # The Matrix
        3   # Interstellar
    ]

    recommendations = recommend_movies(
        model,
        selected_movies,
        n=5
    )

    print(
        "\nRecommendations:"
    )

    print(
        recommendations[
            [
                "title",
                "cosine_score"
            ]
        ]
    )