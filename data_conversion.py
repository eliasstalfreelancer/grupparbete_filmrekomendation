import ast

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def normalize_genres(x):
        # Normalize genres into a list, regardless of input type.
        # If input is a list, tuple, or set, remove extra spaces and empty values.
        # If input is a string, split by commas and remove extra spaces.
        # For missing or unsupported values, return an empty list.
        if isinstance(x, (list, tuple, set)):
            return [str(genre).strip() for genre in x if str(genre).strip()]

        if isinstance(x, str):
            return [genre.strip() for genre in x.split(",") if genre.strip()]

        return []

def genre_multi_label_binary(df):
    """
    Convert the genre column into binary genre features.
    
    Supports:
        - ["Sci-Fi", "Drama"]
        - "Sci-Fi"
        - "Sci-Fi, Drama"


    Returns:
        df: DataFrame with genre columns.
        genre_columns: List containing the names of the genre columns.
    """

    df = df.copy()

    # Convert:
    # "Action, Sci-Fi" -> ["Action", "Sci-Fi"]
    df["genre"] = df["genre"].apply(normalize_genres)

    # Find all unique genres.
    # sorted() is important so column order is always deterministic.
    unique_genres = sorted(
        {
            genre
            for genre_list in df["genre"]
            for genre in genre_list
        }
    )

    # Create one binary column per genre
    for genre in unique_genres:
        df[genre] = df["genre"].apply(
            lambda genres: 1 if genre in genres else 0
        )

    df.drop(columns=["genre"], inplace=True)

    return df, unique_genres


def create_user_profile(df, genre_columns):
    """
    Calculate the average genre vector for the selected movies.
    """

    user_profile = df[genre_columns].mean(axis=0)

    return user_profile

if __name__ == "__main__":
    #exampl user input
    df = pd.DataFrame({"movie": ["The Matrix", "Inception", "Interstellar","Up","Jumanji","rocky"], "year": [1999, 2010, 2014, 2009, 2019, 1976], "genre": [["Sci-Fi", "Horror"], "Sci-Fi", "Sci-Fi", "Animation", "Adventure", "Drama"]})
    
    #convert genre column to binary features and create user profile
    df, genre_columns = genre_multi_label_binary(df)
    user_profile = create_user_profile(df, genre_columns)
    
    print(user_profile)
    
    #from print stament, the output will be:
    """Adventure    0.166667
    Animation    0.166667
    Drama        0.166667
    Horror       0.166667
    Sci-Fi       0.500000
    dtype: float64"""

def _clean_keyword_name(value):
    """
    Clean one keyword name.
    """

    if value is None:
        return None

    name = str(value).strip().lower()

    if not name:
        return None

    return name


def normalize_keywords(x):
    """
    Convert different keyword formats into a clean list
    containing keyword names.

    Supports the Kaggle format:

        "[{'id': 1, 'name': 'artificial intelligence'},
          {'id': 2, 'name': 'future'}]"

    Also supports:

        ["artificial intelligence", "future"]

    and:

        "artificial intelligence, future"

    Returns:

        [
            "artificial intelligence",
            "future"
        ]
    """

    if x is None or x is pd.NA:
        return []

    if isinstance(x, (float, np.floating)) and pd.isna(x):
        return []

    parsed_value = x

    # --------------------------------------------------------
    # If keywords come directly from a CSV they will normally
    # be stored as a string.
    # --------------------------------------------------------
    if isinstance(x, str):

        text = x.strip()

        if not text:
            return []

        if text.lower() in {
            "nan",
            "none",
            "null",
            "[]"
        }:
            return []

        # Try converting a string representation of a
        # Python list/dictionary into actual Python objects.
        try:
            parsed_value = ast.literal_eval(text)

        # If it is just comma-separated text, use that instead.
        except (ValueError, SyntaxError):
            parsed_value = [
                keyword.strip()
                for keyword in text.split(",")
                if keyword.strip()
            ]

    # A single dictionary becomes a list containing
    # one dictionary.
    if isinstance(parsed_value, dict):
        parsed_value = [parsed_value]

    # A single value becomes a list.
    if not isinstance(
        parsed_value,
        (list, tuple, set)
    ):
        parsed_value = [parsed_value]

    cleaned_keywords = []
    seen_keywords = set()

    for keyword in parsed_value:

        # Kaggle format:
        # {"id": 123, "name": "artificial intelligence"}
        if isinstance(keyword, dict):
            name = _clean_keyword_name(
                keyword.get("name")
            )

        # Already cleaned list:
        # ["artificial intelligence", "future"]
        else:
            name = _clean_keyword_name(keyword)

        # Avoid duplicates inside the same movie.
        if (
            name
            and name not in seen_keywords
        ):
            seen_keywords.add(name)
            cleaned_keywords.append(name)

    return cleaned_keywords


def keyword_analyzer(document):
    """
    Analyzer used by TfidfVectorizer.

    The keywords are already tokenized, so we simply return
    the list as it is.

    This is useful because a keyword such as:

        "artificial intelligence"

    stays ONE feature instead of becoming:

        "artificial"
        "intelligence"
    """

    return document


def create_keyword_tfidf_matrix(
    df,
    keyword_column="keywords",
    min_df=1
):
    """
    Convert movie keywords into a TF-IDF feature matrix.

    Returns:
        keyword_matrix:
            Sparse matrix:
            movies x keyword features

        vectorizer:
            Fitted TfidfVectorizer.
    """

    keyword_lists = df[keyword_column].apply(
        normalize_keywords
    )

    # Make sure we actually have keywords.
    if keyword_lists.map(len).sum() == 0:
        raise ValueError(
            "No valid keywords were found in the dataset."
        )

    vectorizer = TfidfVectorizer(
        analyzer=keyword_analyzer,

        # Keep keywords that exist in at least this
        # many movies.
        min_df=min_df,

        # TF-IDF vectors are L2 normalized.
        norm="l2",

        # Saves memory compared with float64.
        dtype=np.float32
    )

    keyword_matrix = vectorizer.fit_transform(
        keyword_lists
    )

    return keyword_matrix, vectorizer


# ============================================================
# MAIN MODEL FEATURE MATRIX
# ============================================================

def create_main_feature_matrix(
    df,
    genre_weight=1.0,
    keyword_weight=1.0,
    min_keyword_df=1
):
    """
    Create the complete feature matrix used by the main model.

    Main model:

        Genre features
              +
        TF-IDF keyword features

    Each feature block is normalized separately before
    they are combined.

    Parameters:
        df:
            DataFrame containing at least:
            - genre
            - keywords

        genre_weight:
            Importance of genre features.

        keyword_weight:
            Importance of keyword features.

        min_keyword_df:
            Minimum number of movies a keyword must appear in
            to be included by TF-IDF.

    Returns:
        movie_matrix:
            Sparse feature matrix containing genre + keywords.

        genre_columns:
            Names of genre features.

        keyword_vectorizer:
            Fitted TfidfVectorizer.
    """

    if genre_weight < 0:
        raise ValueError(
            "genre_weight cannot be negative."
        )

    if keyword_weight < 0:
        raise ValueError(
            "keyword_weight cannot be negative."
        )

    if genre_weight == 0 and keyword_weight == 0:
        raise ValueError(
            "At least one feature weight must be greater than 0."
        )

    # --------------------------------------------------------
    # GENRE MATRIX
    # --------------------------------------------------------

    genre_df, genre_columns = genre_multi_label_binary(
        df[["genre"]].copy()
    )

    if not genre_columns:
        raise ValueError(
            "No genres were found in the dataset."
        )

    genre_matrix = csr_matrix(
        genre_df[genre_columns].to_numpy(
            dtype=np.float32
        )
    )

    # Normalize each movie's genre vector.
    genre_matrix = normalize(
        genre_matrix,
        norm="l2"
    )

    # Apply genre weight.
    genre_matrix = (
        genre_matrix * genre_weight
    )

    # --------------------------------------------------------
    # KEYWORD MATRIX
    # --------------------------------------------------------

    keyword_matrix, keyword_vectorizer = (
        create_keyword_tfidf_matrix(
            df,
            keyword_column="keywords",
            min_df=min_keyword_df
        )
    )

    # Apply keyword weight.
    keyword_matrix = (
        keyword_matrix * keyword_weight
    )

    # --------------------------------------------------------
    # COMBINE GENRE + KEYWORDS
    # --------------------------------------------------------

    movie_matrix = hstack(
        [
            genre_matrix,
            keyword_matrix
        ],
        format="csr",
        dtype=np.float32
    )

    # Normalize the final combined movie vectors.
    movie_matrix = normalize(
        movie_matrix,
        norm="l2"
    )

    return (
        movie_matrix,
        genre_columns,
        keyword_vectorizer
    )


def create_feature_user_profile(
    movie_matrix,
    selected_indices
):
    """
    Create a user profile from selected movie vectors.

    The user profile is the average vector of all
    selected movies.

    Works with the sparse feature matrix used by
    the main model.
    """

    if len(selected_indices) == 0:
        raise ValueError(
            "At least one movie must be selected."
        )

    selected_movie_vectors = movie_matrix[
        selected_indices
    ]

    # Average all selected movie vectors.
    user_profile = selected_movie_vectors.mean(
        axis=0
    )

    # mean() returns a dense matrix.
    # Convert it back to sparse CSR format.
    user_profile = csr_matrix(
        user_profile,
        dtype=np.float32
    )

    # Normalize user profile so cosine similarity
    # works consistently.
    user_profile = normalize(
        user_profile,
        norm="l2"
    )

    return user_profile