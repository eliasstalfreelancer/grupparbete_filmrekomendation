import pandas as pd

import pandas as pd
import numpy as np


def genre_multi_label_binary(df):
    """
    Convert the genre column into binary genre features.

    Returns:
        df: DataFrame with genre columns.
        genre_columns: List containing the names of the genre columns.
    """

    df = df.copy()

    # Convert:
    # "Action, Sci-Fi" -> ["Action", "Sci-Fi"]
    df["genre"] = df["genre"].apply(
        lambda x: [genre.strip() for genre in x.split(",")]
    )

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
    df = pd.DataFrame({"movie": ["The Matrix", "Inception", "Interstellar","Up","Jumanji","rocky"], "year": [1999, 2010, 2014, 2009, 2019, 1976], "genre": ["Sci-Fi", "Sci-Fi", "Sci-Fi", "Animation", "Adventure", "Drama"]})
    
    #convert genre column to binary features and create user profile
    df, genre_columns = genre_multi_label_binary(df)
    user_profile = create_user_profile(df, genre_columns)
    
    print(user_profile)
    
    #from print stament, the output will be:
    """Adventure    0.166667
    Animation    0.166667
    Drama        0.166667
    Sci-Fi       0.500000
    dtype: float64"""