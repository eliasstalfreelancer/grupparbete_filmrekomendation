
import streamlit as st
import pandas as pd
from streamlit_searchbox import st_searchbox

from baseline import baseline_recommendation
from display_results import display_recommendations

st.set_page_config(page_title="Filmrekommendation", page_icon="🎥")

@st.cache_data
def load_clean_movies():
    df = pd.read_csv("Kaggle_Movie_Data/movies_clean.csv")
    df["genre"] = df["genres"].str.split("|")

    return df

movies_df = load_clean_movies()
titles = dict(zip(movies_df["id"], movies_df["title"]))


def search_movies(search_term):
    """
    Returns movies that match the search term.
    """
    if not search_term or len(search_term) < 2:
        return []

    matches = movies_df[movies_df["title"].str.contains(search_term, case=False, na=False, regex=False)]
    return [(row.title, row.id) for row in matches.head(20).itertuples()]


st.title("🎥 Filmrekommendation")

# Search fields that shows a dropdown of matching movie titles when the user types
movie1 = st_searchbox(search_movies, key="movie1", placeholder="Sök film")
movie2 = st_searchbox(search_movies, key="movie2", placeholder="Sök film")
movie3 = st_searchbox(search_movies, key="movie3", placeholder="Sök film")
movie4 = st_searchbox(search_movies, key="movie4", placeholder="Sök film")
movie5 = st_searchbox(search_movies, key="movie5", placeholder="Sök film")


if st.button("Hämta rekommendationer"):
    selected_ids = [movie1, movie2, movie3, movie4, movie5]

    if None in selected_ids:
        st.warning("Du måste välja alla fem filmer.")
    elif len(set(selected_ids)) < 5:
        st.warning("Du har valt samma film flera gånger.")
    else:
        selected_titles = [titles[movie_id] for movie_id in selected_ids]
        st.success(f"Du valde: {', '.join(selected_titles)}")

        # Recommendations based on the selected movies
        recommendations = baseline_recommendation(movies_df[["id", "title", "genre"]], selected_ids, n=10)
        display_recommendations(recommendations)
