import streamlit as st
import pandas as pd

@st.cache_data
def load_clean_movies():
    return pd.read_csv("Kaggle_Movie_Data/movies_clean.csv")

movies_df = load_clean_movies()
movie_titles = movies_df["title"].tolist()

def movie_autocomplete(label):
    user_input = st.text_input(label)

    if len(user_input) < 3:
        st.write("Skriv minst 3 bokstäver…")
        return None

    # Filtrera titlar baserat på input
    filtered = [title for title in movie_titles if user_input.lower() in title.lower()]

    if not filtered:
        st.write("Inga filmer matchar din sökning.")
        return None

    return st.selectbox("Välj film", filtered)

st.title("🎥 Filmrekommendation")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    movie1 = movie_autocomplete("Film 1")

with col2:
    movie2 = movie_autocomplete("Film 2")

with col3:
    movie3 = movie_autocomplete("Film 3")

with col4:
    movie3 = movie_autocomplete("Film 4")

with col5:
    movie3 = movie_autocomplete("Film 5")


if st.button("Hämta rekommendationer"):
    if not movie1 or not movie2 or not movie3:
        st.warning("Du måste välja alla tre filmer.")
    else:
        st.success(f"Du valde: {movie1}, {movie2}, {movie3}")
