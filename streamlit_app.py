
import streamlit as st
import pandas as pd

@st.cache_data
def load_clean_movies():
    df = pd.read_csv("Kaggle_Movie_Data/movies_clean.csv")
    df["genre"] = df["genres"].str.split("|")

    return df

movies_df = load_clean_movies()
titles = dict(zip(movies_df["id"], movies_df["title"]))

def movie_autocomplete(label, key):
    user_input = st.text_input(label, key=f"{key}_search")

    if len(user_input) < 3:
        st.write("Skriv minst 3 bokstäver…")
        return None

    # Filters titles based on user input
    filtered = movies_df[movies_df["title"].str.contains(user_input, case=False, na=False, regex=False)]

    if filtered.empty:
        st.write("Inga filmer matchar din sökning.")
        return None

    elif len(set(selected_ids)) < 5:
        st.warning("Du har valt samma film flera gånger.")

    return st.selectbox("Välj film", filtered["id"], format_func=lambda movie_id: titles[movie_id], key=f"{key}_select")

st.title("🎥 Filmrekommendation")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    movie1 = movie_autocomplete("Film 1", "movie1")

with col2:
    movie2 = movie_autocomplete("Film 2", "movie2")

with col3:
    movie3 = movie_autocomplete("Film 3", "movie3")

with col4:
    movie4 = movie_autocomplete("Film 4", "movie4")

with col5:
    movie5 = movie_autocomplete("Film 5", "movie5")


if st.button("Hämta rekommendationer"):
    selected_ids = [movie1, movie2, movie3, movie4, movie5]

    if None in selected_ids:
        st.warning("Du måste välja alla fem filmer.")
    else:
        selected_titles = [titles[movie_id] for movie_id in selected_ids]
        st.success(f"Du valde: {', '.join(selected_titles)}")

        recommendations = baseline_recommendation(movie_df[["id", "title", "genres"]], selected_ids, n=10)
        st.dataframe(recommendations[["title", "cosine_score"]])
