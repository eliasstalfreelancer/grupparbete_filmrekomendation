
import pandas as pd
import streamlit as st

def display_recommendations(recommendations):
    """
    Displays the recommended movies in a Streamlit app.
    """

    st.subheader("Dina top 10 rekommendationer")

    for position, row in enumerate(recommendations.itertuples(), start=1):
        match = round(row.cosine_score * 100)

        col1, col2 = st.columns([4, 1])
        col1.markdown(f"**{position}. {row.title}**")
        col2.caption(f"{match}% match")

if __name__ == "__main__":
    # Fake data for testing
    fake = pd.DataFrame({
        "id": [1, 2, 3],
        "title": ["Inception", "Up", "Jumanji"],
        "cosine_score": [0.95, 0.68, 0.33],
    })
    display_recommendations(fake)