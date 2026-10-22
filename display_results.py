
import pandas as pd
import streamlit as st

def display_recommendations(recommendations):
    """
    Displays the recommended movies in a Streamlit app.
    """

    st.subheader("Dina rekommendationer: ")

    for position, row in enumerate(recommendations.itertuples(), start=1):
        match = round(row.cosine_score * 100)
        st.write(f"{position}. {row.title}: {match}% match")

if __name__ == "__main__":
    # Fake data for testing
    fake = pd.DataFrame({
        "id": [1, 2, 3],
        "title": ["Inception", "Up", "Jumanji"],
        "cosine_score": [0.95, 0.68, 0.33],
    })
    display_recommendations(fake)