
import pandas as pd

from baseline import baseline_recommendation
from evaluation import hit_rate, recall, ndcg
from model import build_main_model, recommend_movies 

def load_movies():
    """
    Loads the cleaned movie list with genres as lists and keywords
    """
    df = pd.read_csv("Kaggle_Movie_Data/movies_clean.csv")
    df["genre"] = df["genres"].str.split("|")

    keywords = pd.read_csv("Kaggle_Movie_Data/keywords.csv")
    keywords = keywords.drop_duplicates(subset="id")
    df = df.merge(keywords, on="id", how="left")
    df["keywords"] = df["keywords"].fillna("[]")

    return df.drop_duplicates(subset="id")


def load_ratings(movie_ids):
    """
    Loads ratings and connects them to movie ids
    """
    ratings = pd.read_csv("Kaggle_Movie_Data/ratings_small.csv")
    links = pd.read_csv("Kaggle_Movie_Data/links_small.csv").dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    ratings = ratings.merge(links, on="movieId")
    return ratings[ratings["tmdbId"].isin(movie_ids)]


def evaluate(movies, liked):
    """
    Gives users 5 liked movies as input and then checks id the hidden liked moviews are among the 10 recommendations
    """
    model = build_main_model(movies[["id", "title", "genre", "keywords"]])
    results = []

    # Main model onlt reccomends movies with keywords
    has_keywords = set(movies.loc[movies["keywords"] != "[]", "id"])

    for user_id, group in liked.groupby("userId"):
        user_movies = group["tmdbId"].drop_duplicates()
        if len(user_movies) < 10:
            continue

        # Shuffles movies
        user_movies = user_movies.sample(frac=1, random_state=42).tolist()
        selected = user_movies[:5]     # The 5 movies the user chooses
        hidden = user_movies[5:]       # Rest of movies are hidden

        main_recs = recommend_movies(model, selected, n=len(movies))
        main_recs = main_recs[main_recs["id"].isin(has_keywords)].head(10)

        recommendations = {
            "Baseline": baseline_recommendation(movies, selected),
            "Huvudmodell": main_recs,
        }
        for name, recs in recommendations.items():
            ids = recs["id"].tolist()
            results.append({
                "Modell": name, 
                "Hit Rate@10": hit_rate(ids, hidden),
                "Recall@10": recall(ids, hidden),
                "NDCG@10": ndcg(ids, hidden),
            })
    print("Antal testanvändare:", len(results) //2)
    return pd.DataFrame(results).groupby("Modell").mean()
            


if __name__ == "__main__":
    movies = load_movies()
    ratings = load_ratings(movies["id"])

    # Only uses movies with ratings, that we know users liked 
    movies = movies[movies["id"].isin(ratings["tmdbId"])].reset_index(drop=True)

    liked = ratings[ratings["rating"] >= 4.0]


    print("Antal filmer:", len(movies))
    print("Utan keywords:", (movies["keywords"] == "[]").sum())
    print(evaluate(movies, liked).round(3))
