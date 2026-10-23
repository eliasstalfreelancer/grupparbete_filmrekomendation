import ast
import pandas as pd
from load_data import load_data

genrer = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
          "Drama", "Family", "Fantasy", "Foreign", "History", "Horror", "Music",
          "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"]


def get_genres(text):
    try:
        lista = ast.literal_eval(text)
        if not isinstance(lista, list):
            return []
        return [g["name"] for g in lista if g["name"] in genrer]
    except Exception:
        return []


def get_keywords(text):
    try:
        lista = ast.literal_eval(text)
        if not isinstance(lista, list):
            return []
        namn = [k["name"].strip().lower() for k in lista]
        return sorted(set(namn))
    except Exception:
        return []


def clean_movies(path="Kaggle_Movie_Data/movies_metadata.csv"):
    df = load_data(path)
    print("Rader från början:", len(df))

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)
    print("Efter trasiga rader:", len(df))

    df = df[["id", "title", "genres"]].copy()

    df["title"] = df["title"].str.strip()
    df = df.dropna(subset=["title"])
    df = df[df["title"] != ""]
    print("Efter filmer utan titel:", len(df))

    df = df.drop_duplicates(subset="id").copy()
    print("Efter dubbletter:", len(df))

    df["genres"] = df["genres"].apply(get_genres)
    df = df[df["genres"].str.len() > 0]
    print("Efter filmer utan genre:", len(df))

    return df.reset_index(drop=True)


def add_keywords(df, path="Kaggle_Movie_Data/keywords.csv"):
    kw = load_data(path)
    print("Rader i keywords.csv:", len(kw))

    kw["id"] = pd.to_numeric(kw["id"], errors="coerce")
    kw = kw.dropna(subset=["id"]).copy()
    kw["id"] = kw["id"].astype(int)
    kw = kw.drop_duplicates(subset="id").copy()
    print("Keywords efter dubbletter:", len(kw))

    kw["keywords"] = kw["keywords"].apply(get_keywords)

    df = df.merge(kw[["id", "keywords"]], on="id", how="left")
    df["keywords"] = df["keywords"].apply(lambda x: x if isinstance(x, list) else [])

    print("Filmer utan keywords:", (df["keywords"].str.len() == 0).sum())
    return df


if __name__ == "__main__":
    movies = clean_movies()
    movies = add_keywords(movies)
    spara = movies.copy()
    spara["genres"] = spara["genres"].str.join("|")
    spara["keywords"] = spara["keywords"].str.join("|")
    spara.to_csv("Kaggle_Movie_Data/movies_clean.csv", index=False)
    print(movies.head())