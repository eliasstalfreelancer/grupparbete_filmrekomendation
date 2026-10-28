import pandas as pd


def get_imdb_links(path="Kaggle_Movie_Data/links.csv"):
    links = pd.read_csv(path)

    links = links.dropna(subset=["tmdbId", "imdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype(int)
    links = links.drop_duplicates(subset="tmdbId")

    # imdbId blir ett tal när den läses in
    links["imdbId"] = links["imdbId"].astype(int).astype(str)
    links["imdbId"] = links["imdbId"].str.zfill(7)
    links["imdb_url"] = "https://www.imdb.com/title/tt" + links["imdbId"]

    return dict(zip(links["tmdbId"], links["imdb_url"]))


if __name__ == "__main__":
    imdb = get_imdb_links()
    print("Filmer med imdb-länk:", len(imdb))
    print("Toy Story:", imdb[862])