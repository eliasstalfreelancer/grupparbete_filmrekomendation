import pandas as pd
import ast
import json
import os
from load_data import load_data
from collections import Counter


def parse_json_column(column_series):
    """Omvandlar strängifierad JSON till lista med namn."""
    def extract(val):
        if pd.isna(val) or not isinstance(val, str):
            return []
        
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [item['name'] for item in parsed if isinstance(item, dict) and 'name' in item]
        except Exception:
            pass

        # Försök 2: json.loads (standard-JSON)
        try:
            # Byt enkelsnuttar mot dubbelsnuttar om JSON-standarden kräver det
            clean_val = val.replace("'", '"')
            parsed = json.loads(clean_val)
            if isinstance(parsed, list):
                return [item['name'] for item in parsed if isinstance(item, dict) and 'name' in item]
        except Exception:
            pass

        return []

    return column_series.apply(extract)

#Läser in data
def load_raw_data(data_dir='Kaggle_Movie_Data'):
    """Läser in rådata från Kaggle Movie dataset"""
    print("\n Läser in data...")
    movies = load_data(os.path.join(data_dir, 'movies_metadata.csv'))
    keywords = load_data(os.path.join(data_dir, 'keywords.csv'))
    links = load_data(os.path.join(data_dir, 'links.csv'))
    ratings = load_data(os.path.join(data_dir, 'ratings.csv'))
    return movies, keywords, links, ratings

def prepare_eda_dataframe(movies, keywords):
    """Förbereder och mergar movies och keywords med parsad JSON."""
    movies['tmdbId_clean'] = pd.to_numeric(movies['id'], errors='coerce')
    keywords['id_int'] = pd.to_numeric(keywords['id'], errors='coerce')

    movies['parsed_genres'] = parse_json_column(movies['genres'])
    keywords['parsed_keywords'] = parse_json_column(keywords['keywords'])

    movies_keyw = movies.merge(
    keywords[['id_int', 'parsed_keywords']], 
    left_on='tmdbId_clean', 
    right_on='id_int', 
    how='left')
    return movies_keyw

def run_coverage_analysis(movies, links):
    """ID länkning och täckningsanalys"""
    movies['tmdbId_clean'] = pd.to_numeric(movies['id'], errors='coerce')
    links_clean = links.dropna(subset=['tmdbId']).copy()
    links_clean['tmdbId_clean'] = links_clean['tmdbId'].astype(int)

    merged_links = links_clean.merge(movies, on='tmdbId_clean', how='left')
    missing_movies = merged_links['title'].isnull().sum()

    
    print(f"Totalt antal unika filmer i links_small: {links_clean['tmdbId_clean'].nunique()}")
    print(f"Filmer i links_small som saknar matchning i movies_metadata: {missing_movies}")


def run_content_analysis(movies_keyw):
    """Analys av Genre och Keywords"""

    #Genre
    all_genres = [g for sublist in movies_keyw['parsed_genres'] for g in sublist]
    genre_counts = Counter(all_genres)
    empty_genres = (movies_keyw['parsed_genres'].apply(len) == 0).sum()

    unique_genres = []
    for g in all_genres:
        if g not in unique_genres:
            unique_genres.append(g)

    print(f"\n[GENRES]")
    print(f"Totalt antal unika genrer: {len(genre_counts)}")

    for g in unique_genres:
        print(f"  - {g}")
    print(f"Filmer som helt saknar genrer: {empty_genres}")
    print("Top 5 vanligaste genrer:")
    for g, count in genre_counts.most_common(5):
        print(f"  - {g}: {count} filmer")

    #Keywords

    keyw_lists = movies_keyw['parsed_keywords'].dropna().apply(lambda x: x if isinstance(x, list) else [])
    all_keywords = [k for sublist in keyw_lists for k in sublist]
    keyw_counts = Counter(all_keywords)
    no_keywords_count = (keyw_lists.apply(len) == 0).sum()

    print(f"\n[KEYWORDS]")
    print(f"Totalt antal unika keywords: {len(keyw_counts)}")
    print(f"Filmer som helt saknar keywords: {no_keywords_count} ({(no_keywords_count/len(movies_keyw))*100:.1f}%)")
    print("Top 5 vanligaste keywords:")
    for k, count in keyw_counts.most_common(5):
        print(f"  - {k}: {count} gånger")


def run_eval_analysis(ratings):
    """Användar analys för evaluation"""
    high_ratings = ratings[ratings['rating'] >= 4.0]
    user_high_counts = high_ratings.groupby('userId').size()
    valid_users = user_high_counts[user_high_counts >= 10]
    print(f"Totalt antal användare i ratings_small: {ratings['userId'].nunique()}")
    print(f"Användare med minst 10 filmer med rating >= 4.0: {len(valid_users)}")
    print(f"Andel godkända testanvändare: {(len(valid_users)/ratings['userId'].nunique())*100:.1f}%")


if __name__ == "__main__":
    # Läs data
    movies, keywords, links, ratings = load_raw_data()
    movies_keyw = prepare_eda_dataframe(movies, keywords)

    # Slå på/av de delar du vill köra under din review/testning:
    run_coverage_analysis(movies, links)
    run_content_analysis(movies_keyw)
    run_eval_analysis(ratings)
