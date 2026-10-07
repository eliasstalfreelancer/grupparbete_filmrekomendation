import pandas as pd
import numpy as np
import ast
import json
from collections import Counter

#Läser in data
print("\n Läser in data...")
movies = pd.read_csv('Kaggle_Movie_Data/movies_metadata.csv', low_memory=False)
keywords = pd.read_csv('Kaggle_Movie_Data/keywords.csv')
links = pd.read_csv('Kaggle_Movie_Data/links.csv')
ratings = pd.read_csv('Kaggle_Movie_Data/ratings.csv')

#Rensar felaktiga ID-rader i movies_metadata
movies['tmdbId_clean'] = pd.to_numeric(movies['id'], errors='coerce')

links_clean = links.dropna(subset=['tmdbId']).copy()
links_clean['tmdbId_clean'] = links_clean['tmdbId'].astype(int)

merged_links = links_clean.merge(movies, on='tmdbId_clean', how='left')
missing_movies = merged_links['title'].isnull().sum()

print(f"Totalt antal unika filmer i links_small: {links_clean['tmdbId_clean'].nunique()}")
print(f"Filmer i links_small som saknar matchning i movies_metadata: {missing_movies}")

#Parsing av JSON filer för Genre och Keywords

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

# Parsar kolumnerna
movies['parsed_genres'] = parse_json_column(movies['genres'])
keywords['parsed_keywords'] = parse_json_column(keywords['keywords'])

movies['id_int'] = pd.to_numeric(movies['id'], errors='coerce')
keywords['id_int'] = pd.to_numeric(keywords['id'], errors='coerce')

# Merging på id_int
movies_keyw = movies.merge(keywords[['id_int', 'parsed_keywords']], on='id_int', how='left')

# 4. Beräkna saknade keywords på rätt kolumn i den mergade dataramen
kw_lists = movies_keyw['parsed_keywords'].dropna()
no_keywords_count = movies_keyw['parsed_keywords'].apply(lambda x: len(x) if isinstance(x, list) else 0).eq(0).sum()

print(f"Filmer som helt saknar keywords: {no_keywords_count} ({(no_keywords_count/len(movies_keyw))*100:.1f}%)")

# Analysera Genres
all_genres = [g for sublist in movies['parsed_genres'] for g in sublist]
genre_counts = Counter(all_genres)
empty_genres = (movies['parsed_genres'].apply(len) == 0).sum()

print(f"\n[GENRES]")
print(f"Totalt antal unika genrer: {len(genre_counts)}")
print(f"Filmer som helt saknar genrer: {empty_genres}")
print("Top 5 vanligaste genrer:")
for g, count in genre_counts.most_common(5):
    print(f"  - {g}: {count} filmer")

# Analysera Keywords
keyw_lists = movies_keyw['parsed_keywords'].dropna().apply(lambda x: x if isinstance(x, list) else [])
all_keywords = [k for sublist in keyw_lists for k in sublist]
keyw_counts = Counter(all_keywords)
no_keywords_count = (keyw_lists.apply(len) == 0).sum()

print(f"\n[KEYWORDS]")
print(f"Totalt antal unika keywords: {len(keyw_counts)}")
print(f"Filmer som helt saknar keywords: {no_keywords_count} ({(no_keywords_count/len(movies))*100:.1f}%)")
print("Top 5 vanligaste keywords:")
for k, count in keyw_counts.most_common(5):
    print(f"  - {k}: {count} gånger")


# Filtrera på relevanta ratings (>= 4.0)
high_ratings = ratings[ratings['rating'] >= 4.0]
user_high_counts = high_ratings.groupby('userId').size()

# Räkna användare som har minst 10 relevanta ratings
valid_users = user_high_counts[user_high_counts >= 10]

print(f"Totalt antal användare i ratings_small: {ratings['userId'].nunique()}")
print(f"Användare med minst 10 filmer med rating >= 4.0: {len(valid_users)}")
print(f"Andel godkända testanvändare: {(len(valid_users)/ratings['userId'].nunique())*100:.1f}%")