import pandas as pd
import numpy as np
import ast
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
    def extract(val):
        try:
            if pd.isna(val): return []
            parsed = ast.literal_eval(val)
            return[item['name'] for iten in parsed if isinstane(item,dict) and 'name' in item]
        except:
            return []
    return column_series.apply(extract)

movies['parsed_genre'] = parse_json_column(movies['genres'])
keywords['parsed_keywords'] = parse_json_column(keywords['keywords'])

