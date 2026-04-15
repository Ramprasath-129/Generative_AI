import pandas as pd
import numpy as np
import faiss
import zipfile
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# DOWNLOAD DATASET AUTOMATICALLY
# -----------------------------
url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
zip_path = "ml-latest-small.zip"
extract_folder = "ml-latest-small"

if not os.path.exists("movies.csv"):
    print("Downloading dataset...")

    r = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(r.content)

    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

    # Move movies.csv to current folder
    os.rename(f"{extract_folder}/movies.csv", "movies.csv")

    print("Dataset ready ✅")

# -----------------------------
# LOAD DATASET
# -----------------------------
movies = pd.read_csv("movies.csv")

print("\nDataset Loaded:")
print(movies.head())

# -----------------------------
# PREPROCESS TEXT
# -----------------------------
movies["text"] = movies["title"] + " " + movies["genres"]

# -----------------------------
# CREATE EMBEDDINGS (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
embeddings = vectorizer.fit_transform(movies["text"]).toarray()

print("Embedding shape:", embeddings.shape)

# -----------------------------
# BUILD FAISS INDEX
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

print("FAISS index built with", index.ntotal, "items")

# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def recommend(movie_title, k=5):
    if movie_title not in movies["title"].values:
        print("Movie not found!")
        return

    idx = movies[movies["title"] == movie_title].index[0]
    query_vector = embeddings[idx].reshape(1, -1).astype('float32')

    distances, indices = index.search(query_vector, k+1)

    print(f"\nRecommendations for: {movie_title}\n")

    for i in range(1, k+1):
        rec_idx = indices[0][i]
        print(movies.iloc[rec_idx]["title"])

# -----------------------------
# TEST
# -----------------------------
recommend("Toy Story (1995)")
