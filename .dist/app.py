import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import os
from dotenv import load_dotenv

import concurrent.futures

from requests.adapters import HTTPAdapter
from urllib3.util import Retry

load_dotenv() # Load .env file
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Loading the DataFrame from ipynb file in which we did Data-processing
movies = pd.read_pickle(".dist/new_df.pkl")

embeddings_matrix = np.stack(movies['embeddings'].values)

# Use session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Fetch movie posters
# @st.cache_data(ttl=86400) # Cache for 24 hours
def fetch_poster(movie_id):
    try:
        response = session.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={
                'api_key': TMDB_API_KEY,
                'language': 'en-US'
            },
            timeout=10 #Avoid long waits
        
        )
        response.raise_for_status() # Raise HTTP errors
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
        else:
            return "https://via.placeholder.com/200x300?text=Poster+Not+Found"

    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster: {e}")
        return "https://via.placeholder.com/200x300?text=Connection+Failed"

def fetch_all_posters(movie_ids):
    """Fetch posters in parallel"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(fetch_poster, movie_ids))

# Function to recommend movies
def recommend_movies(user_input, num_recommendations=5):
    #Encoding the user's input
    input_embedding = model.encode(user_input)

    #Calculating cosine similarity
    similarity_scores = cosine_similarity([input_embedding], embeddings_matrix).flatten()

    # Sort scores and exclude the input movie if it exists in the dataset
    movies['similarity'] = similarity_scores
    filtered_df = movies[movies['title'].str.lower() != user_input.lower()]

    #Get top recommendations
    top_indices = filtered_df['similarity'].nlargest(num_recommendations).index
    return filtered_df.loc[top_indices][['title', 'movie_id']]


st.title('Movie Recommender System')

selected_movie = st.selectbox("Enter the title of movie", movies['title'])

if st.button('Recommend'):
    recommendations = recommend_movies(selected_movie)
    movie_ids = recommendations['movie_id'].tolist()

    # Fetch all posters in parallel
    posters = fetch_all_posters(movie_ids)

    # Display recommended movies horizontally
    cols = st.columns(len(recommendations))  # Create as many columns as recommendations
    for col, poster, title in zip(cols, posters, recommendations['title']):
        with col:
            st.image(poster, width=200)
            st.write(title)

    # Custom HTML Code to add spacing between movie recommendations
    st.markdown(
        """
        <style>
        .stHorizontal { display: flex; justify-content: center; gap: 50px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
