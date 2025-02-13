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

st.set_page_config(
    page_title="CineMatch - Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Load environment variables
load_dotenv() # Load .env file
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Custom CSS styling
st.markdown("""
    <style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        color: #ffffff;
    }
    
    /* Title styling */
    .title {
        font-size: 2.5em;
        color: #ff4b4b;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 30px;
    }
    
    /* Select box styling */
    .stSelectbox>div>div>div>input {
        background-color: #333333 !important;
        color: white !important;
        border-radius: 10px;
        padding: 12px;
        font-size: 1.1em;
    }
            
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #ff4b4b, #ff6b6b);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-size: 1.2em;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 8px rgba(255,75,75,0.3);
    }
            
    /* Movie card styling */
    .movie-card {
        background: #333333;
        border-radius: 15px;
        padding: 15px;
        transition: transform 0.3s;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
    }
            
    /* Poster image styling */
    .stImage>img {
        border-radius: 10px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        transition: transform 0.3s;
    }
    
    .stImage>img:hover {
        transform: scale(1.03);
    }
    
    /* Loading spinner */
    .stSpinner>div>div {
        border-color: #ff4b4b transparent transparent transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="title">🍿 CineMatch</div>', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h3 style="color: #cccccc;">Discover Your Next Favorite Movie</h3>
        <p style="color: #888888;">Powered by AI Recommendations & TMDB</p>
    </div>
""", unsafe_allow_html=True)

# Load data and models
def load_data_and_model():
    # Loading the DataFrame from ipynb file in which we did Data-processing
    movies = pd.read_pickle(".dist/new_df.pkl")
    embeddings_matrix = np.stack(movies['embeddings'].values)

    # Load sentence transformer model from HuggingFace
    model = SentenceTransformer('all-MiniLM-L6-v2')

    return movies, embeddings_matrix, model

movies, embeddings_matrix, model = load_data_and_model()


# Configuring API Session
# Use session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Fetch movie posters
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

# UI
selected_movie = st.selectbox(
    "🎬 Search for a movie:",
    movies['title'],
    index=0,
    help="Start typing to search our movie database"
)

if st.button('✨ Get Recommendations'):
    with st.spinner('🔍 Finding your perfect matches...'):
        recommendations = recommend_movies(selected_movie)
        movie_ids = recommendations['movie_id'].tolist()

        # Fetch all posters in parallel
        posters = fetch_all_posters(movie_ids)

    st.markdown('### 🎉 Recommended for You')

    # Display recommended movies horizontally
    cols = st.columns(len(recommendations))  # Create as many columns as recommendations
    for col, poster, (_, row) in zip(cols, posters, recommendations.iterrows()):
        with col:
            poster_url = poster if poster else "https://via.placeholder.com/200x300?text=Poster+Not+Found"
            title = row['title']
            
            st.markdown(f"""
            <div class="movie-card">
                <div style="position: relative;">
                    <img src="{poster_url}" width="100%" style="border-radius: 10px;">
                </div>
                <h4 style="margin: 15px 0 5px 0; color: white; text-align: center;">{title}</h4>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: #666;">
        <hr style="border-color: #333;">
        <p>Movie data powered by TMDb • Made with ❤️ by Yash Malaviya</p>
    </div>
    """, unsafe_allow_html=True)
