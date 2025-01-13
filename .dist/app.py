import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import requests

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Loading the DataFrame from ipynb file in which we did Data-processing
movies = pd.read_pickle(".dist/new_df.pkl")

embeddings_matrix = np.stack(movies['embeddings'].values)

# Fetch movie posters
def fetch_poster(movie_id):
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=REDACTED&language=en-US")
    data = response.json()

    return "https://image.tmdb.org/t/p/w500" + data['poster_path']

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

    # Display recommended movies horizontally
    cols = st.columns(len(recommendations))  # Create as many columns as recommendations
    for col, (i, row) in zip(cols, recommendations.iterrows()):
        with col:
            movie_poster = fetch_poster(row['movie_id'])
            movie_title = row['title']
            st.image(movie_poster, width=200)
            st.write(movie_title)

    # Custom HTML Code to add spacing between movie recommendations
    st.markdown(
        """
        <style>
        .stHorizontal { display: flex; justify-content: center; gap: 50px; }
        </style>
        """,
        unsafe_allow_html=True,
    )