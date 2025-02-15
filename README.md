# Movie Recommender System 🎥

## 📜 Overview
This is a Natural Language Processing (NLP)-based movie recommender system that suggests movies similar to a user-input movie. It utilizes **HuggingFace's Sentence Transformers** for text vectorization and **Streamlit** for an interactive front-end.

## 🧰 Features
- **Interactive Front-End**: Built with Streamlit for user-friendly movie recommendations.
- **NLP-powered Recommendations**: Uses HuggingFace's `sentence-transformers` to vectorize movie descriptions and calculate cosine similarity.
- **Dynamic Posters**: Fetches movie posters dynamically using TMDB API.
- **Efficient Data Processing**: Data preprocessed and stored as a pickle file for faster application performance.

---

## 🛠️ Technologies Used
- **Programming Language**: Python
- **Natural Language Processing**: HuggingFace's Sentence Transformers
- **Web Framework**: Streamlit
- **Visualization**: TMDB API for dynamic movie posters
- **Dataset**: [TMDB 5000 Movies Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## 📂 Project Structure
Movie-Recommender-System/.dist ├── app.py # Main application script ├── movie-recommender-system.ipynb # Data preprocessing notebook ├── new_df.pkl # Pickle file containing processed data ├── requirements.txt # Python dependencies └── README.md # Project documentation


---

## 🚀 How to Run the Project Locally
### Prerequisites
- Python 3.8 or above installed.
- TMDB API key (for fetching movie posters).

### Installation
1. Clone the repository:
   ```bash
    git clone https://github.com/your-username/Movie-Recommender-System.git
   cd Movie-Recommender-System
2. Install Dependencies:
   `pip install -r requirements.txt`
3. Add your TMDB API key:
    Open app.py and replace YOUR_TMDB_API_KEY with your TMDB API key. To get that go to TMDB and create your account > Settings > API > Create your API Key (Replace it with 'YOUR_TMDB_API_KEY'.
4. Run the application:
   `streamlit run app.py`
5. Access your application on your browser.

### 🎯 Usage
1. Enter the title of a movie in the input box.
2. Click the Recommend button.
3. View recommended movies along with their posters.

### 🔗 API Integration
TMDB API: Used to fetch dynamic movie posters based on movie IDs.

### Screenshots
![image](https://github.com/user-attachments/assets/1766ca53-8f6f-4303-8339-10beae6edb01)

![image](https://github.com/user-attachments/assets/cbe8fd03-389d-4aa9-a2bb-5d1d0078bdb3)

![image](https://github.com/user-attachments/assets/230a0d68-484d-4d50-b717-724510663053)






### 🧑‍💻 Author
Developed by **Yash Malaviya**.
Feel free to connect on LinkedIn: `https://www.linkedin.com/in/yash-malaviya/` or check out my other projects on GitHub.

