
import streamlit as st
import pandas as pd
import requests
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline  # 🔥 NEW


st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Smart Context-Aware Movie Recommender")

# ==============================
# 🎨 UI STYLING
# ==============================
st.markdown("""
<style>
.movie-card {
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.title {
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("### 🎯 Get Personalized Recommendations")
st.caption("Upload your face image to detect emotion automatically")

# ==============================
# TMDB CONFIG
# ==============================
API_KEY = "c80d31b559df69fee4249b03dc001d90"

# ==============================
# CLEAN TITLE
# ==============================
def clean_title(title):
    return re.sub(r"\(\d{4}\)", "", title).strip()

def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)
    return match.group(1) if match else ""

# ==============================
# TMDB FETCH
# ==============================
@st.cache_data(show_spinner=False)
def get_movie_details(title):
    clean = clean_title(title)
    year = extract_year(title)

    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={clean}&year={year}"

    try:
        response = requests.get(url).json()

        if response.get('results'):
            movie = max(response['results'], key=lambda x: x.get('vote_count', 0))

            poster_path = movie.get('poster_path')
            poster = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

            overview = movie.get('overview') or ""
            rating = movie.get('vote_average') or "N/A"

            return poster, overview, rating

    except:
        pass

    return None, "", "N/A"

# ==============================
# LOAD DATA
# ==============================
movies = pd.read_csv("ml-latest-small/movies.csv")
movies['overview'] = movies['title'].fillna('') + " " + movies['genres'].fillna('')

import numpy as np

# ==============================
# 🔥 LOAD EMBEDDINGS (BERT)
# ==============================
@st.cache_data
def load_embeddings():
    return np.load("embeddings.npy")

embeddings = load_embeddings()

# ==============================
# 🔥 TF-IDF (SECOND SIGNAL)
# ==============================
@st.cache_data
def compute_tfidf(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(data)

tfidf_matrix = compute_tfidf(movies['overview'])

# ==============================
# 🔥 HYBRID SIMILARITY
# ==============================
@st.cache_data
def compute_hybrid_similarity():
    sim_bert = cosine_similarity(embeddings)
    sim_tfidf = cosine_similarity(tfidf_matrix)

    # 🔥 FINAL HYBRID SCORE
    return 0.7 * sim_bert + 0.3 * sim_tfidf

cos_sim = compute_hybrid_similarity()

# ==============================
# RL MEMORY
# ==============================
if "Q_table" not in st.session_state:
    st.session_state.Q_table = {}

def get_q(user, movie):
    return st.session_state.Q_table.get((user, movie), 0)

def update_q(user, movie, reward):
    old = get_q(user, movie)
    new = old + 0.7 * (reward - old)
    st.session_state.Q_table[(user, movie)] = new

def encode_state(sim, c_score, q_score):
    return torch.tensor([sim, c_score, q_score], dtype=torch.float32)

# ==============================
# 🤖 DQN MODEL (PHASE 7)
# ==============================
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


dqn = DQN()
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


# ==============================
# CONTEXT SCORING
# ==============================
def context_score(row, mood, time):
    score = 0
    genres = str(row['genres'])

    # 🌙 NIGHT
    if time == "night":
        if "Horror" in genres or "Thriller" in genres:
            score += 2

    # ☀️ DAY (NEW)
    if time == "day":
        if "Comedy" in genres or "Animation" in genres or "Adventure" in genres:
            score += 2

    # 😊 MOOD
    if mood == "happy":
        if "Comedy" in genres:
            score += 2
        elif "Adventure" in genres:
            score += 1

    if mood == "sad":
        if "Drama" in genres:
            score += 2
        elif "Romance" in genres:
            score += 1

    return score



# ==============================
# LOAD MODELS (CACHED)
# ==============================
@st.cache_resource
def load_hf_model():
    return pipeline(
        "image-classification",
        model="dima806/facial_emotions_image_detection"
    )

@st.cache_resource
def load_fer_model():
    try:
        from fer import FER
        return FER(mtcnn=False)
    except Exception as e:
        st.warning("FER model failed to load, using fallback")
        return None

hf_model = load_hf_model()
fer_model = load_fer_model()

# ==============================
# HYBRID EMOTION DETECTION 🔥
# ==============================
def detect_emotion(image):
    try:
        # HuggingFace (always works)
        hf_result = hf_model(image)
        hf_emotions = {r['label'].lower(): r['score'] for r in hf_result}

        # FER (optional)
        fer_emotions = {}

        if fer_model is not None:
            fer_result = fer_model.detect_emotions(np.array(image))
            if fer_result:
                fer_emotions = fer_result[0]["emotions"]

        # Combine
        combined = {}

        for emo in ["happy", "sad", "angry", "surprise", "neutral"]:
            combined[emo] = (
                0.7 * hf_emotions.get(emo, 0) +
                0.3 * fer_emotions.get(emo, 0)
            )

        return max(combined, key=combined.get)

    except:
        return "neutral"

# ==============================
# MAPPING (UNCHANGED)
# ==============================
def map_emotion_to_mood(emotion):
    if emotion in ["happy", "surprise"]:
        return "happy"
    elif emotion in ["sad", "angry", "fear"]:
        return "sad"
    else:
        return "neutral"

# ==============================
# EXPLANATION
# ==============================
def generate_explanation(row, mood, time, q_score, sim):
    genres = str(row['genres'])
    reasons = []

    score = 0.5*q_score + 0.3*sim

    if score > 0.75:
        level = "🔥 Highly Recommended"
    elif score > 0.6:
        level = "⭐ Strong Match"
    elif score > 0.5:
        level = "👍 Good Choice"
    else:
        level = "🔍 Worth Exploring"

    if sim > 0.7:
        reasons.append("very similar to your selected movie")
    elif sim > 0.5:
        reasons.append("shares similar themes")
    else:
        reasons.append("offers a different experience")

    if mood == "happy" and "Comedy" in genres:
        reasons.append("fits your happy mood")
    if mood == "sad" and "Drama" in genres:
        reasons.append("matches your mood")
    # ☀️ DAY (NEW)
    if time == "night":
        if "Horror" in genres:
            reasons.append("perfect for a night thrill")
        elif "Thriller" in genres:
            reasons.append("keeps you engaged at night")
    # ☀️ DAY (NEW)
    if time == "day":
        if "Comedy" in genres:
            reasons.append("great for a light daytime watch")
        elif "Animation" in genres:
            reasons.append("fun and refreshing for daytime")
        elif "Adventure" in genres:
            reasons.append("keeps the energy up during the day")

    # 🧠 RL reasoning (better)
    if q_score > 0.7:
        reasons.append("you strongly preferred similar movies before")
    elif q_score > 0.3:
        reasons.append("you showed interest in similar movies")

    # ✅ Ensure at least 1 reason exists
    if not reasons:
        reasons.append("it matches your overall preferences")

    return f"{level} 💡 Because it " + ", ".join(reasons[:2]) + "."

# ==============================
# TEXT HANDLING
# ==============================
def smart_truncate(text, length=150):
    if not text:
        return "No description available"
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0] + "..."

def enhance_description(overview, title, genres):
    if not overview or len(overview) < 80:
        return f"{title} is a {genres} movie recommended based on your preferences."
    return overview

# ==============================
# RECOMMENDER
# ==============================
def recommend(movie_title, mood, time, user_id=1):
    idx = movies[movies['title'] == movie_title].index[0]

    scores = list(enumerate(cos_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:20]

    results = []

    for i, sim in scores:
        row = movies.iloc[i]
        c_score = context_score(row, mood, time)
        q_score = get_q(user_id, row['movieId'])

        state = encode_state(sim, c_score, q_score)

        with torch.no_grad():
            dqn_score = dqn(state).item()

        results.append((row, row['title'], sim, c_score, q_score, dqn_score, row['movieId']))

    results = sorted(results, key=lambda x: (0.5*x[5] + 0.25*x[4] + 0.15*x[3] + 0.10*x[2]), reverse=True)

    return results[:10]

# ==============================
# UI INPUT
# ==============================
col1, col2, col3 = st.columns(3)

with col1:
    movie = st.selectbox("🎥 Select Movie", movies['title'])

with col2:
    uploaded_file = st.file_uploader("📸 Upload Face Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")  # 🔥 FIX
        st.image(image, width=120)

        emotion = detect_emotion(image)
        mood = map_emotion_to_mood(emotion)

        st.success(f"Detected Emotion: {emotion}")
    else:
        mood = "neutral"

with col3:
    time = st.selectbox("⏰ Time", ["day", "night"])

# ==============================
# BUTTON
# ==============================
if st.button("🚀 Get Recommendations"):

    recs = recommend(movie, mood, time)

    st.subheader("🎯 Recommended Movies")

    for i, (row, title, sim, c_score, _, dqn_score, mid) in enumerate(recs, 1):

        q_score = get_q(1, mid)



        poster, overview, rating = get_movie_details(title)
        overview = enhance_description(overview, title, row['genres'])
        explanation = generate_explanation(row, mood, time, q_score, sim)

        st.markdown('<div class="movie-card">', unsafe_allow_html=True)

        col1, col2 = st.columns([1,3])

        with col1:
            if poster:
                st.image(poster)
            else:
                st.write("🎬 No Image")

        with col2:
            st.markdown(f'<div class="title">{i}. {title}</div>', unsafe_allow_html=True)
            st.caption(f"⭐ {rating} | 🎯 Context: {c_score} | 🧠 Learned score: {round(q_score,2)}")


            st.caption(f"🤖 AI Score: {round(dqn_score,2)}")

            preview = smart_truncate(overview)
            st.write(preview)

            with st.expander("📖 Read Full Description"):
                st.write(overview)

            st.info(explanation)

            col_like, col_dislike = st.columns(2)

            with col_like:


                if st.button(f"👍 Like {i}", key=f"like_{i}"):

                    # 🔁 Update Q-table
                    update_q(1, mid, 1)
                    new_q = get_q(1, mid)

                    # 🔥 TRAIN DQN
                    state = encode_state(sim, c_score, new_q).unsqueeze(0)

                    target = torch.tensor([[1.0]])  # reward = 1

                    pred = dqn(state)

                    loss = loss_fn(pred, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # 🔥 FORCE REFRESH
                    st.rerun()

            with col_dislike:
                if st.button(f"👎 Dislike {i}", key=f"dislike_{i}"):

                    update_q(1, mid, -1)
                    new_q = get_q(1, mid)

                    # 🔥 TRAIN DQN (negative reward)
                    state = encode_state(sim, c_score, new_q).unsqueeze(0)

                    target = torch.tensor([[-1.0]])

                    pred = dqn(state)

                    loss = loss_fn(pred, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # 🔥 FORCE REFRESH
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

