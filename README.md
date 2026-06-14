# Smart Context-Aware Movie Recommendation System

## Overview

This project is a movie recommendation system that goes beyond traditional content-based recommendations by considering a user's current context and feedback. Instead of recommending movies solely based on genres or keywords, the system also takes into account factors such as user mood, time of day, and past interactions.

The project was developed to explore recommender systems, machine learning, emotion recognition, and reinforcement learning in a single application. The goal is to create a recommendation engine that adapts to user preferences over time and provides more personalized movie suggestions.

---

## Key Features

### Hybrid Movie Similarity

The recommendation engine combines multiple techniques to identify similar movies:

- BERT embeddings for semantic understanding of movie information
- TF-IDF vectorization for keyword-based similarity
- Cosine similarity for recommendation generation

Using both approaches helps capture semantic meaning as well as direct textual similarity.

### Context-Aware Recommendations

Recommendations are adjusted based on:

- User mood
- Time of day (Day/Night)

This allows the system to prioritize movies that better match the user's current context.

### Emotion Recognition

Users can upload a facial image, and the system automatically detects emotions using:

- Hugging Face emotion classification model
- FER (Facial Emotion Recognition)

The detected emotion is then mapped to a mood that influences recommendation ranking.

### Learning from User Feedback

Users can like or dislike recommended movies.

The system uses this feedback to learn user preferences and improve future recommendations through reinforcement learning techniques.

### Deep Learning-Based Ranking

A Deep Q Network (DQN) is used to rank recommendations using:

- Similarity score
- Context score
- Learned user preference score

This enables the system to become more personalized as feedback is collected.

### Movie Information Integration

Additional movie details are fetched using TMDB API, including:

- Movie posters
- Ratings
- Descriptions

This improves the overall user experience and makes recommendations easier to explore.

---

## Dataset

The project uses the **MovieLens Latest Small Dataset** provided by GroupLens Research.

The dataset contains:

- Movie titles
- Genres
- Ratings
- Metadata required for recommendation generation

Dataset Link:

https://grouplens.org/datasets/movielens/

---

## System Workflow

### Recommendation Pipeline

```text
Movie Dataset
      ↓
Feature Extraction
(BERT + TF-IDF)
      ↓
Similarity Calculation
      ↓
Context Scoring
(Mood + Time)
      ↓
Reinforcement Learning
(User Feedback)
      ↓
Deep Q Network Ranking
      ↓
Final Recommendations
      ↓
Streamlit Interface
```

### Emotion Detection Pipeline

```text
Image Upload
      ↓
Emotion Detection
(Hugging Face + FER)
      ↓
Mood Identification
      ↓
Context Scoring
      ↓
Recommendation Ranking
```

---

## Technologies Used

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Sentence Transformers

### Deep Learning

- PyTorch
- Transformers

### Computer Vision

- FER
- OpenCV
- Pillow

### Web Framework

- Streamlit

### APIs

- TMDB API

### Deployment

- ngrok

---

## Project Structure

```text
Movie-Recommendation-System/
│
├── app.py
├── embeddings.npy
├── README.md
│
├── ml-latest-small/
│   ├── movies.csv
│   ├── ratings.csv
│
└── assets/
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

### Install Dependencies

```bash
pip install streamlit pyngrok transformers sentence-transformers \
scikit-learn pandas numpy fer mtcnn opencv-python-headless pillow torch
```

---

## Running the Application

```bash
streamlit run app.py
```

---

## Learning Outcomes

This project provided practical experience in:

- Recommender Systems
- Natural Language Processing (NLP)
- Computer Vision
- Reinforcement Learning
- Deep Reinforcement Learning
- Streamlit Application Development
- API Integration
- Model Deployment

It also demonstrates how multiple AI techniques can be integrated into a single real-world application.

---

## Future Improvements

Potential enhancements include:

- Real-time webcam emotion detection
- Voice-based emotion recognition
- Collaborative filtering integration
- Multi-user support
- Mobile application development
- Cloud deployment on AWS, Azure, or GCP

---

## Authors

### Ch. Durga Prasad

B.Tech, Computer Science and Engineering  
B.V. Raju Institute of Technology

### Ch. Nimika

B.Tech, Computer Science and Engineering  
B.V. Raju Institute of Technology

### Project Guide

**Mr. S. Anjanayya**  
Assistant Professor  
Department of Computer Science and Engineering  
B.V. Raju Institute of Technology

---

## License

This project was developed for academic and educational purposes.
