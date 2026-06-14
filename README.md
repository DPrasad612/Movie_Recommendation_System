Smart Context-Aware Movie Recommendation System
Overview

This project is a movie recommendation system that goes beyond traditional content-based recommendations by considering a user's current context and feedback. Instead of recommending movies only based on genres or keywords, the system also takes into account factors such as user mood, time of day, and past interactions.

The project was developed as part of an exploration into recommender systems, machine learning, emotion recognition, and reinforcement learning. The goal was to build a recommendation engine that can adapt to user preferences over time and provide more personalized suggestions.

What Makes This Different?

Most basic recommendation systems generate suggestions using similarity between movies. While that works, it doesn't always reflect what a user wants to watch at a particular moment.

For example:

A user might prefer comedy movies when feeling happy.
Horror or thriller movies may be more suitable for late-night viewing.
User preferences change over time and should influence future recommendations.

This project attempts to address those challenges by combining multiple recommendation techniques into a single system.

Features
Hybrid Movie Similarity

The recommendation engine combines:

BERT embeddings for semantic understanding of movie information
TF-IDF vectorization for keyword-based similarity
Cosine similarity for finding related movies

Using both approaches helps capture semantic meaning as well as direct textual similarity.

Context-Aware Recommendations

Recommendations are adjusted based on:

User mood
Time of day (Day/Night)

This allows the system to prioritize movies that better fit the user's current situation.

Emotion Recognition

Users can upload a facial image and the system automatically detects emotion using:

Hugging Face emotion classification model
FER (Facial Emotion Recognition)

The detected emotion is mapped to a mood that influences recommendation ranking.

Learning from User Feedback

Users can like or dislike recommended movies.

The system stores this feedback and uses it to learn user preferences over time through reinforcement learning techniques.

Deep Learning-Based Ranking

A Deep Q Network (DQN) is used to rank recommendations using:

Similarity score
Context score
Learned user preference score

This creates a ranking mechanism that becomes more personalized as feedback is collected.

Movie Information Integration

Additional movie details are fetched from TMDB, including:

Posters
Ratings
Descriptions

This improves the overall user experience and makes recommendations easier to explore.

Dataset

The project uses the MovieLens Latest Small Dataset provided by GroupLens Research.

The dataset contains:

Movie titles
Genres
Ratings
Metadata required for recommendation generation

Dataset:
https://grouplens.org/datasets/movielens/
