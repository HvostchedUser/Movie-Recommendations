# Movie Recommendation System Report

## Introduction

This report is about the development of a movie recommendation system using the MovieLens 100K dataset. The system provides personalized movie suggestions to users based on their past ratings. Three different models were implemented and compared: Singular Value Decomposition (SVD), K-Nearest Neighbors (KNN), and a Neural Network using PyTorch.

## Data Analysis

The MovieLens 100K dataset comprises 100,000 ratings from 943 users on 1682 movies. Ratings range from 1 to 5. The dataset includes user demographic information and movie details.

- Most ratings are in the 3-4 range, indicating a tendency towards positive ratings.
- The user demographic is varied, with a noticeable concentration in the younger age group.
- Movies are distributed across diverse genres, some genres are more common.

There is lots of information in the data exploration notebooks

## Model Implementation

### 1. SVD Model

Implemented using the `surprise` library, SVD is a matrix factorization technique that decomposes the user-item rating matrix into latent factors.

### 2. KNN Model

Also implemented with `surprise`, this model utilizes user-based collaborative filtering to find similar users and recommend movies based on user similarity.

### 3. Neural Network Model

A model built using PyTorch, featuring embedding layers for users and movies for rating prediction.

## Model Advantages and Disadvantages

### SVD

- **Advantages**: Handles sparse data well; efficient in capturing latent factors.
- **Disadvantages**: Less interpretable; struggles with new items.

### KNN

- **Advantages**: Intuitive, interpretable; can provide personalized recommendations.
- **Disadvantages**: Sensitive to sparse data; scalability issues with large datasets.

### Neural Network

- **Advantages**: Highly flexible and capable of modeling complex non-linear relationships.
- **Disadvantages**: Requires more computational resources; risk of overfitting; less transparent.

## Training Process

The models were trained using respective a splitted dataset. For the neural network, a custom training loop was implemented with loss calculation and backpropagation.

## Evaluation

The models were evaluated based on the Root Mean Square Error (RMSE) metric. This metric measures the average magnitude of the errors between predicted and actual ratings.

## Results

The results varied between models, but all of them have shown relatively good results. 
