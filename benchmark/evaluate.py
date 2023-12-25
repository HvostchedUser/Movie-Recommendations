import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset as SurpriseDataset
from surprise import Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import KNNWithMeans
folder_path = '../data/raw/ml-100k'
import pickle


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Re-loading the ratings DataFrame
ratings_file = folder_path+'/u.data'
ratings_df = pd.read_csv(ratings_file, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Define a custom dataset for PyTorch
class MovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = torch.LongTensor(users)
        self.movies = torch.LongTensor(movies)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


# Encoding user IDs and movie IDs
user_ids = ratings_df['user_id'].astype('category').cat.codes.values
movie_ids = ratings_df['movie_id'].astype('category').cat.codes.values

# Normalizing ratings
scaler = MinMaxScaler()
ratings = scaler.fit_transform(ratings_df[['rating']].values.astype(float)).flatten()

# Splitting the dataset into training and test set
train_user_ids, test_user_ids, train_movie_ids, test_movie_ids, train_ratings, test_ratings = train_test_split(
    user_ids, movie_ids, ratings, test_size=0.2, random_state=42)

# Creating PyTorch datasets
train_dataset = MovieLensDataset(train_user_ids, train_movie_ids, train_ratings)
test_dataset = MovieLensDataset(test_user_ids, test_movie_ids, test_ratings)

# Creating data loaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Number of unique users and movies
num_users = len(np.unique(user_ids))
num_movies = len(np.unique(movie_ids))

num_users, num_movies

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, user_input, movie_input):
        user_embedded = self.user_embedding(user_input)
        movie_embedded = self.movie_embedding(movie_input)
        x = torch.cat([user_embedded, movie_embedded], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


# Load and preprocess data
# Re-loading the ratings DataFrame
ratings_file = folder_path+'/u.data'
movies_file =  folder_path+'/u.item'

ratings_df = pd.read_csv(ratings_file, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies_df = pd.read_csv(movies_file, sep='|', encoding='latin-1', names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDB_URL'] + ['genre'+str(i) for i in range(19)])

# --- SVD Model (using surprise library) ---
data = SurpriseDataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], Reader(rating_scale=(1, 5)))
trainset, testset = surprise_train_test_split(data, test_size=0.2)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)
svd_loss = accuracy.rmse(predictions)

# --- KNN Model ---
knn_model = KNNWithMeans(sim_options={'name': 'cosine', 'user_based': True})
knn_model.fit(trainset)
predictions = knn_model.test(testset)
knn_loss = accuracy.rmse(predictions)

# Initialize the model
num_epochs=15
embedding_size = 10
model = RecommenderNet(num_users, num_movies, embedding_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for user_input, movie_input, ratings in train_loader:
        # Convert inputs to LongTensor
        user_input = user_input.long()
        movie_input = movie_input.long()
        ratings = ratings.float()

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(user_input, movie_input)
        loss = criterion(outputs, ratings.view(-1, 1))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")



# Display losses
print(f"SVD Model Loss (RMSE): {svd_loss}")
print(f"KNN Model Loss (RMSE): {knn_loss}")
print(f"NN Model Loss (MSE): {loss.item()}")
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(svd_model, f)
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)
torch.save(model.state_dict(), 'neural_network_model.pth')


