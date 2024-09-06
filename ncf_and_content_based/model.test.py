import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import csv

# Load preprocessed data
df = pd.read_pickle('preprocessed_data_with_interactions.pkl')

# Ensure no out-of-range values
num_users = len(df['username_encoded'].unique())
num_items = len(df['id_scaled'].unique())

df['username_encoded'] = df['username_encoded'].clip(lower=0, upper=num_users - 1)
df['id_scaled'] = df['id_scaled'].clip(lower=0, upper=num_items - 1)

# Create a mapping from encoded usernames to original usernames
username_mapping = pd.Series(df['username'].values, index=df['username_encoded']).to_dict()

class RecommendationDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # Ensure indices are within valid range and are integers
        self.dataframe['id_scaled'] = self.dataframe['id_scaled'].astype(int)  # Ensure ids are integers
        self.dataframe['username_encoded'] = self.dataframe['username_encoded'].astype(int)  # Ensure ids are integers
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            'username_encoded': torch.tensor(row['username_encoded'], dtype=torch.long),
            'id_scaled': torch.tensor(row['id_scaled'], dtype=torch.long),
            'title_bert_embeddings': torch.tensor(row['title_bert_embeddings'], dtype=torch.float32),
            'description_bert_embeddings': torch.tensor(row['description_bert_embeddings'], dtype=torch.float32),
            'title_bert_embeddings_user_avg': torch.tensor(row['title_bert_embeddings_user_avg'], dtype=torch.float32),
            'description_bert_embeddings_user_avg': torch.tensor(row['description_bert_embeddings_user_avg'], dtype=torch.float32),
            'interaction': torch.tensor(row['interaction'], dtype=torch.float32)
        }


# Splitting data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = RecommendationDataset(train_df)
test_dataset = RecommendationDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the NCF model with user embeddings
class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, title_embedding_dim, description_embedding_dim):
        super(NCFModel, self).__init__()
        # Embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # Additional embeddings for user and item
        self.title_embedding_user_avg = nn.Linear(title_embedding_dim, embedding_dim)
        self.description_embedding_user_avg = nn.Linear(description_embedding_dim, embedding_dim)
        # Neural network layers
        self.fc1_input_dim = embedding_dim * 4  # Updated to include all embeddings
        self.fc1 = nn.Linear(self.fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(self.fc1_input_dim)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        # Dropout layers
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, user_ids, item_ids, title_embeddings, description_embeddings, title_embeddings_user_avg, description_embeddings_user_avg):
        # Ensure ids are of type LongTensor
        user_ids = user_ids.long()
        item_ids = item_ids.long()

        # Clip indices to be within the valid range
        user_ids = torch.clamp(user_ids, 0, self.user_embedding.num_embeddings - 1)
        item_ids = torch.clamp(item_ids, 0, self.item_embedding.num_embeddings - 1)

        # Lookup user and item embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Average embeddings for the user
        avg_title_emb = self.title_embedding_user_avg(title_embeddings_user_avg)
        avg_description_emb = self.description_embedding_user_avg(description_embeddings_user_avg)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb, avg_title_emb, avg_description_emb], dim=-1)
        
        # Feed through neural network layers
        x = self.bn1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.bn2(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.bn3(x)
        x = self.fc3(x)
        
        return x.squeeze()


# Initialize model
num_users = len(df['username_encoded'].unique())
num_items = len(df['id_scaled'].unique())
embedding_dim = 50
title_embedding_dim = len(df['title_bert_embeddings'][0])
description_embedding_dim = len(df['description_bert_embeddings'][0])

model = NCFModel(num_users, num_items, embedding_dim, title_embedding_dim, description_embedding_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
for epoch in range(20):  # Train for 20 epochs
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        user_ids = batch['username_encoded']
        item_ids = batch['id_scaled']
        title_embeddings = batch['title_bert_embeddings']
        description_embeddings = batch['description_bert_embeddings']
        title_embeddings_user_avg = batch['title_bert_embeddings_user_avg']
        description_embeddings_user_avg = batch['description_bert_embeddings_user_avg']
        interactions = batch['interaction']
        
        outputs = model(user_ids, item_ids, title_embeddings, description_embeddings, title_embeddings_user_avg, description_embeddings_user_avg)
        loss = criterion(outputs, interactions)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'ncf_model.pth')

# Function to recommend items
def recommend_for_all_users(top_k=60):
    model.eval()
    
    all_recommendations = {}

    try:
        # Get item embeddings once
        item_embeddings = np.array([model.item_embedding(torch.tensor([i], dtype=torch.long)).detach().numpy().flatten() for i in range(num_items)])
        
        for encoded_user_id in df['username_encoded'].unique():
            try:
                # Get the user index from the encoded user ID
                user_idx = torch.tensor([encoded_user_id], dtype=torch.long)
                
                # Get the user embedding
                user_embedding = model.user_embedding(user_idx).detach().numpy().flatten()
                
                # Calculate cosine similarity between user and item embeddings
                similarities = cosine_similarity([user_embedding], item_embeddings)[0]
                
                # Get top K similar items
                top_items = similarities.argsort()[-top_k:][::-1]
                
                # Map encoded user ID to original username
                original_username = username_mapping[encoded_user_id]
                
                # Store the recommendations
                all_recommendations[original_username] = top_items
                
            except Exception as e:
                print(f"Error recommending items for encoded_user_id '{encoded_user_id}': {e}")
                original_username = username_mapping.get(encoded_user_id, "unknown_user")
                all_recommendations[original_username] = []

    except Exception as e:
        print(f"Error in recommending items: {e}")

    return all_recommendations

# Example usage
try:
    all_user_recommendations = recommend_for_all_users(top_k=50)
    for user, recommendations in all_user_recommendations.items():
        print(f"Top recommendations for {user}: {recommendations}")
except Exception as e:
    print(f"Error in recommending items: {e}")


'''
# if want to save recommendations in csv
try:
    all_user_recommendations = recommend_for_all_users(top_k=50)
    with open('recommendations.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["User", "Recommendations"])  # header
        for user, recommendations in all_user_recommendations.items():
            item_titles = [item_title_mapping[id] for id in recommendations]
            writer.writerow([user, ', '.join(item_titles)])
except Exception as e:
    print(f"Error in recommending items: {e}")

'''


