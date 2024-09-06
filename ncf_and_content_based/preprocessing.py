import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import BertTokenizer, BertModel
import torch

# Load preprocessed data
try:
    df_sorted = pd.read_csv('formatted_extracted_data.csv')
except FileNotFoundError:
    print("Error: Data file not found.")
    exit(1)

# Define functions for BERT embeddings and calculating interaction score
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
except Exception as e:
    print(f"Error loading BERT model: {e}")
    exit(1)


def get_bert_embeddings(text):
    if pd.isna(text) or not isinstance(text, str):
        # Handle missing or non-string values by returning a zero vector
        return np.zeros((768,))
    
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    return embeddings

def calculate_dot_product(embedding1, embedding2):
    return np.dot(embedding1, embedding2)

def calculate_similarity(embedding1, embedding2):
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(embedding1, embedding2) / (norm1 * norm2)

# Calculate embeddings
df_sorted['title_bert_embeddings'] = df_sorted['title'].apply(lambda x: get_bert_embeddings(x))
df_sorted['description_bert_embeddings'] = df_sorted['description'].apply(lambda x: get_bert_embeddings(x))

# Label Encoding for username
le = LabelEncoder()
df_sorted['username_encoded'] = le.fit_transform(df_sorted['username'])

# Standard Scaling for id
scaler = StandardScaler()
df_sorted['id_scaled'] = scaler.fit_transform(df_sorted['id'].values.reshape(-1, 1))

# Calculate average embeddings for each user
user_embeddings = df_sorted.groupby('username').agg({
    'title_bert_embeddings': lambda x: np.mean(np.array(list(x)), axis=0),
    'description_bert_embeddings': lambda x: np.mean(np.array(list(x)), axis=0)
}).reset_index()

# Merge average user embeddings with the original DataFrame
df_sorted = pd.merge(df_sorted, user_embeddings, on='username', suffixes=('', '_user_avg'))

# Calculate interaction score
def calculate_interaction_score(row):
    user_title_emb_avg = row['title_bert_embeddings_user_avg']
    user_description_emb_avg = row['description_bert_embeddings_user_avg']
    item_title_emb = row['title_bert_embeddings']
    item_description_emb = row['description_bert_embeddings']
    
    # User-Item Interaction
    user_item_interaction = (calculate_dot_product(user_title_emb_avg, item_title_emb) +
                             calculate_dot_product(user_description_emb_avg, item_description_emb)) / 2
    
    # User's Past Post Reference (for simplicity, using only average embeddings here)
    past_posts_similarity = (calculate_similarity(user_title_emb_avg, item_title_emb) +
                             calculate_similarity(user_description_emb_avg, item_description_emb)) / 2
    
    # Interaction Score (combine both signals)
    interaction_score = (user_item_interaction + past_posts_similarity) / 2
    
    return interaction_score

df_sorted['interaction'] = df_sorted.apply(calculate_interaction_score, axis=1)

# Save the preprocessed data with interaction scores
df_sorted.to_pickle('preprocessed_data_with_interactions.pkl')

print("Preprocessing complete with interaction scores. Data saved to 'preprocessed_data_with_interactions.pkl'.")
