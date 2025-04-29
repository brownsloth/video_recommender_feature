import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import json
from datasets import load_dataset

# Step 0: Load real data
news_dataset = load_dataset('ag_news')
articles = news_dataset['train'].shuffle(seed=42).select(range(5000))

article_texts = [item['text'] for item in articles]

# Load your transcribed video data
with open('video_transcripts.json', 'r') as f:
    video_transcripts_dict = json.load(f)

video_filenames = list(video_transcripts_dict.keys())
video_texts = list(video_transcripts_dict.values())

# Configuration
NUM_USERS = 1000
TOPICS = ['World', 'Sports', 'Business', 'Sci/Tech']  # AG News topics
users = [f"user_{i}" for i in range(NUM_USERS)]

# Step 1: Simulate User Preferences
fixed_user_preferences = {
    "user_0": ["Sports"],
    "user_1": ["Sci/Tech"],
    "user_2": ["Business", "World"],
    "user_3": ["Sports", "Business"],
    "user_4": ["World"],
    # others will be random
}

# Step 2: Map articles to topics
article_topics = [TOPICS[item['label']] for item in articles]

article_df = pd.DataFrame({
    "article_id": list(range(len(article_texts))),
    "text": article_texts,
    "topic": article_topics
})

# Step 3: Simulate User-Article Interactions
user_article_interactions = []

for user in users:
    if user in fixed_user_preferences:
        preferred_topics = fixed_user_preferences[user]
        candidate_articles = article_df[article_df['topic'].isin(preferred_topics)]
        selected_articles = candidate_articles.sample(n=random.randint(10, 20), random_state=random.randint(1, 1000))
    else:
        selected_articles = article_df.sample(n=random.randint(5, 20), random_state=random.randint(1, 1000))
    
    for article_id in selected_articles['article_id'].tolist():
        user_article_interactions.append((user, article_id))

user_article_df = pd.DataFrame(user_article_interactions, columns=["user_id", "article_id"])

# Step 4: Embeddings
vectorizer = TfidfVectorizer(max_features=1000)

# Combine articles and videos into same vocab space
vectorizer.fit(article_texts + video_texts)

article_embeddings = vectorizer.transform(article_texts)
video_embeddings = vectorizer.transform(video_texts)

# Step 5: Build User Profiles
user_profiles = {}

for user in users:
    article_ids = user_article_df[user_article_df['user_id'] == user]['article_id'].tolist()
    if article_ids:
        profile_embedding = article_embeddings[article_ids].mean(axis=0).A  # Convert to array
        user_profiles[user] = profile_embedding

# Step 6: Recommend Videos
def recommend_videos(user_profile, video_embeddings, top_k=5):
    similarities = cosine_similarity(user_profile, video_embeddings)
    similarities = similarities.flatten()  # (n_videos,)
    top_k_indices = np.argsort(similarities)[::-1][:top_k]  # indices of top similarities
    top_k_similarities = similarities[top_k_indices]  # pick corresponding similarity scores
    return top_k_indices, top_k_similarities

# Example: Recommend for one user
sample_user = random.choice(users)
user_profile = user_profiles[sample_user]
recommended_video_ids, similarity_scores = recommend_videos(user_profile, video_embeddings)

print(f"\nRecommendations for {sample_user}:")
for idx, score in zip(recommended_video_ids, similarity_scores):
    print(f"Video File: {video_filenames[idx]}, Similarity: {score:.4f}")
    print(f"Transcript: {video_texts[idx][:80]}...\n")