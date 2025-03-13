import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model
import re

# Load CSV data for two users
user2_df = pd.read_csv('/Users/thalialightstone/Documents/LIS 598D Search and Discovery /proj2/friendBooks_with_genres.csv')
user1_df = pd.read_csv('/Users/thalialightstone/Documents/LIS 598D Search and Discovery /proj2/mybooks_with_genres.csv')


# Extract relevant columns
columns = ['Book Id', 'Title', 'ISBN', 'My Rating', 'Genre']
user1_df = user1_df[columns]
user2_df = user2_df[columns]

# Filter books with ratings >= 3
user1_df = user1_df[user1_df['My Rating'] >= 3]
user2_df = user2_df[user2_df['My Rating'] >= 3]

# Encode Book Ids and Genres
genre_encoder = LabelEncoder()
book_encoder = LabelEncoder()

# Combine Book Ids from both users
all_book_ids = pd.concat([user1_df['Book Id'], user2_df['Book Id']])

# Fit encoder on combined Book Ids and transform both
book_encoder.fit(all_book_ids)
user1_df['Book Id'] = book_encoder.transform(user1_df['Book Id'])
user2_df['Book Id'] = book_encoder.transform(user2_df['Book Id'])


# Combine genres from both users
all_genres = pd.concat([user1_df['Genre'], user2_df['Genre']])

# Fit encoder on combined genres and transform both
genre_encoder.fit(all_genres)
user1_df['Genre'] = genre_encoder.transform(user1_df['Genre'])
user2_df['Genre'] = genre_encoder.transform(user2_df['Genre'])


# Merge data for training
merged_df = pd.concat([user1_df, user2_df])

# Prepare training data
user_ids = np.concatenate([np.zeros(len(user1_df)), np.ones(len(user2_df))])  # 0 for user1, 1 for user2
book_ids = merged_df['Book Id'].values
genres = merged_df['Genre'].values
ratings = merged_df['My Rating'].values

# Train-test split
train_data, test_data, train_labels, test_labels = train_test_split(
    np.stack([user_ids, book_ids, genres], axis=1), ratings, test_size=0.2, random_state=42)

# Build TensorFlow model
user_input = Input(shape=(1,))
book_input = Input(shape=(1,))
genre_input = Input(shape=(1,))

user_embedding = Embedding(2, 8)(user_input)
book_embedding = Embedding(len(book_encoder.classes_), 8)(book_input)
genre_embedding = Embedding(len(genre_encoder.classes_), 4)(genre_input)

user_vec = Flatten()(user_embedding)
book_vec = Flatten()(book_embedding)
genre_vec = Flatten()(genre_embedding)

merged = Concatenate()([user_vec, book_vec, genre_vec])
dense = Dense(64, activation='relu')(merged)
output = Dense(1)(dense)

model = Model(inputs=[user_input, book_input, genre_input], outputs=output)
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit([train_data[:, 0], train_data[:, 1], train_data[:, 2]], train_labels, epochs=5, batch_size=32)

# Generate recommendations for user2 based on books liked by user1
user1_liked_books = user1_df[user1_df['My Rating'] >= 3]
user2_read_books = set(user2_df['Book Id'].values)

# Function to check if a book is part of a series but not the first book
def is_non_first_in_series(title):
    # Look for patterns like (#2), (#3), etc. in the title indicating series
    match = re.search(r'#(\d+)', title)
    if match:
        # Exclude if it's not the first book in the series
        return int(match.group(1)) > 1
    return False

recommendations = []
for _, row in user1_liked_books.iterrows():
    # Skip non-first books in a series
    if is_non_first_in_series(row['Title']):
        continue

    # Recommend if not already read and rating prediction is >= 3
    if row['Book Id'] not in user2_read_books:
        predicted_rating = model.predict([
            np.array([[1]]), 
            np.array([[row['Book Id']]]), 
            np.array([[row['Genre']]])
        ])[0][0]
        if predicted_rating >= 3:
            recommendations.append((row['Title'], predicted_rating))

# Sort recommendations by predicted rating
recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

# Display recommendations
print("Recommended books for User 2 based on User 1's preferences:")
for title, rating in recommendations:
    print(f"{title} (Predicted Rating: {rating:.2f})")
