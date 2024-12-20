# 🗂️ Introduction to Recommender Systems

**Recommender Systems** are algorithms designed to help users discover products, information, or content tailored to their preferences. They play a significant role in applications from online shopping and streaming services to news feeds and social media, enhancing user experience by reducing the effort required to find relevant items.

---

## 📖 What is a Recommender System?

A **Recommender System** is a specialized type of information filtering system that learns from user behaviors, preferences, and other data to make personalized suggestions. It aims to present items that are most relevant to each user, typically based on:

- **Historical interactions** (e.g., items viewed, liked, or purchased).
- **User demographics** and **preferences**.
- **Content characteristics** (e.g., genre, category, attributes).

Common examples of recommendation systems include movie suggestions on Netflix, product recommendations on Amazon, and friend suggestions on social media platforms.

---

## 🧠 Key Approaches in Recommender Systems

Recommender systems generally fall into three main categories:

### 1. **Collaborative Filtering**

Collaborative filtering methods predict a user’s interests by identifying patterns from the behavior of similar users. Collaborative filtering operates on two main techniques:

   - **User-Based Collaborative Filtering**: Finds users with similar preferences and recommends items liked by those similar users. For example, if User A and User B both liked Item X, and User A also liked Item Y, User B may be recommended Item Y.
   
   - **Item-Based Collaborative Filtering**: Looks for items that are similar based on users' past interactions. For example, if many users who liked Item X also liked Item Y, those who like Item X are recommended Item Y.

#### User-Item Matrix Representation
A **user-item matrix** is often used to represent interactions, where each row is a user, each column is an item, and cell values represent the user’s interaction with the item (e.g., a rating or binary indicator).

#### Cosine Similarity for Collaborative Filtering
One method to measure similarity between users or items is **cosine similarity**:

$\text{cosineSimilarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$

where $A$ and $B$ are user or item vectors.

#### Example Code (Using Cosine Similarity)
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# User-item matrix
user_item_matrix = np.array([[5, 0, 3], [4, 0, 0], [1, 1, 0], [0, 0, 5]])

# Calculate cosine similarity between items
similarity_matrix = cosine_similarity(user_item_matrix)
```

### 2. **Content-Based Filtering**

Content-based filtering recommends items based on the features of the items themselves and the user's past preferences. Each item is represented by a set of features (e.g., genre, author, or description in the case of movies or books). The system recommends items with similar features to those that a user has shown interest in.

#### Term Frequency-Inverse Document Frequency (TF-IDF)
Content-based systems often use **TF-IDF** to analyze the item’s content, helping recommend items that match the user's profile.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample content (e.g., descriptions of items)
descriptions = ["A sci-fi movie about space", "A romantic drama set in Paris"]

# Convert content to TF-IDF representation
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(descriptions)
```

### 3. **Hybrid Recommender Systems**

Hybrid approaches combine collaborative filtering and content-based methods to benefit from both approaches. For example, they may use collaborative filtering as the primary recommendation technique but incorporate content-based filtering when user interaction data is sparse (cold-start problem).

---

## 📏 Key Metrics for Evaluating Recommender Systems

To measure the effectiveness of a recommender system, several metrics are commonly used:

- **Precision and Recall**: Evaluate the relevance of the recommended items.
  - **Precision** measures the proportion of recommended items that are relevant.
  - **Recall** measures the proportion of relevant items that were recommended.

- **Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)**: Used when working with rating predictions, these metrics quantify the difference between predicted and actual ratings.

  $$\text{MAE} = \frac{1}{N} \sum{i=1}^N |yi - \hat{y}i|$$

  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum{i=1}^N (yi - \hat{y}i)^2}$$

- **Coverage**: The proportion of items the system is able to recommend, indicating the diversity of recommendations.

- **Diversity and Novelty**: Assesses the variety of recommendations and how likely the recommendations are to introduce users to new items.

---

## 📐 Implementing a Simple Collaborative Filtering Model in Python

Below is an example of a collaborative filtering-based recommendation using a user-item interaction matrix and cosine similarity.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item matrix
user_item_matrix = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 4, 4],
    [0, 1, 5, 4],
])

# Calculate item-item similarity matrix
item_similarity = cosine_similarity(user_item_matrix.T)

# Function to predict ratings for a user based on item similarity
def predict_ratings(user_index, user_item_matrix, item_similarity):
    user_ratings = user_item_matrix[user_index]
    ratings_pred = item_similarity.dot(user_ratings) / np.array([np.abs(item_similarity).sum(axis=1)])
    return ratings_pred

# Predict ratings for the first user
predicted_ratings = predict_ratings(0, user_item_matrix, item_similarity)
print("Predicted Ratings:", predicted_ratings)
```

---

## 🔥 Applications of Recommender Systems

Recommender systems are widely applied across domains and industries:

- **E-commerce**: Personalized product recommendations based on purchase history, browsing habits, or items in a user’s cart.
- **Media and Entertainment**: Suggesting music, movies, or TV shows on platforms like Spotify or Netflix.
- **News and Social Media**: Curating articles, friends, or content creators tailored to the user’s interests.
- **Education**: Personalized learning paths and resource recommendations for students based on their performance and interests.
