```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
ratings = pd.read_csv("ratings.csv")  # userId, movieId, rating
movies = pd.read_csv("movies.csv")    # movieId, title, genres
ratings.head()
movies.head()
```

```python
ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
ratings_matrix.head()
```

```python
user_similarity = cosine_similarity(ratings_matrix)
item_similarity = cosine_similarity(ratings_matrix.T)
```

```python
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)
item_similarity_df = pd.DataFrame(item_similarity, index=ratings_matrix.columns, columns=ratings_matrix.columns)
```

```python
def predict_ratings_user_based(ratings, similarity):
    mean_user_rating = ratings.mean(axis=1).values.reshape(-1,1)
    ratings_diff = (ratings - mean_user_rating)
    pred = mean_user_rating + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pred

user_pred = predict_ratings_user_based(ratings_matrix.values, user_similarity)
```

```python
def predict_ratings_item_based(ratings, similarity):
    pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_pred = predict_ratings_item_based(ratings_matrix.values, item_similarity)
```

```python
def rmse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(pred, actual))

print("User-based CF RMSE: ", rmse(user_pred, ratings_matrix.values))
print("Item-based CF RMSE: ", rmse(item_pred, ratings_matrix.values))
```

```python
user_pred_df = pd.DataFrame(user_pred, index=ratings_matrix.index, columns=ratings_matrix.columns)
```

```python
def recommend_movies(user_id, ratings_pred_df, movies_df, n=5):
    user_row = ratings_pred_df.loc[user_id]
    recommended_movies = user_row.sort_values(ascending=False).head(n).index
    recommended_titles = movies_df[movies_df['movieId'].isin(recommended_movies)]['title'].values
    return recommended_titles

recommend_movies(1, user_pred_df, movies)
```

```python
movie_mean_ratings = ratings.groupby('movieId')['rating'].mean()
movie_counts = ratings.groupby('movieId')['rating'].count()
sns.histplot(movie_counts, bins=30, color='skyblue')
plt.title("Distribution of Number of Ratings per Movie")
plt.xlabel("Number of Ratings")
plt.ylabel("Count")
plt.show()
```

```python
sns.histplot(movie_mean_ratings, bins=30, color='salmon')
plt.title("Distribution of Average Movie Ratings")
plt.xlabel("Average Rating")
plt.ylabel("Count")
plt.show()
```

```python
def hybrid_recommendation(user_id, ratings_pred_df, movies_df, alpha=0.7, n=5):
    user_row = ratings_pred_df.loc[user_id]
    movie_score = user_row * alpha + movie_mean_ratings * (1-alpha)
    top_movies = movie_score.sort_values(ascending=False).head(n).index
    return movies_df[movies_df['movieId'].isin(top_movies)]['title'].values

hybrid_recommendation(1, user_pred_df, movies)
```
