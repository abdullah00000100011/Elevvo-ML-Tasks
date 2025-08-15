import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

movies_cols = ['movie_id', 'title']
movies_path = "u.item"
movies = pd.read_csv(movies_path, sep='|', names=movies_cols, usecols=[0, 1], encoding='latin-1')

ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_path = "u.data"
ratings = pd.read_csv(ratings_path, sep='\t', names=ratings_cols, usecols=[0, 1, 2], encoding='latin-1')

user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_movies(user_id, top_n=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    recommended_movies = {}
    for other_user in similar_users:
        for movie_id, rating in user_item_matrix.loc[other_user].items():
            if user_item_matrix.loc[user_id, movie_id] == 0 and rating > 0:
                if movie_id not in recommended_movies:
                    recommended_movies[movie_id] = rating * user_similarity_df.loc[user_id, other_user]
                else:
                    recommended_movies[movie_id] += rating * user_similarity_df.loc[user_id, other_user]
    recommended_movies = dict(sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True))
    recommended_movie_ids = list(recommended_movies.keys())[:top_n]
    recommended_movie_titles = movies[movies['movie_id'].isin(recommended_movie_ids)]['title'].tolist()
    return recommended_movie_titles

user_to_recommend = 1
recommended = recommend_movies(user_to_recommend, top_n=5)
print(f"Recommended movies for user {user_to_recommend}:")
for movie in recommended:
    print(movie)
