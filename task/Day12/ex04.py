# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# 예제 사용자-아이템 매트릭스 생성
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4],
    'item_id': [1, 2, 3, 1, 3, 2, 3, 4, 4],
    'rating': [5, 4, 1, 4, 5, 2, 4, 5, 4]
}
df = pd.DataFrame(data)

# 사용자-아이템 매트릭스 생성
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

# 코사인 유사도를 사용하여 사용자 유사도 계산
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# 유사도 매트릭스 시각화
plt.figure(figsize=(10, 7))
sns.heatmap(user_similarity_df, annot=True, cmap='coolwarm')
plt.title('User Similarity Matrix')
plt.show()

# 추천 함수 정의
def recommend(user_id, user_item_matrix, user_similarity, k=2):
    # 유사한 사용자 선택
    similar_users = user_similarity[user_id - 1]
    similar_users_indices = similar_users.argsort()[-k-1:-1]

    # 유사한 사용자들의 아이템 평균 평점 계산
    similar_users_ratings = user_item_matrix.iloc[similar_users_indices]
    recommendations = similar_users_ratings.mean(axis=0)

    # 이미 평가한 아이템 제외
    user_rated_items = user_item_matrix.loc[user_id]
    recommendations = recommendations[user_rated_items == 0]

    return recommendations.sort_values(ascending=False)

# 사용자 1에게 아이템 추천
recommendations = recommend(1, user_item_matrix, user_similarity)
print("Recommendations for user 1:")
print(recommendations)