# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 분할된 CSV 파일 로드 및 병합
file_paths = ['creditcard_part1.csv', 'creditcard_part2.csv', 'creditcard_part3.csv']
data = pd.concat([pd.read_csv(file) for file in file_paths])

# 필요한 특성 선택
X = data.drop(['Time', 'Class'], axis=1)

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-평균 클러스터링 모델 생성 및 학습
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# 클러스터링 결과 시각화 (PCA를 사용하여 2차원으로 축소)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering of Credit Card Fraud Dataset')
plt.show()
