# 주성분 분석 (PCA) 예제

# 필요한 라이브러리 임포트
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 데이터셋 로드
file_paths = ['creditcard_part1.csv', 'creditcard_part2.csv', 'creditcard_part3.csv']
data = pd.concat([pd.read_csv(file) for file in file_paths])

# 필요한 특성 선택
X = data.drop(['Time', 'Class'], axis=1)
y = data['Class']

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 적용
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA 결과 시각화
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Credit Card Fraud Dataset')
plt.show()
