import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 샘플 데이터 생성
data = np.random.rand(10, 12)

# 히트맵 생성
plt.figure(figsize=(10, 8))
sns.heatmap(data, annot=True, cmap='coolwarm')
plt.title('Sample Heatmap')
plt.show()