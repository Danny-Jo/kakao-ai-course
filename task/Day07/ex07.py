import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 데이터 생성
np.random.seed(10)
data = {
    'x': np.random.randn(100),
    'y': np.random.randn(100)
}
df = pd.DataFrame(data)

# 산점도 생성
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
sns.scatterplot(x='x', y='y', data=df)
plt.title('Scatter Plot of Random Data')  # 제목 추가
plt.xlabel('X')  # x축 레이블 추가
plt.ylabel('Y')  # y축 레이블 추가
plt.grid(True)  # 그리드 추가
plt.show()