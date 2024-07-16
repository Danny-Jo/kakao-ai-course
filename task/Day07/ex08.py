import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 데이터 생성
np.random.seed(10)
data = {
    'Category': np.random.choice(['A', 'B', 'C'], 100),
    'Value': np.random.randn(100)
}
df = pd.DataFrame(data)

print(df.head())

# 박스플롯 생성
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
sns.boxplot(x='Category', y='Value', data=df)
plt.title('Box Plot of Random Data by Category')  # 제목 추가
plt.xlabel('Category')  # x축 레이블 추가
plt.ylabel('Value')  # y축 레이블 추가
plt.grid(True)  # 그리드 추가
plt.show()