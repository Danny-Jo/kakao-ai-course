import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 데이터 정의
data = {'Year': [2018, 2019, 2020, 2021],
        'Product A': [1500, 2000, 2400, 2800],
        'Product B': [2300, 2700, 3000, 3500],
        'Product C': [1800, 2200, 2600, 3000]}
df = pd.DataFrame(data)

# 복합 차트 생성
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
sns.lineplot(x='Year', y='Product A', data=df, marker='o', label='Product A')
sns.lineplot(x='Year', y='Product B', data=df, marker='o', label='Product B')
sns.lineplot(x='Year', y='Product C', data=df, marker='o', label='Product C')
plt.title('Annual Product Sales')  # 제목 추가
plt.xlabel('Year')  # x축 레이블 추가
plt.ylabel('Sales')  # y축 레이블 추가
plt.legend()  # 범례 추가
plt.grid(True)  # 그리드 추가
plt.show()