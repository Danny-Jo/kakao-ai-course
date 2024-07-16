import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 데이터 정의
data = {'Year': [2018, 2019, 2020, 2021], 'Visitors': [12000, 18000, 2300, 30000]}
df = pd.DataFrame(data)

# 선그래프 생성
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
sns.lineplot(x='Year', y='Visitors', data=df, marker='o', label='Visitors')
plt.title('Annual Website Visitors')  # 제목 추가
plt.xlabel('Year')  # x축 레이블 추가
plt.ylabel('Number of Visitors')  # y축 레이블 추가
plt.legend()  # 범례 추가
plt.grid(True)  # 그리드 추가
plt.show()