import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 데이터 생성
np.random.seed(10)
data = np.random.randn(1000)

# 히스토그램 생성
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Random Data')  # 제목 추가
plt.xlabel('Value')  # x축 레이블 추가
plt.ylabel('Frequency')  # y축 레이블 추가
plt.grid(True)  # 그리드 추가
plt.show()