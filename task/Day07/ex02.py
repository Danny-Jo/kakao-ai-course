import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 데이터 정의
labels = ['Product A', 'Product B', 'Product C', 'Product D']
sizes = [400, 300, 500, 100]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 0.1, 0, 0)  # 첫 번째 조각을 약간 떨어뜨리기

# 간결한 디자인의 파이차트
plt.figure(figsize=(8, 8))  # 그래프 크기 설정
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Sales Distribution - Simple Design')
plt.axis('equal')  # 파이차트를 원형으로 유지
plt.show()