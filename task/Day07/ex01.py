import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 데이터 정의
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales = [150, 200, 180, 220, 250, 230]

# 막대그래프 생성
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
plt.bar(months, sales, color='skyblue', label='Sales')  # 막대그래프 생성
plt.title('Monthly Sales')  # 제목 추가
plt.xlabel('Month')  # x축 레이블 추가
plt.ylabel('Sales')  # y축 레이블 추가
plt.legend()  # 범례 추가
plt.grid(True)  # 그리드 추가
plt.show()  # 그래프 출력