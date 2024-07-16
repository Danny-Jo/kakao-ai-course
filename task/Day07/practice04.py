# 기본 라이브러리
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

weather_data = pd.read_csv('weather_classification_data.csv')

# 파이차트 생성
plt.figure(figsize=(8, 8))
weather_counts = weather_data['Weather Type'].value_counts()
plt.pie(weather_counts, labels=weather_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Weather Type Distribution')
plt.show()