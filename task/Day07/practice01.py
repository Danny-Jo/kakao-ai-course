# 기본 라이브러리
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# kaggle datasets download -d nikhil7280/weather-type-classification

weather_data = pd.read_csv('weather_classification_data.csv')

print(weather_data.dtypes)
print(weather_data.tail())

# * 1. 막대 그래프 (Bar Plot)
plt.figure(figsize=(10, 6))
sns.countplot(data=weather_data, x='Season')
plt.title('Count of Seasons')
plt.xlabel("Season")
plt.ylabel("Count")
plt.show()

