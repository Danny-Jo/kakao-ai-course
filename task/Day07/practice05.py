# 기본 라이브러리
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

weather_data = pd.read_csv('weather_classification_data.csv')

# 산점도 생성
plt.figure(figsize=(10, 6))
plt.scatter(weather_data['Humidity'], weather_data['Temperature'], alpha=0.7)
plt.title('Temperature vs Humidity')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.grid(True)
plt.show()