# 기본 라이브러리
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

weather_data = pd.read_csv('weather_classification_data.csv')

plt.figure(figsize=(10, 6))
plt.hist(weather_data['Temperature'], bins=10, edgecolor='k')
plt.title('Histogram of Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.show()