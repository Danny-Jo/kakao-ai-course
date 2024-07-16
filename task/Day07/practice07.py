# 기본 라이브러리
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

weather_data = pd.read_csv('weather_classification_data.csv')


# 여러 시각화 차트 생성
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# 히스토그램
axs[0, 0].hist(weather_data['Temperature'], bins=10, edgecolor='k')
axs[0, 0].set_title('Histogram of Temperature')
axs[0, 0].set_xlabel('Temperature (°C)')
axs[0, 0].set_ylabel('Frequency')

# 산점도
axs[0, 1].scatter(weather_data['Temperature'], weather_data['Humidity'], alpha=0.7)
axs[0, 1].set_title('Temperature vs Humidity')
axs[0, 1].set_xlabel('Temperature (°C)')
axs[0, 1].set_ylabel('Humidity (%)')

# 박스플롯
sns.boxplot(data=weather_data, y='Temperature', ax=axs[1, 0])
axs[1, 0].set_title('Box Plot of Temperature')
axs[1, 0].set_ylabel('Temperature (°C)')

# 선그래프
axs[1, 1].plot(weather_data.index, weather_data['Temperature'], marker='o')
axs[1, 1].set_title('Temperature Over Index')
axs[1, 1].set_xlabel('Index')
axs[1, 1].set_ylabel('Temperature (°C)')

plt.tight_layout()
plt.show()
