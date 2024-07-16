# 기본 라이브러리
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

weather_data = pd.read_csv('weather_classification_data.csv')

# 계절별로 데이터 그룹화
seasons = weather_data['Season'].unique()
colors = ['blue', 'green', 'orange', 'red']
mid_temp = (weather_data['Temperature'].min() + weather_data['Temperature'].max()) / 2
temp_range = 100

# 서브플롯 생성
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# 각 계절별 그래프 그리기
for ax, season, color in zip(axs.ravel(), seasons, colors):
    season_data = weather_data[weather_data['Season'] == season]
    ax.plot(season_data.index, season_data['Temperature'], marker='o', color=color)
    ax.set_title(f'Temperature Over {season}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Temperature (°C)')
    ax.set_ylim(mid_temp - temp_range, mid_temp + temp_range)
    ax.grid(True)

plt.tight_layout()
plt.show()