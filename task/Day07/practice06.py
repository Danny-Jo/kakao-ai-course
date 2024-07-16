# 기본 라이브러리
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

weather_data = pd.read_csv('weather_classification_data.csv')

# 박스플롯 생성
plt.figure(figsize=(10, 6))
sns.boxplot(data=weather_data, y='Temperature')
plt.title('Box Plot of Temperature')
plt.ylabel('Temperature (°C)')
plt.show()