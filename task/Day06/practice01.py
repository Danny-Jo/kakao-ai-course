import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# * 데이터셋 로드
data = pd.read_csv('weather_classification_data.csv')

# * 데이터 분석 한 번 해보기
# * 어떤 인사이트를 얻을 수 있을까?, 어떤 서비스를 개발해볼 수 있을까?

print('#'*30)
print("데이터의 처음 몇 줄을 출력하여 구조 확인")
print(data.head())

print('#'*30)
print('데이터의 각 컬럼에 대한 정보 확인')
print(data.info())

print('#'*30)
print("데이터 타입 확인")
print('Data Types:\n ', data.dtypes)

# * 그래프 설정
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# * 기온과 습도의 관계 분석
sns.scatterplot(ax=ax1, x='Humidity', y='Temperature', data=data)
ax1.set_title('Relationship between Humidity and Temperature')

# * 기온 예측 모델
# 특성과 목표 변수 분리
features = data[['Humidity', 'Wind Speed', 'Precipitation (%)', 'Cloud Cover', 'Atmospheric Pressure', 'UV Index', 'Season', 'Visibility (km)']]
target = data['Temperature']

# 원-핫 인코딩
features = pd.get_dummies(features, columns=['Cloud Cover', 'Season'])

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_test)
predictions_last_500 = predictions[-500:]


# 두 번째 그래프: 예측된 기온

ax2.plot(predictions_last_500, marker='o', linestyle='-', color='b')
ax2.set_title('Predicted Temperatures (Last 500 Samples)')
ax2.set_xlabel('Index')
ax2.set_ylabel('Temperature')
ax2.grid(True)

# 그래프 표시
plt.tight_layout()
plt.show()