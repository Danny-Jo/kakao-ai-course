# pandas, numpy 라이브러리 사용
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# 범주형 데이터 인코딩 (레이블 인코딩), 데이터 스케일링
from sklearn.preprocessing import LabelEncoder, StandardScaler
# PCA (주성분 분석)
from sklearn.decomposition import PCA
# 전처리 작업의 자동화 (파이프라인 구축)
from sklearn.pipeline import Pipeline  # 파이프라인 구축을 위한 클래스
from sklearn.impute import SimpleImputer  # 결측값 처리
from sklearn.preprocessing import StandardScaler  # 데이터 스케일링

# CSV load
data = pd.read_csv('customer_purchase_data.csv')

# 데이터 구조 확인
data.info()
# 데이터 샘플 확인
data.head(5)
# 데이터 기초통계 확인
data.describe()

# 결측값 확인
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# 결측값 변환 함수
def add_missing_values(df, col_name, missing_frac):
    """
    특정 열에 지정된 비율의 결측값을 추가합니다.

    :param df: 데이터프레임
    :param col_name: 결측값을 추가할 열 이름
    :param missing_frac: 결측값 비율 (0.0 ~ 1.0)
    """
    np.random.seed(42)  # 재현성을 위해 랜덤 시드 설정
    n_rows = df.shape[0] # 데이터프레임의 행 수
    n_missing = int(n_rows * missing_frac) # 결측값을 추가할 행의 수를 계산

    missing_indices = np.random.choice(n_rows, n_missing, replace=False) # 결측값을 추가할 행의 인덱스를 무작위로 선택
    df.loc[missing_indices, col_name] = np.nan # 선택된 인덱스의 열 값을 NaN으로 설정


# 결측값 추가
add_missing_values(data, 'Age', 0.1)  # Age 열에 10% 결측값 추가
add_missing_values(data, 'Gender', 0.1)  # Gender 열에 10% 결측값 추가

# 결측값이 잘 추가되었는지 확인
print("Data with missing values:\n", data.tail(10))
print("Missing values count:\n", data.isnull().sum())

data['Age'].mean()

# 숫자 데이터 전처리 - 결측값 처리 (평균값으로 대체)
data['Age'].fillna(data['Age'].mean(), inplace=True)

print("Missing values count:\n", data.isnull().sum())

data.tail(10)

data['Gender'].mode()[0]

# 범주형 데이터 전처리 - 결측값 처리 (최빈값으로 대체)
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)

print("Missing values count:\n", data.isnull().sum())

data['Gender'].head()

data['Gender'] = ['Male' if gender == 0 else 'Female' for gender in data['Gender']]
data['Gender'].head()


le = LabelEncoder() # 객체 생성
data['Gender'] = le.fit_transform(data['Gender']) # 레이블 인코딩 (Alphabetic)

data['Gender'].head()

# 현재 시간
current_time = datetime.now()

# 예시 데이터 생성
time_data = pd.DataFrame({
    'PurchaseTime': [current_time - timedelta(days=i) for i in range(5)],
    'DeliveryTime': [current_time + timedelta(days=i) for i in range(5)]
})

print("Original Data:\n", time_data)

# 시간 데이터를 datetime 형식으로 변환 (이미 datetime)
time_data['PurchaseTime'] = pd.to_datetime(time_data['PurchaseTime'])
time_data['DeliveryTime'] = pd.to_datetime(time_data['DeliveryTime'])

# 시간 컴포넌트 추출
time_data['PurchaseYear'] = time_data['PurchaseTime'].dt.year
time_data['PurchaseMonth'] = time_data['PurchaseTime'].dt.month
time_data['PurchaseDay'] = time_data['PurchaseTime'].dt.day
time_data['PurchaseHour'] = time_data['PurchaseTime'].dt.hour
time_data['PurchaseMinute'] = time_data['PurchaseTime'].dt.minute
time_data['PurchaseSecond'] = time_data['PurchaseTime'].dt.second

# 시간 차이 계산
time_data['DeliveryDays'] = (time_data['DeliveryTime'] - time_data['PurchaseTime']).dt.days

print("\nData after extracting components and calculating differences:\n", time_data)

print("Data before scaling:\n", data[['Age', 'AnnualIncome']].head())

scaler = StandardScaler() # 객체 생성
data[['Age', 'AnnualIncome']] = scaler.fit_transform(data[['Age', 'AnnualIncome']]) # 스케일링

print("Data after scaling:\n", data[['Age', 'AnnualIncome']].head())

print("Original table:\n", data.head())
# 데이터 피벗
pivot_table = data.pivot_table(values='AnnualIncome', index='ProductCategory', columns='PurchaseStatus', aggfunc='mean') # aggfunc: 집계함수
print("Pivot table:\n", pivot_table.head())

# 데이터 합치기
# 예제 데이터프레임 생성
data_1 = data[['Age', 'PurchaseStatus', 'AnnualIncome']]
# data_1 데이터 형태
print("Original data_1:\n", data_1.head())
data_2 = data[['Gender', 'AnnualIncome']]
# data_2 데이터 형태
print("Original data_2:\n", data_2.head())
merged_data = pd.merge(data_1, data_2, on='AnnualIncome')
print("Merged data:\n", merged_data.head())

# 데이터 그룹화
grouped_data = data.groupby('Gender').agg({'AnnualIncome': 'mean', 'Age': 'mean'})
print("Grouped data:\n", grouped_data)

data.head()

# 파생 변수 생성
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 18, 35, 50, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
print("Data with AgeGroup:\n", data[['Age', 'AgeGroup']].head())

# 데이터 샘플링
sampled_data = data.sample(frac=0.1, random_state=42)
print("Sampled data:\n", sampled_data.head())
print("*"*30)
print("Sampled data info:\n", sampled_data.info())

# PCA 적용
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data[['Age', 'AnnualIncome', 'NumberOfPurchases']])

# 결과를 데이터프레임으로 변환
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

print("PCA Result:\n", pca_df.head())

# 예제를 위한 데이터 분리
data_1 = data[['Age', 'AnnualIncome', 'NumberOfPurchases']]

# 파이프라인 구축
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # 결측값 대체
    ('scaler', StandardScaler())  # 데이터 스케일링
])

# 데이터 변환
processed_data = pipeline.fit_transform(data_1)

# 결과를 데이터프레임으로 변환
processed_data_df = pd.DataFrame(processed_data, columns=['Age', 'AnnualIncome', 'NumberOfPurchases'])

print("Processed Data:\n", processed_data_df.head())