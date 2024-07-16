import pandas as pd
from scipy import stats

# * 데이터셋 로드
data = pd.read_csv('HeartDiseaseTrain-Test.csv')

print('#'*30)
print("데이터의 처음 몇 줄을 출력하여 구조 확인")
print(data.head())

print('#'*30)
print('데이터의 각 컬럼에 대한 정보 확인')
print(data.info())

print('#'*30)
print("데이터 타입 확인")
print('Data Types:\n ', data.dtypes)


# *  범주형과 수치형 데이터를 분리하여 분석
categorical_cols = data.select_dtypes(include=['object', 'category']).columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

print('#'*30)
print("범주형 데이터")
print("\nCategorical Columns: \n", categorical_cols)

print('#'*30)
print("수치형 데이터")
print("\nNumerical Columns: \n", numerical_cols)

# * 결측치 확인
print("\nMissing Values: \n", data.isnull().sum())

for col in categorical_cols:
    print(f'\nUnique Values in {col}: \n', data[col].value_counts())

print("\nDescriptive Statistics for Numerical Data: \n", data[numerical_cols].describe())

# * 왜도와 첨도 확인
# ? 왜도(Skewness): 0에 가까울수록 정규분포에 근사, 양의 값은 오른쪽 꼬리가 긴 분포(왼쪽으로 치우친), 음의 값은 왼쪽 고리가 긴 분포(오른쪽으로 치우친)
# ? 첨도(Kurtosis): 0에 가까울수록 정규분포에 근사, 높으면 분포가 뾰족하고, 낮으면 평평

print('\nSkewness of the data: \n', data[numerical_cols].skew())
print('\nKurtosis of the data: \n', data[numerical_cols].kurt())

# * 상관계수 값이 1에 가까울수록 완벽한 양의 상관관계, -1에 가까울수록 완벽한 음의 상관관계를 나타냅니다.
# * 피어슨 상관 계수
print('\nPearson Correlation: \n ', data[numerical_cols].corr(method='pearson'))
# * 스피어만 상관 계수
print('\nSpearman Correlation: \n ', data[numerical_cols].corr(method='spearman'))


# * T-통계량의 절대값이 크면 클수록 두 그룹 간의 차이가 크다고 할 수 있다.
# * 일반적으로 P-값이 0.05보다 작으면 귀무 가설을 기각하고, 통계적으로 유의미한 차이가 있음을 인정합니다.
# * 귀무 가설이란 주어진 데이터에 대해 이미 알려진 사실을 반박하기 위해 만들어진 가설이다.

male_max_hr = data[data['sex'] == 'Male']['Max_heart_rate']
female_max_hr = data[data['sex'] == 'Female']['Max_heart_rate']

t_stat, p_val = stats.ttest_ind(male_max_hr, female_max_hr)
print(f'T-statistic: {t_stat}, P-value: {p_val}')

print(male_max_hr)