import matplotlib.pyplot as plt

# 데이터 정의
years = ['2018', '2019', '2020', '2021']
sales = [15000, 18000, 22000, 25000]

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(years, sales)
plt.title('Annual Sales')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.show()
