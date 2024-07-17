import matplotlib.pyplot as plt

# Visual History

# 데이터 정의
years = ['2018', '2019', '2020', '2021']
sales = [15000, 18000, 22000, 25000]

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(years, sales, marker='o', linestyle='-', color='b')
plt.title('Annual Sales Over Time')
plt.xlabel('Year')
plt.ylabel('Sales')
for i, txt in enumerate(sales):
    plt.annotate(txt, (years[i], sales[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.show()