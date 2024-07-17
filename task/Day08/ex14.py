import matplotlib.pyplot as plt

# 데이터 정의
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales = [150, 180, 175, 200, 190, 210]
customers = [3000, 3500, 3400, 3700, 3600, 3900]

# 그래프 생성 (나쁜 예시)
plt.figure(figsize=(10, 6))
plt.plot(months, sales, marker='o', label='Sales', color='blue')
plt.plot(months, customers, marker='x', label='Customers', color='green')
plt.title('Monthly Sales and Customers (Too Complex)')
plt.xlabel('Month')
plt.ylabel('Value')
plt.legend()
plt.show()