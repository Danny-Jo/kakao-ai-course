import matplotlib.pyplot as plt

# 데이터 정의
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales = [150, 180, 175, 200, 190, 210]

# 그래프 생성 (나쁜 예시)
plt.figure(figsize=(10, 6))
plt.pie(sales, labels=months, autopct='%1.1f%%')
plt.title('Monthly Sales (Inappropriate Chart Type)')
plt.show()