import matplotlib.pyplot as plt

# 데이터 정의
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales = [150, 180, 175, 200, 190, 210]

# 그래프 생성 (나쁜 예시)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.bar(months, sales, zs=0, zdir='y', alpha=0.8)
ax.set_title('Monthly Sales (Confusing 3D Effect)')
ax.set_xlabel('Month')
ax.set_ylabel('Sales')
plt.show()