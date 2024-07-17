import matplotlib.pyplot as plt

# 샘플 데이터 정의
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 35]
sizes = [100, 200, 300, 400, 500]

# 버블 차트 생성
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=sizes, alpha=0.5, c=sizes, cmap='viridis')
plt.title('Sample Bubble Chart')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.colorbar()
plt.show()
