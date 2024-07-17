import matplotlib.pyplot as plt

# 데이터 정의
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
returns = [5, 4.5, 4, 3.8, 4.2, 3.5]

# 그래프 생성 (나쁜 예시)
plt.figure(figsize=(10, 6))
plt.bar(months, returns, color='red')
plt.title('Monthly Returns Rate (Distorted)')
plt.xlabel('Month')
plt.ylabel('Returns (%)')
plt.ylim(3.5, 5)
plt.show()