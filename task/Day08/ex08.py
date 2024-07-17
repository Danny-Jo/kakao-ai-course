import matplotlib.pyplot as plt

# 데이터 정의
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 20, 25]

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.bar(categories, values, color='grey')
plt.title('Category Values')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()