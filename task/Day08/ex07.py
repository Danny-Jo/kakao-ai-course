import matplotlib.pyplot as plt

# 데이터 정의
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 20, 25]

# 강조할 데이터
highlight = [False, False, False, True]

# 그래프 생성
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color=['grey' if not h else 'orange' for h in highlight])
plt.title('Category Values with Emphasis')
plt.xlabel('Category')
plt.ylabel('Value')

# 강조할 데이터 포인트에 텍스트 추가
for bar in bars:
    if bar.get_height() == max(values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, 'Highest Value', ha='center', color='white', weight='bold')

plt.show()