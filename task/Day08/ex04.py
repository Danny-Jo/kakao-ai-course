import matplotlib.pyplot as plt
import numpy as np

# 샘플 데이터 정의
labels = ['Shots', 'Speeds', 'Touches', 'Dribbles', 'Heading']
values = [4, 3, 2, 5, 4]
values += values[:1]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

# 레이더 차트 생성
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='red', alpha=0.4)
ax.plot(angles, values, color='black', linewidth=2)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.title('Football Player A')
plt.show()