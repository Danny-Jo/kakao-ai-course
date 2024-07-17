import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의
x = np.arange(0, 10, 0.1)
y = np.sin(x)

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



