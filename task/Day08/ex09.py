import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 데이터 정의
fig, ax = plt.subplots()
x = np.arange(0, 10, 0.1)
line, = ax.plot(x, np.sin(x))

# 애니메이션 함수
def update(num, x, line):
    line.set_ydata(np.sin(x + num / 10.0))
    return line,

ani = animation.FuncAnimation(fig, update, frames=100, fargs=[x, line], interval=50)
plt.title('Sine Wave Animation')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()