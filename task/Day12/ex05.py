# 필요한 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 그리드월드 환경 정의
class GridWorld:
    def __init__(self, size):
        # 그리드의 크기를 설정합니다.
        self.size = size
        # 초기 상태를 (0, 0)으로 설정합니다.
        self.state = (0, 0)
        # 목표 상태를 그리드의 오른쪽 아래 모서리로 설정합니다.
        self.goal = (size-1, size-1)

    def reset(self):
        # 상태를 초기 상태로 리셋합니다.
        self.state = (0, 0)
        return self.state

    def step(self, action):
        # 현재 상태의 x, y 좌표를 가져옵니다.
        x, y = self.state
        # 행동에 따라 새로운 상태를 결정합니다.
        if action == 0:
            x = max(0, x - 1)  # 위로 이동
        elif action == 1:
            x = min(self.size - 1, x + 1)  # 아래로 이동
        elif action == 2:
            y = max(0, y - 1)  # 왼쪽으로 이동
        elif action == 3:
            y = min(self.size - 1, y + 1)  # 오른쪽으로 이동

        # 새로운 상태를 설정합니다.
        self.state = (x, y)
        # 새로운 상태가 목표 상태인지 확인합니다.
        reward = 1 if self.state == self.goal else -0.1
        done = self.state == self.goal
        # 새로운 상태, 보상, 완료 여부를 반환합니다.
        return self.state, reward, done

# Q-learning 파라미터 설정
size = 5  # 그리드의 크기
env = GridWorld(size)  # 그리드월드 환경 생성
q_table = np.zeros((size, size, 4))  # Q-테이블 초기화 (상태-행동 가치 함수)
alpha = 0.1  # 학습률
gamma = 0.9  # 할인 인자
epsilon = 0.1  # 탐험 확률
episodes = 1000  # 학습 에피소드 수

# Q-learning 알고리즘
for episode in range(episodes):
    state = env.reset()  # 에피소드 시작 시 상태를 초기화
    done = False  # 에피소드가 끝났는지 여부

    while not done:
        if np.random.rand() < epsilon:
            action = np.random.choice(4)  # 탐험: 무작위로 행동 선택
        else:
            action = np.argmax(q_table[state[0], state[1]])  # 활용: Q-값이 최대인 행동 선택

        next_state, reward, done = env.step(action)  # 환경에서 행동 수행
        q_value = q_table[state[0], state[1], action]  # 현재 상태의 Q-값
        best_next_q_value = np.max(q_table[next_state[0], next_state[1]])  # 다음 상태에서의 최대 Q-값

        # Q-테이블 업데이트
        q_table[state[0], state[1], action] = q_value + alpha * (reward + gamma * best_next_q_value - q_value)

        state = next_state  # 상태 업데이트

# Q-테이블 시각화
# Q-learning 알고리즘에서 사용하는 상태-행동 가치 함수(State-Action Value Function)를 저장하는 테이블
# Q-테이블의 각 항목은 특정 상태에서 특정 행동을 취했을 때의 기대 보상을 나타냄
plt.figure(figsize=(10, 7))
sns.heatmap(np.max(q_table, axis=2), annot=True, cmap='viridis')
plt.title('Q-Table')
plt.xlabel('State (y)')
plt.ylabel('State (x)')
plt.show()
