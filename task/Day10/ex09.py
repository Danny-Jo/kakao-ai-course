# 결정 트리 예제

# 필요한 라이브러리 임포트
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 와인 데이터셋 로드
wine = load_wine()
X, y = wine.data, wine.target

# 데이터셋을 학습 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 결정 트리 모델 학습
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# 결정 트리 구조 시각화
plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=wine.feature_names, class_names=wine.target_names)
plt.title("Decision Tree Structure")
plt.show()