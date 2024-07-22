import warnings
warnings.filterwarnings("ignore")

# 다층 퍼셉트론 (Multilayer Perceptron) 예제

# 필요한 라이브러리 임포트
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# MNIST 데이터셋 로드
mnist = fetch_openml('mnist_784')
X, y = mnist.data / 255., mnist.target

# 데이터셋을 학습 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 다층 퍼셉트론 모델 학습
mlp = MLPClassifier(hidden_layer_sizes=(30,), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=42,
                    learning_rate_init=0.1)
mlp.fit(X_train, y_train)

# 예측 및 평가
y_pred = mlp.predict(X_test)
print(classification_report(y_test, y_pred))

# 혼동 행렬 시각화
ConfusionMatrixDisplay.from_estimator(mlp, X_test, y_test)
plt.title("MLP Confusion Matrix")
plt.show()