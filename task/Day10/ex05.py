# 서포트 벡터 머신 (SVM) 예제

# 필요한 라이브러리 임포트
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 와인 데이터셋 로드
wine = load_wine()
X, y = wine.data, wine.target

# 데이터셋을 학습 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM 모델 학습
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# 예측 및 평가
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))