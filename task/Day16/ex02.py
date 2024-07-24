import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 임의의 데이터 생성 (예: MNIST 데이터셋 형태)
data = np.random.rand(1000, 1, 28, 28).astype(np.float32)  # 1000개의 28x28 이미지
labels = np.random.randint(0, 10, size=(1000,))  # 0에서 9 사이의 임의의 정수 레이블

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 첫 번째 합성곱 층
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 배치 정규화
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 최대 풀링
        self.dropout1 = nn.Dropout(p=0.25)  # 드롭아웃

        # 두 번째 합성곱 층
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 배치 정규화
        self.dropout2 = nn.Dropout(p=0.25)  # 드롭아웃

        # 완전 연결 층
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)  # 배치 정규화
        self.dropout3 = nn.Dropout(p=0.5)  # 드롭아웃
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        # 데이터의 크기를 확인
        print(f"Shape before flattening: {x.shape}")
        
        x = x.view(x.size(0), -1)  # 평탄화
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 모델 초기화
model_cnn = CNN()
print(model_cnn)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_cnn.parameters(), lr=0.001)

# 학습 횟수
num_epochs = 10

# 학습 루프
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cnn.to(device)

for epoch in range(num_epochs):
    model_cnn.train()  # 모델을 학습 모드로 설정
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()        # 기울기 초기화
        outputs = model_cnn(inputs)  # 모델 예측
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()              # 역전파 수행
        optimizer.step()             # 가중치 업데이트
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 평가 모드 전환
model_cnn.eval()
with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_cnn(inputs)
        # 평가 코드 작성

# GPU 사용 (Using GPU), 객체를 GPU로 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cnn.to(device)
inputs, labels = inputs.to(device), labels.to(device)
