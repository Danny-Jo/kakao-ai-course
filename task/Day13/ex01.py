import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 데이터셋 불러오기 및 전처리
transform = transforms.Compose([transforms.ToTensor(),  # 데이터를 텐서로 변환
                                transforms.Normalize((0.5,), (0.5,))])  # 데이터를 0.5의 평균과 0.5의 표준편차로 정규화

# 훈련 데이터셋과 테스트 데이터셋 다운로드 및 로드
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 데이터로더를 사용하여 데이터셋을 배치 단위로 로드
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(train_dataset[0])

# 신경망 모델 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 입력층: 28x28 이미지를 1차원 배열로 변환, 128개의 뉴런
        self.fc2 = nn.Linear(128, 10)  # 출력층: 10개의 뉴런 (0~9 클래스)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 이미지를 1차원 배열로 변환
        x = F.relu(self.fc1(x))  # 은닉층: ReLU 활성화 함수 적용
        x = self.fc2(x)  # 출력층
        return F.log_softmax(x, dim=1)  # 소프트맥스 함수로 클래스 확률 반환

# 모델 초기화
model = SimpleNN()

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()  # 교차 엔트로피 손실 함수
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD 옵티마이저

print(model)

# 모델 학습 함수 정의
def train(model, device, train_loader, optimizer, epoch):
    model.train()  # 모델을 학습 모드로 설정
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 데이터를 장치로 이동
        optimizer.zero_grad()  # 이전 기울기 초기화
        output = model(data)  # 모델 예측
        loss = criterion(output, target)  # 손실 계산
        loss.backward()  # 역전파를 통해 기울기 계산
        optimizer.step()  # 가중치 업데이트
        if batch_idx % 100 == 0:  # 100번째 배치마다 로그 출력
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# 모델 평가 함수 정의
def test(model, device, test_loader):
    model.eval()  # 모델을 평가 모드로 설정
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 평가 시에는 기울기를 계산하지 않음
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 손실 합산
            pred = output.argmax(dim=1, keepdim=True)  # 가장 높은 확률을 가진 클래스 예측
            correct += pred.eq(target.view_as(pred)).sum().item()  # 맞춘 개수 합산

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({test_accuracy:.0f}%)\n')
    return test_loss, test_accuracy

# 학습 및 평가
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 1
train_losses, test_losses, test_accuracies = [], [], []

print(device)

# 모델 학습
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test_loss, test_accuracy = test(model, device, test_loader)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

from torchsummary import summary
summary(model, input_size=(1, 28, 28))