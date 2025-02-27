# 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import requests
dataset_file_origin = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
# 요청을 보내고 응답을 받기
response = requests.get(dataset_file_origin)

# 응답이 성공적이면 파일로 저장
if response.status_code == 200:
    # 파일을 열고 데이터 쓰기
    with open("./shakespeare.txt", "w", encoding="utf-8") as file:
        file.write(response.text)
    print("파일이 성공적으로 다운로드되었습니다.")
else:
    print(f"파일 다운로드 실패: 상태 코드 {response.status_code}")

text = ""
with open("shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()
    print(text[:100])

# 데이터 전처리
chars = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# 하이퍼파라미터 설정
vocab_size = len(chars)
hidden_size = 256
output_size = len(chars)
num_layers = 2

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 시퀀스 길이 설정
    seq_length = 100

    # 시퀀스 데이터 생성
    sequences, targets = create_sequences(text, seq_length)

    print(sequences[:10])

    # 데이터셋 및 데이터로더 인스턴스 생성
    dataset = TextDataset(sequences, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    
    # 모델 인스턴스 생성 및 GPU로 이동
    model = RNNModel(vocab_size, hidden_size, output_size, num_layers).to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(model)

    # 모델 훈련
    train_model(model, dataloader, criterion, optimizer, num_epochs=5)

    # 테스트 시작 문자열 및 생성할 텍스트 길이
    start_str = "To be, or not to be, that is the question:"
    length = 200

    # 텍스트 생성
    generated_text = generate_text(model, start_str, length, temperature=0.8)
    print(generated_text)

# 시퀀스 데이터 생성 함수 정의
def create_sequences(text, seq_length):
    sequences = []
    targets = []
    for i in range(0, len(text) - seq_length):
        seq = text[i:i+seq_length]   # 시퀀스 생성
        target = text[i+seq_length]  # 시퀀스 다음에 오는 문자
        sequences.append([char_to_idx[char] for char in seq])
        targets.append(char_to_idx[target])
    return sequences, targets

# PyTorch 데이터셋 및 데이터로더 생성
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])


# RNN 모델을 사용한 클래스 정의
class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(vocab_size, hidden_size, num_layers, batch_first=True)  # RNN 레이어
        self.fc = nn.Linear(hidden_size, output_size)  # 완전 연결층

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)  # RNN 순전파
        out = self.fc(out[:, -1, :])  # 마지막 시퀀스 출력만 사용
        return out, hidden

    def init_hidden(self, batch_size):
        # 초기 hidden state 설정
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

# 모델 훈련 함수
def train_model(model, dataloader, criterion, optimizer, num_epochs=20):
    model.train()  # 모델을 훈련 모드로 설정
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = nn.functional.one_hot(inputs, num_classes=vocab_size).float().to(device)  # 원-핫 인코딩 및 GPU로 이동
            labels = labels.to(device)

            hidden = model.init_hidden(inputs.size(0))  # 각 배치마다 hidden 상태 초기화

            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 순전파, 역전파, 최적화
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # hidden 상태를 detach하여 그래프의 연결을 끊음
            hidden = hidden.detach()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}')

    print('Finished Training')

def generate_text(model, start_str, length, temperature=1.0):
    model.eval()  # 모델을 평가 모드로 설정
    hidden = model.init_hidden(1)  # 초기 hidden 상태 설정

    input_seq = torch.tensor([char_to_idx[char] for char in start_str]).unsqueeze(0).to(device)
    generated_text = start_str

    with torch.no_grad():
        for _ in range(length):
            input_seq = nn.functional.one_hot(input_seq, num_classes=vocab_size).float()
            output, hidden = model(input_seq, hidden)

            # 다음 문자를 샘플링
            output = output.squeeze().div(temperature).exp().cpu()
            top_char = torch.multinomial(output, 1)[0]

            generated_char = idx_to_char[top_char.item()]
            generated_text += generated_char

            # 다음 입력 시퀀스 준비
            input_seq = torch.tensor([[top_char]]).to(device)

    return generated_text

if __name__ == '__main__':
    main()