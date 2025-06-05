from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# 평가 지표 계산 함수
def compute_sequence_metrics(pred_orders, target_orders):
    y_true = []
    y_pred = []

    if isinstance(pred_orders[0], list):
        for gt, pred in zip(target_orders, pred_orders):
            if len(gt) != len(pred):
                continue
            y_true.extend(gt)
            y_pred.extend(pred)
    else:
        y_true = target_orders
        y_pred = pred_orders

    # Macro-average : 클래스별 성능 지표를 각각 계산 후 평균 -> 클래스별 데이터셋이 균등하게 분포되어 있을 때 적절
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    return f1, precision, recall



class Dataset_Module:
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

        df = pd.read_csv(f'../data/{self.file_name}')
        X = df.drop(columns=["churn", "remaining_contract", "is_tv_subscriber", "is_movie_package_subscriber"], axis=1)
        y = df['churn']

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, random_state=42, test_size=0.2,
                                                                              stratify=y)

    def get_train_dataset(self):
        train_dataset = Train_Dataset(self.X_train, self.y_train)
        val_dataset = Train_Dataset(self.X_val, self.y_val)

        return train_dataset, val_dataset


# PyTorch용 Dataset 정의
class Train_Dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    # getter
    def __getitem__(self, idx):
        data = self.data.iloc[idx].to_numpy() # Pandas Series -> Numpy 배열
        label = self.label.iloc[idx] # int 형식의 값들 이므로 Numpy 배열 변환 X

        return data, label



# 다층 퍼셉트론 Multi-Layer Perceptron 신경망 정의
class MLPNet(nn.Module):
    def __init__(self, Cin):
        super().__init__()
        self.Cin = Cin

        self.layers = nn.Sequential(
            nn.Linear(Cin, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 이진분류 출력 노드 1개
        )

    def forward(self, x):
        return self.layers(x)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = Dataset_Module(file_name='train.csv')
    train_dataset, val_dataset = dataset.get_train_dataset()

    # 입력 특성 수 확인
    Cin = train_dataset.__getitem__(0)[0].shape
    print(f"num Feature : {Cin}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = MLPNet(Cin[0])
    criterion = nn.BCEWithLogitsLoss() # 이진분류 손실함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    train_epoch = 100
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 최고 성능 기록 위한 dict
    best_metrics = {'f1': 0, 'precision': 0, 'recall': 0, 'accuracy': 0}

    for epoch in range(train_epoch):
        # 학습
        model.train()
        train_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"[Epoch {epoch + 1}/{train_epoch}] Training")

        for batch_data, batch_label in train_iter:
            batch_data, batch_label = batch_data.to(device).float(), batch_label.to(device).float().unsqueeze(-1)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iter.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 검증
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_data, batch_label in tqdm(val_loader, desc=f"[Epoch {epoch + 1}/{train_epoch}] Validation"):
                batch_data, batch_label = batch_data.to(device).float(), batch_label.to(device).float().unsqueeze(-1)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_label)
                val_loss += loss.item()
                outputs = F.sigmoid(outputs)
                preds = (outputs > 0.5).float()

                correct += (preds == batch_label).sum().item()
                total += batch_label.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_label.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        # 평가지표 계산
        val_f1, val_precision, val_recall = compute_sequence_metrics(all_preds, all_labels)

        print(f"Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {val_accuracy:.4f}%")
        print(f"Epoch {epoch + 1} - F1 : {val_f1:.4f}, Precision : {val_precision:.4f}, Recall : {val_recall:.4f}")

        # 최고 성능 갱신 시 모델 저장
        if val_accuracy > best_metrics['accuracy']:
            best_metrics['f1'] = val_f1
            best_metrics['precision'] = val_precision
            best_metrics['recall'] = val_recall
            best_metrics['accuracy'] = val_accuracy
            torch.save(model.state_dict(), '../best_model.pt')  # 모델 파라미터 저장

    # 최종능 평가 지표 출력
    print("\n=== 최종 평가 지표 (Validation Best Accuracy 기준) ===")
    print(f"F1 Score     : {best_metrics['f1']:.4f}")
    print(f"Precision    : {best_metrics['precision']:.4f}")
    print(f"Recall       : {best_metrics['recall']:.4f}")
    print(f"Accuracy (%) : {best_metrics['accuracy']:.2f}%")



if __name__ == '__main__':
    main()


# === 최종 평가 지표 (Validation Best Accuracy 기준) ===
# F1 Score     : 0.7327
# Precision    : 0.7386
# Recall       : 0.7308
# Accuracy (%) : 73.95%