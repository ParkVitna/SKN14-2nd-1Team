from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


def compute_sequence_metrics(pred_orders, target_orders):
    
    y_true = []
    y_pred = []
    if isinstance(pred_orders[0],list):
        for gt, pred in zip(target_orders, pred_orders):
            if len(gt) != len(pred):
                continue
            y_true.extend(gt)
            y_pred.extend(pred)
    else:
        y_true = target_orders
        y_pred = pred_orders

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    return f1, precision, recall

class StandardScalerTorch:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean()
        self.std = X.std()
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class Dataset_Module:
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name
        
        df = pd.read_csv(f'./data/{self.file_name}')
        X = df.drop('churn', axis=1)
        y = df['churn']
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
        
        self.scaler = StandardScalerTorch()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
    
    def get_train_dataset(self):
        train_dataset = Train_Dataset(self.X_train, self.y_train)
        val_dataset = Train_Dataset(self.X_val, self.y_val)
        
        return train_dataset, val_dataset, self.scaler
        


class Train_Dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx].to_numpy()
        label = self.label.iloc[idx]
        
        return data, label

class MLPNet(nn.Module):
    def __init__(self, Cin):
        super().__init__()
        self.Cin = Cin
        
        self.layers = nn.Sequential(
            nn.Linear(Cin, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,1)
        )
    
    def forward(self,x):
        
        return self.layers(x)
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    dataset = Dataset_Module(file_name = 'train_2.csv')
    
    train_dataset, val_dataset, scaler = dataset.get_train_dataset()
    
    Cin = train_dataset.__getitem__(0)[0].shape
    
    print(f"num Feature : {Cin}")
    
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 32)
    
    
    model = MLPNet(Cin[0])
    
    best_acc = 0.
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    os.makedirs('./mlp_ckpt', exist_ok=True)
    
    model.to(device)
    
    train_epoch = 100
    
    for epoch in range(train_epoch):
        
        model.train()
        train_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{train_epoch}] Training")
        
        for step, (batch_data, batch_label) in enumerate(train_iter):
            
            batch_data, batch_label = batch_data.to(device).float(), batch_label.to(device).float().unsqueeze(-1)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_label)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            train_iter.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_label in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{train_epoch}] Validation"):
                
                batch_data, batch_label = batch_data.to(device).float(), batch_label.to(device).float().unsqueeze(-1)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_label)
                val_loss += loss.item()
                
                outputs = F.sigmoid(outputs)
                
                preds = (outputs > 0.5).float()
                correct += (preds == batch_label).sum().item()
                total += batch_label.size(0)
                
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(batch_label.cpu().numpy())
            
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        val_f1, val_precision, val_recall = compute_sequence_metrics(all_preds, all_labels)
        
        print(f"Train Loss : {avg_train_loss:.4f} || Valid Loss : {avg_val_loss:.4f} | Valid Accuracy : {val_accuracy:.4f}%")
        print(f"Epoch {epoch+1} - F1 : {val_f1:.4f}, Precision : {val_precision:.4f}, Recall : {val_recall:.4f}")
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save({'model_state_dict' : model.state_dict(),
                        'scaler' : scaler},'./mlp_ckpt/best_mlp_2.pth')
            print(f"Best model saved at epoch {epoch+1} (acc : {val_accuracy:.4f})")

if __name__ == '__main__':
    main()