import os
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# 커스텀 데이터셋 클래스 정의
class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        
        for root, _, files in os.walk(folder_path):
            label = Path(root).name
            if label not in self.label_to_idx:
                self.label_to_idx[label] = len(self.label_to_idx)
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
                    self.labels.append(self.label_to_idx[label])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label, img_path
        except (IOError, OSError, Image.UnidentifiedImageError):
            print(f"Skipping corrupted image: {img_path}")
            return None, label, img_path

# 커스텀 collate_fn 정의
def custom_collate_fn(batch):
    filtered_batch = [(img, label, path) for img, label, path in batch if img is not None]
    if not filtered_batch:
        return None, None, None
    images, labels, paths = zip(*filtered_batch)
    return torch.stack(images), torch.tensor(labels), paths

# 이미지 변환 정의
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

# EfficientNet 모델 정의
def get_convnext(num_classes):
    model = models.convnext_tiny(pretrained=True)
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, num_classes)
    )
    return model

# 폴더 선택 함수
def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="폴더를 선택하세요")
    root.destroy()
    return folder_path

# 학습 함수
def train_model(model, dataloader, criterion, optimizer, num_epochs, device, save_path):
    model.train()
    best_loss = float('inf')  # 최소 손실 초기화 (무한대)
    best_epoch = 0  # 최소 손실을 기록한 epoch

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        epoch_loss = 0.0  # 한 epoch의 평균 손실 계산용
        num_batches = 0   # 처리된 배치 수
        
        for i, batch in enumerate(dataloader):
            if i >= 10:  # 10배치만 처리
                break
            
            images, labels, paths = batch
            if images is None:
                print("이 배치에 유효한 이미지가 없습니다.")
                continue
            
            print("현재 학습 중인 배치:")
            for img_path, label_idx in zip(paths, labels):
                file_name = os.path.basename(img_path)
                label = Path(img_path).parent.name
                print(f"파일명: {file_name}, 라벨: {label} (인덱스: {label_idx})")
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            print(f"Loss: {loss.item():.4f}")
            print("-" * 50)
        
        # 한 epoch의 평균 손실 계산
        epoch_avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Epoch {epoch + 1} 평균 손실: {epoch_avg_loss:.4f}")

        # 최소 손실 갱신 및 모델 저장
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            best_epoch = epoch + 1
            best_model_path = f"[Convnext]{save_path}_best_epoch_{best_epoch}_Loss_{best_loss:.4f}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"최소 손실 갱신! 모델이 {best_model_path}에 저장되었습니다. (Loss: {best_loss:.4f})")

        # 10 에포크마다 모델 저장
        if (epoch + 1) % 10 == 0:
            epoch_no = epoch + 1
            epoch_save_path = f"[Convnext]{save_path}_epoch_{epoch_no}_Loss_{best_loss:.4f}.pth"
            torch.save(model.state_dict(), epoch_save_path)
            print(f"모델이 {epoch_save_path}에 저장되었습니다.")

# 메인 실행 함수
def main():
    folder_path = select_folder()
    if not folder_path:
        print("폴더가 선택되지 않았습니다.")
        return
    
    transform = get_transforms()
    dataset = ImageDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.label_to_idx)
    
    model = get_convnext(num_classes).to(device)
    
    #=============================================================================#
    # 학습된 모델 가중치 로드
    pretrained_path = ('C:/Users/Evan/OneDrive/바탕 화면/VScode/[1st]train/[Convnext]trained_convnext_model_best_epoch_35_Loss_0.1221.pth')
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"사전 학습된 모델이 {pretrained_path}에서 성공적으로 로드되었습니다.")
    else:
        print(f"사전 학습된 모델 파일을 찾을 수 없습니다: {pretrained_path}")
        print("새로운 모델로 학습을 시작합니다.")
    #=============================================================================#
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    save_path = "trained_convnext_model"
    
    num_epochs = 50  # 총 50 epoch
    train_model(model, dataloader, criterion, optimizer, num_epochs, device, save_path)
    
    # 저장된 모델 확인 (마지막 저장 모델 로드)
    print("\n저장된 모델 확인:")
    loaded_model = get_convnext(num_classes).to(device)
    loaded_model.load_state_dict(torch.load(f"[Convnext]{save_path}_epoch_50_.pth"))
    loaded_model.eval()
    print(f"Epoch 50의 모델이 성공적으로 로드되었습니다.")
    print(f"클래스 수: {num_classes}, 라벨 매핑: {dataset.label_to_idx}")

if __name__ == "__main__":
    main()