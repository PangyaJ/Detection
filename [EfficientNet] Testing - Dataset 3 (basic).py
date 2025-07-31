import os
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image, ImageTk  # ImageTk 추가
from pathlib import Path
import pickle

# 커스텀 데이터셋 클래스 정의 (label_to_idx 생성용)
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

# 이미지 변환 정의 (테스트용)
def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Vision Transformer (ViT-B/16) 모델 정의
def get_efficientnet(num_classes):
    model = models.efficientnet_b1(weights=None)  # 사전 학습 가중치 없이 초기화
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, num_classes)
    )
    return model

# 폴더 선택 함수
def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(
        title="학습 데이터셋 폴더를 선택하세요 (label_to_idx 생성용)")
    root.destroy()
    return folder_path

# 모델 파일 선택 함수
def select_model_file():
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(
        title="학습된 모델 파일을 선택하세요",
        filetypes=[("PyTorch Model Files", "*.pth")]
    )
    root.destroy()
    return model_path

# 모델 로드 함수
def load_model(model_path, num_classes, device):
    model = get_efficientnet(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 이미지 예측 함수
def predict_image(model, image_path, transform, device, label_to_idx):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
            
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        predicted_label = idx_to_label[predicted_idx]
        
        return predicted_label, confidence, img  # 원본 이미지도 반환
    except Exception as e:
        return f"오류 발생: {str(e)}", 0.0, None

# GUI 클래스 정의
class ImageClassifierGUI:
    def __init__(self, root, model_path, label_to_idx):
        self.root = root
        self.root.title("이미지 분류기")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_test_transforms()
        self.label_to_idx = label_to_idx
        self.num_classes = len(label_to_idx)
        
        self.model = load_model(model_path, self.num_classes, self.device)
        
        # GUI 요소
        self.instruction_label = tk.Label(root, text="테스트 이미지를 선택하세요")
        self.instruction_label.pack(pady=10)
        
        self.select_button = tk.Button(root, text="이미지 선택", command=self.select_and_predict)
        self.select_button.pack(pady=5)
        
        # 이미지 표시용 라벨
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
        
        # 결과 텍스트 표시용 라벨
        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

    def select_and_predict(self):
        file_path = filedialog.askopenfilename(
            title="테스트 이미지를 선택하세요",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        
        if file_path:
            predicted_label, confidence, img = predict_image(
                self.model, file_path, self.transform, self.device, self.label_to_idx
            )
            
            if img is not None:
                # 이미지 크기 조정 (화면에 맞게)
                img_resized = img.resize((300, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img_resized)
                self.image_label.config(image=photo)
                self.image_label.image = photo  # 참조 유지
                
                # 결과 텍스트 업데이트
                result_text = f"예측 라벨: {predicted_label}\n확신도: {confidence:.4f}"
                self.result_label.config(text=result_text)
            else:
                self.image_label.config(image='')  # 이미지 없음
                self.result_label.config(text=f"예측 실패: {predicted_label}")
        else:
            messagebox.showwarning("경고", "이미지가 선택되지 않았습니다.")

# 메인 함수
def main():
    # 학습 데이터셋 폴더 선택 (label_to_idx 생성용)
    folder_path = select_folder()
    if not folder_path:
        messagebox.showerror("오류", "학습 데이터셋 폴더가 선택되지 않았습니다.")
        return

    # label_to_idx 생성
    dataset = ImageDataset(folder_path)  # transform 없이도 label_to_idx 생성 가능
    label_to_idx = dataset.label_to_idx
    with open('label_to_idx.pkl', 'wb') as f:
        pickle.dump(label_to_idx, f)
    print("label_to_idx가 'label_to_idx.pkl' 파일에 저장되었습니다.")
    print(f"생성된 label_to_idx: {label_to_idx}")

    # 학습된 모델 파일 선택 (팝업으로)
    model_path = select_model_file()
    if not model_path:
        messagebox.showerror("오류", "모델 파일이 선택되지 않았습니다.")
        return
    if not os.path.exists(model_path):
        messagebox.showerror("오류", f"모델 파일을 찾을 수 없습니다: {model_path}")
        return

    # GUI 실행
    root = tk.Tk()
    app = ImageClassifierGUI(root, model_path, label_to_idx)
    root.mainloop()

if __name__ == "__main__":
    main()