import os
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import pickle


# 이미지 변환 정의 (테스트용)
def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


# Vision Transformer (ViT) 모델 정의
def get_vit(num_classes):
    model = models.vit_b_16(weights=None)
    num_features = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, num_classes)
    )
    return model


# 모델 로드 함수
def load_model(model_path, num_classes, device):
    model = get_vit(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# 이미지 예측 함수 (단일 모델)
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

        return predicted_label, confidence, img
    except Exception as e:
        return f"오류 발생: {str(e)}", 0.0, None


# 세 모델로 공동 예측 함수
def predict_with_three_models(model1, model2, model3, image_path, transform, device, label_to_idx):
    label1, conf1, img = predict_image(model1, image_path, transform, device, label_to_idx)
    label2, conf2, _ = predict_image(model2, image_path, transform, device, label_to_idx)
    label3, conf3, _ = predict_image(model3, image_path, transform, device, label_to_idx)

    # 오류 체크
    if any(label.startswith("오류 발생") for label in [label1, label2, label3]):
        return label1, 0.0, None

    # 공통 라벨 결정 (다수결)
    if label1 == label2 == label3:
        common_label = label1
        avg_confidence = (conf1 + conf2 + conf3) / 3
    elif label1 == label2 or label1 == label3:
        common_label = label1
        avg_confidence = (conf1 + (conf2 if label1 == label2 else conf3)) / 2
    elif label2 == label3:
        common_label = label2
        avg_confidence = (conf2 + conf3) / 2
    else:
        common_label = "Undefined Sample"
        avg_confidence = (conf1 + conf2 + conf3) / 3

    return common_label, avg_confidence, img, (label1, conf1), (label2, conf2), (label3, conf3)


# GUI 클래스 정의
class TripleViTImageClassifierGUI:
    def __init__(self, root, vit1_model_path, vit2_model_path, vit3_model_path, label_to_idx):
        self.root = root
        self.root.title("")  # 기본 제목 비우기
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_test_transforms()
        self.label_to_idx = label_to_idx
        self.num_classes = len(label_to_idx)

        # 세 개의 ViT 모델 로드
        self.vit1_model = load_model(vit1_model_path, self.num_classes, self.device)
        self.vit2_model = load_model(vit2_model_path, self.num_classes, self.device)
        self.vit3_model = load_model(vit3_model_path, self.num_classes, self.device)

        # 커스텀 제목 라벨 추가 (폰트 크기 20)
        self.title_label = tk.Label(root, text="[Triple Vision Transformer] Image Classifier", font=("Arial", 20, "bold"))
        self.title_label.pack(pady=10)

        self.instruction_label = tk.Label(root, text="Please select a test sample.", font=("Arial", 14))
        self.instruction_label.pack(pady=10)

        self.select_button = tk.Button(root, text="이미지 선택", command=self.select_and_predict, font=("Arial", 12))
        self.select_button.pack(pady=5)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(root, text="", justify="left", font=("Arial", 16))
        self.result_label.pack(pady=10)

    def select_and_predict(self):
        file_path = filedialog.askopenfilename(
            title="테스트 이미지를 선택하세요",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )

        if file_path:
            common_label, avg_confidence, img, pred1, pred2, pred3 = predict_with_three_models(
                self.vit1_model, self.vit2_model, self.vit3_model, file_path, self.transform, self.device, self.label_to_idx
            )

            if img is not None:
                img_resized = img.resize((300, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img_resized)
                self.image_label.config(image=photo)
                self.image_label.image = photo

                result_text = (
                    f"ViT Model 1: {pred1[0]} (Confidence: {pred1[1]:.4f})\n"
                    f"ViT Model 2: {pred2[0]} (Confidence: {pred2[1]:.4f})\n"
                    f"ViT Model 3: {pred3[0]} (Confidence: {pred3[1]:.4f})\n"
                    f"Predicted Label: {common_label}\n"
                    f"Average Confidence: {avg_confidence:.4f}"
                )
                self.result_label.config(text=result_text)
            else:
                self.image_label.config(image='')
                self.result_label.config(text=f"예측 실패: {common_label}")
        else:
            messagebox.showwarning("경고", "이미지가 선택되지 않았습니다.")


# 메인 함수
def main():
    pkl_path = '//Users/lifepathimac/Documents/GitHub/[AI Tutor] Project/label_to_idx.pkl'

    # label_to_idx.pkl 파일 로드
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            label_to_idx = pickle.load(f)
        print(f"'{pkl_path}' 파일에서 label_to_idx를 로드했습니다.")
        print(f"로드된 label_to_idx: {label_to_idx}")
    else:
        messagebox.showerror("오류", f"'{pkl_path}' 파일을 찾을 수 없습니다. 학습 과정에서 생성된 파일이 필요합니다.")
        return

    # 세 개의 ViT 모델 경로 (사용자가 실제 경로로 교체해야 함)
    vit1_model_path = ('/Volumes/Koo Diary/AI Tutor/Trained AI Models/29 Classes/'
                       '[ViT-3]trained_vit_model_best_epoch_42_Loss_0.0323.pth')
    vit2_model_path = ('/Volumes/Koo Diary/AI Tutor/Trained AI Models/29 Classes/'
                       '[ViT-4]trained_vit_model_best_epoch_34_Loss_0.0544.pth')
    vit3_model_path = ('/Volumes/Koo Diary/AI Tutor/Trained AI Models/29 Classes/'
                       '[ViT-5]trained_vit_model_best_epoch_37_Loss_0.0378.pth')  # 세 번째 모델 경로 추가 (예시)

    for model_path in [vit1_model_path, vit2_model_path, vit3_model_path]:
        if not os.path.exists(model_path):
            messagebox.showerror("오류", f"모델 파일을 찾을 수 없습니다: {model_path}")
            return

    # GUI 실행
    root = tk.Tk()
    app = TripleViTImageClassifierGUI(root, vit1_model_path, vit2_model_path, vit3_model_path, label_to_idx)
    root.mainloop()


if __name__ == "__main__":
    main()