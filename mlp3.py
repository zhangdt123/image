import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# ====== 配置区域 ======
class Config:
    moe_model_path = "moe_model_best4.pth"
    real_pos_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_used"     # 正样本，label=1（用于生成的图）
    real_neg_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_not_used/" # 负样本，label=0（未参与生成）
    mlp_save_path = "mlp_classifier.pth"
    input_size = 224
    batch_size = 128
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from train3 import MOEContrastiveModel

# ====== 提取器定义 ======
class FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = MOEContrastiveModel()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = model.to(Config.device).eval()

    @torch.no_grad()
    def forward(self, x):
        # 直接使用封装好的 momentum_forward
        emb, _ = self.model.momentum_forward(x)
        return emb

# ====== 图像数据加载类 ======
class LabeledImageDataset(Dataset):
    def __init__(self, img_dir, label):
        self.img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        self.labels = [label] * len(self.img_paths)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(Config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

# ====== 特征提取函数 ======
def extract_features(model, dataset):
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    features, labels = [], []
    for imgs, labs in loader:
        imgs = imgs.to(Config.device)
        with torch.no_grad():
            feats = model(imgs).cpu()
        features.append(feats)
        labels.extend(labs)
    return torch.cat(features, dim=0), torch.tensor(labels)

# ====== MLP分类器结构 ======
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(x)

# ====== MLP训练流程 ======
def train_classifier(train_feats, train_labels):
    model = MLPClassifier(input_dim=train_feats.shape[1]).to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(train_feats, train_labels)
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    for epoch in range(Config.epochs):
        model.train()
        total_loss, all_preds, all_gts = 0, [], []
        for feats, labels in loader:
            feats, labels = feats.to(Config.device), labels.to(Config.device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())

        acc = accuracy_score(all_gts, all_preds)
        f1 = f1_score(all_gts, all_preds)
        prec = precision_score(all_gts, all_preds)
        rec = recall_score(all_gts, all_preds)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataset):.4f} | Acc: {acc:.3f} | F1: {f1:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f}")

    return model

# ====== 主程序 ======
def main():
    print("加载特征提取器...")
    feature_extractor = FeatureExtractor(Config.moe_model_path)

    print("加载正负样本图像...")
    pos_set = LabeledImageDataset(Config.real_pos_dir, label=1)
    neg_set = LabeledImageDataset(Config.real_neg_dir, label=0)
    full_set = pos_set + neg_set

    print("提取特征...")
    feats, labels = extract_features(feature_extractor, full_set)

    print("训练MLP分类器...")
    mlp = train_classifier(feats, labels)

    print("保存模型...")
    torch.save(mlp.state_dict(), Config.mlp_save_path)
    print(f"✅ MLP 分类器已保存至: {Config.mlp_save_path}")

if __name__ == "__main__":
    main()
