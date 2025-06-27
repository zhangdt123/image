import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# ====== 配置区域 ======
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moe_model_path = "moe_model_best4.pth"       # 对比学习模型
    mlp_model_path = "mlp_classifier.pth"       # 训练好的MLP分类器
    test_dir = "imageclef data/real_unknown"                   # 要预测的真实图像目录
    output_csv = "run3.csv"
    input_size = 224
    batch_size = 128

# ====== 导入训练模型结构 ======
from train3 import MOEContrastiveModel

# ====== 特征提取器（冻结的 encoder） ======
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

# ====== 简单 MLP 分类器结构 ======
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

# ====== 图像加载数据集 ======
class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        self.filenames = [os.path.basename(p) for p in self.img_paths]
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
        return self.transform(img), self.filenames[idx]

# ====== 主推理函数 ======
def main():
    print("加载模型和分类器...")
    feature_extractor = FeatureExtractor(Config.moe_model_path)
    mlp = MLPClassifier().to(Config.device)
    mlp.load_state_dict(torch.load(Config.mlp_model_path, map_location='cpu'))
    mlp.eval()

    print("加载图像数据...")
    dataset = ImageDataset(Config.test_dir)
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_preds = []
    all_names = []

    with torch.no_grad():
        for imgs, names in loader:
            imgs = imgs.to(Config.device)
            feats = feature_extractor(imgs)
            logits = mlp(feats)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_names.extend(names)

    print("写入CSV...")
    pd.DataFrame({
        "original_image": all_names,
        "label": all_preds
    }).to_csv(Config.output_csv, index=False)
    print(f"✅ 推理完成，结果保存至 {Config.output_csv}")

if __name__ == "__main__":
    main()
