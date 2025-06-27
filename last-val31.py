import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ==== 配置区域 ====
class Config:
    model_path = "F:/image clef2025/moe_model_best5.pth"
    generated_dir = "F:/image clef2025/imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/generated"
    real_used_dir = "F:/image clef2025/imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_used"
    real_not_used_dir = "F:/image clef2025/imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_not_used"
    input_size = 224
    batch_size = 64
    thresholds = np.arange(0.50, 0.91, 0.02)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 模型结构导入（确保与你训练时一致） ====
from tran31 import MOEContrastiveModel

# ==== 图像数据集类 ====
class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(Config.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

# ==== 特征提取函数 ====
def extract_features(model, dataloader):
    all_feats = []
    with torch.no_grad():
        for imgs in tqdm(dataloader):
            imgs = imgs.to(Config.device)
            feats, _ = model.momentum_forward(imgs)
            all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)

# ==== 主流程 ====
def main():
    print("✅ 加载模型...")
    model = MOEContrastiveModel()
    model.load_state_dict(torch.load(Config.model_path, map_location="cpu"))
    model = model.to(Config.device).eval()

    print("📂 加载生成图...")
    gen_loader = DataLoader(ImageDataset(Config.generated_dir), batch_size=Config.batch_size)
    feats_gen = extract_features(model, gen_loader)

    print("📂 加载 real_used（正样本）...")
    pos_loader = DataLoader(ImageDataset(Config.real_used_dir), batch_size=Config.batch_size)
    feats_pos = extract_features(model, pos_loader)

    print("📂 加载 real_not_used（负样本）...")
    neg_loader = DataLoader(ImageDataset(Config.real_not_used_dir), batch_size=Config.batch_size)
    feats_neg = extract_features(model, neg_loader)

    feats_real = torch.cat([feats_pos, feats_neg], dim=0)
    labels = np.array([1] * len(feats_pos) + [0] * len(feats_neg))

    print("📏 计算相似度矩阵...")
    sim_matrix = torch.mm(feats_gen, feats_real.T).numpy()
    max_sim = np.max(sim_matrix, axis=0)

    print("🔍 多阈值评估：")
    for t in Config.thresholds:
        preds = (max_sim >= t).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        kappa = cohen_kappa_score(labels, preds)
        print(f"Thr={t:.2f} | Kappa={kappa:.3f} | Acc={acc:.3f} | F1={f1:.3f} | Prec={prec:.3f} | Rec={rec:.3f}")

if __name__ == "__main__":
    main()
