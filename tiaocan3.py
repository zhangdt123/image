import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import cohen_kappa_score

# ====== 配置区域 ======
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "moe_model_best4.pth"
    gen_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/generated"         # 训练生成图像目录
    real_pos_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_used"     # 用于生成的真实图（伪标签=1）
    real_neg_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_not_used" # 没有参与生成（伪标签=0）
    strategy = "topk"     # max / mean / topk
    top_k = 3
    input_size = 224
    batch_size = 128

# ====== 导入模型结构 ======
from train3 import MOEContrastiveModel

class FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = MOEContrastiveModel()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = model.to(Config.device).eval()

    @torch.no_grad()
    def extract(self, x):
        feats, _ = self.model.momentum_forward(x)
        return feats

class ImageDataset(Dataset):
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
    features = []
    labels = []
    for imgs, labs in loader:
        imgs = imgs.to(Config.device)
        with torch.no_grad():
            feats = model.extract(imgs).cpu()
        features.append(feats)
        labels.extend(labs.numpy())
    return torch.cat(features, dim=0), np.array(labels)

# ====== 相似度策略处理 ======
def compute_similarity_matrix(F_gen, F_real):
    return torch.mm(F_gen, F_real.T).numpy()

def aggregate_similarity(sim_matrix, strategy="max", top_k=3):
    if strategy == "max":
        return np.max(sim_matrix, axis=0)
    elif strategy == "mean":
        return np.mean(sim_matrix, axis=0)
    elif strategy == "topk":
        top_k_sim = np.sort(sim_matrix, axis=0)[-top_k:, :]
        return np.mean(top_k_sim, axis=0)
    else:
        raise ValueError("Unknown strategy")

# ====== 主程序 ======
def main():
    print("加载模型和图像数据...")
    extractor = FeatureExtractor(Config.model_path)

    gen_set = ImageDataset(Config.gen_dir, label=-1)
    pos_set = ImageDataset(Config.real_pos_dir, label=1)
    neg_set = ImageDataset(Config.real_neg_dir, label=0)

    F_gen, _ = extract_features(extractor, gen_set)
    F_pos, L_pos = extract_features(extractor, pos_set)
    F_neg, L_neg = extract_features(extractor, neg_set)

    F_real = torch.cat([F_pos, F_neg], dim=0)
    L_real = np.concatenate([L_pos, L_neg])

    print(f"计算相似度矩阵... ({Config.strategy})")
    sim_matrix = compute_similarity_matrix(F_gen, F_real)
    agg_sim = aggregate_similarity(sim_matrix, strategy=Config.strategy, top_k=Config.top_k)

    print("扫描不同阈值下的评估分数：")
    for t in np.linspace(0.4, 0.9, 26):
        preds = (agg_sim >= t).astype(int)
        acc = accuracy_score(L_real, preds)
        f1 = f1_score(L_real, preds)
        prec = precision_score(L_real, preds)
        rec = recall_score(L_real, preds)
        kappa = cohen_kappa_score(L_real, preds)
        print(f"Thr={t:.2f} | Kappa={kappa:.3f} | Acc={acc:.3f} | F1={f1:.3f} | Prec={prec:.3f} | Rec={rec:.3f}")

if __name__ == "__main__":
    main()
