import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

# ====== 配置 ======
class Config:
    model_path = "moe_model_best5.pth"
    generated_dir = "imageclef data/generated"
    real_unknown_dir = "imageclef data/real_unknown"
    output_csv = "run.csv"
    input_size = 224
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = 0.62  # 相似度阈值

# ====== 模型结构导入 ======
from tran31  import MOEContrastiveModel  # 保证 train.py 中模型结构没变

# ====== 图像数据集类 ======
class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
        self.filenames = [os.path.basename(p) for p in self.paths]
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
        return self.transform(img), self.filenames[idx]

# ====== 特征提取函数 ======
def extract_features(model, dataloader):
    features, names = [], []
    with torch.no_grad():
        for imgs, fnames in tqdm(dataloader):
            imgs = imgs.to(Config.device)
            feats, _ = model.momentum_forward(imgs)
            features.append(feats.cpu())
            names.extend(fnames)
    return torch.cat(features, dim=0), names

# ====== 主流程 ======
def main():
    print("🔍 加载模型...")
    model = MOEContrastiveModel()
    model.load_state_dict(torch.load(Config.model_path, map_location="cpu"))
    model = model.to(Config.device).eval()

    print("📂 加载数据...")
    gen_loader = DataLoader(ImageDataset(Config.generated_dir), batch_size=Config.batch_size, shuffle=False)
    real_loader = DataLoader(ImageDataset(Config.real_unknown_dir), batch_size=Config.batch_size, shuffle=False)

    print("🧠 提取生成图特征...")
    gen_feats, _ = extract_features(model, gen_loader)  # shape: [N_gen, D]

    print("🧠 提取真实图特征...")
    real_feats, real_names = extract_features(model, real_loader)  # shape: [N_real, D]

    print("📏 计算相似度...")
    sim_matrix = torch.mm(gen_feats, real_feats.T).numpy()  # shape: [N_gen, N_real]
    max_sim = np.max(sim_matrix, axis=0)  # 每张 real 图与所有 gen 图的最大相似度

    print("🧪 判定是否参与生成...")
    labels = (max_sim >= Config.threshold).astype(int)

    print("💾 写入 CSV...")
    pd.DataFrame({
        "original_image": real_names,
        "label": labels
    }).to_csv(Config.output_csv, index=False)
    print(f"✅ 推理完成，结果保存在 {Config.output_csv}")

if __name__ == "__main__":
    main()
