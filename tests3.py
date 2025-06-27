import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# ====== 推理配置 ======
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "moe_model_best3.pth"
    test_gen_dir = "imageclef data/generated"       # 测试生成图像目录
    test_real_dir = "imageclef data/real_unknown"    # 测试未知真实图像目录
    output_csv = "run3.csv"
    similarity_threshold = 0.6        # <<< 使用伪标签调参后的最优值
    strategy = "max"                  # max / mean / topk
    top_k = 3
    input_size = 224
    batch_size = 128

# ====== 导入训练模型结构 ======
from train3 import MOEContrastiveModel

class FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = MOEContrastiveModel()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = model.to(Config.device).eval()

    @torch.no_grad()
    def extract(self, x):
        features, _ = self.model.momentum_forward(x)
        return features

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

# ====== 特征提取函数 ======
def extract_features(model, dataset):
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    features = []
    filenames = []
    for imgs, names in loader:
        imgs = imgs.to(Config.device)
        with torch.no_grad():
            feats = model.extract(imgs).cpu()
        features.append(feats)
        filenames.extend(names)
    return torch.cat(features, dim=0), filenames

# ====== 聚合相似度策略 ======
def compute_similarity_matrix(F_gen, F_real):
    return torch.mm(F_gen, F_real.T).numpy()

def aggregate_similarity(sim_matrix, strategy="topk", top_k=3):
    if strategy == "max":
        return np.max(sim_matrix, axis=0)
    elif strategy == "mean":
        return np.mean(sim_matrix, axis=0)
    elif strategy == "topk":
        top_k_sim = np.sort(sim_matrix, axis=0)[-top_k:, :]
        return np.mean(top_k_sim, axis=0)
    else:
        raise ValueError("Unknown strategy")

# ====== 主函数 ======
def main():
    print("加载模型...")
    extractor = FeatureExtractor(Config.model_path)

    print("加载生成图像...")
    gen_set = ImageDataset(Config.test_gen_dir)
    F_gen, _ = extract_features(extractor, gen_set)

    print("加载未知真实图像...")
    real_set = ImageDataset(Config.test_real_dir)
    F_real, real_names = extract_features(extractor, real_set)

    print("计算相似度矩阵并聚合...")
    sim_matrix = compute_similarity_matrix(F_gen, F_real)
    agg_sim = aggregate_similarity(sim_matrix, strategy=Config.strategy, top_k=Config.top_k)

    print("应用阈值判定...")
    labels = (agg_sim >= Config.similarity_threshold).astype(int)

    print("写入预测结果...")
    df = pd.DataFrame({
        "original_image": real_names,
        "label": labels
    })
    df.to_csv(Config.output_csv, index=False)
    print(f"✅ 预测完成，结果保存至 {Config.output_csv}")

if __name__ == "__main__":
    main()
