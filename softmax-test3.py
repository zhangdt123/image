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
    model_path = "moe_model_best4.pth"
    test_gen_dir = "imageclef data/generated"         # 测试生成图像目录
    test_real_dir = "imageclef data/real_unknown"      # 测试未知真实图像目录
    output_csv = "run3.csv"
    sigmoid_scale = 7.0                 # 控制sigmoid陡峭程度（越大越接近0/1）
    threshold = 0.8                      # sigmoid score 的分类阈值
    input_size = 224
    batch_size = 128

# ====== 模型结构导入 ======
from train3 import MOEContrastiveModel

class FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = MOEContrastiveModel()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = model.to(Config.device).eval()

    @torch.no_grad()
    def forward(self, x):
        feat, _ = self.model.momentum_forward(x)
        return feat

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
    all_feats = []
    all_names = []
    for imgs, names in loader:
        imgs = imgs.to(Config.device)
        with torch.no_grad():
            feats = model(imgs)
        all_feats.append(feats.cpu())
        all_names.extend(names)
    return torch.cat(all_feats, dim=0), all_names

# ====== 主推理逻辑 ======
def main():
    print("加载模型...")
    extractor = FeatureExtractor(Config.model_path)

    print("加载图像...")
    gen_set = ImageDataset(Config.test_gen_dir)
    real_set = ImageDataset(Config.test_real_dir)

    F_gen, _ = extract_features(extractor, gen_set)     # shape: [N_gen, D]
    F_real, real_names = extract_features(extractor, real_set)  # shape: [N_real, D]

    print("计算相似度矩阵...")
    sim_matrix = torch.mm(F_gen, F_real.T).numpy()      # shape: [N_gen, N_real]
    max_sim = np.max(sim_matrix, axis=0)                # [N_real]，每张真实图最大相似度

    print("应用 sigmoid 得分...")
    soft_scores = 1 / (1 + np.exp(-Config.sigmoid_scale * (max_sim - 0.5)))  # Sigmoid归一化
    labels = (soft_scores >= Config.threshold).astype(int)

    print("保存结果...")
    df = pd.DataFrame({
        "original_image": real_names,
        "label": labels
    })
    df.to_csv(Config.output_csv, index=False)
    print(f"✅ 预测完成，结果保存至 {Config.output_csv}")

if __name__ == "__main__":
    main()
