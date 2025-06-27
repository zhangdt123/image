import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np


class TestConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_gen_dir = "imageclef data/generated"
    test_real_dir = "imageclef data/real_unknown"
    model_path = "model_epoch2.pth"
    result_csv = "run.csv"
    similarity_threshold = 0.6
    input_size = 224
    batch_size = 256


# 从训练文件导入模型定义
from train import AdvancedContrastiveModel


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        original_model = AdvancedContrastiveModel()
        original_model.load_state_dict(torch.load(TestConfig.model_path, map_location='cpu'))
        self.encoder = original_model.momentum_encoder

    @torch.no_grad()
    def forward(self, x):
        x = self.encoder[0](x)
        features = self.encoder[1](x.flatten(1))[0]
        return F.normalize(features, dim=1)


class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        self.filenames = [os.path.basename(p) for p in self.img_paths]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(TestConfig.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img), self.filenames[idx]  # 直接返回tensor


def main():
    model = FeatureExtractor().to(TestConfig.device).eval()

    print("加载数据...")
    gen_dataset = ImageDataset(TestConfig.test_gen_dir)
    real_dataset = ImageDataset(TestConfig.test_real_dir)

    def batch_extract(dataset):
        loader = DataLoader(dataset, batch_size=TestConfig.batch_size,
                            num_workers=4, pin_memory=True, shuffle=False)
        all_features = []
        all_names = []
        for batch, names in loader:
            with torch.no_grad():
                features = model(batch.to(TestConfig.device)).cpu()
            all_features.append(features)
            all_names.extend(names)
        return torch.cat(all_features), all_names

    print("处理生成图像...")
    gen_feat, gen_names = batch_extract(gen_dataset)

    print("\n处理原始图像...")
    real_feat, real_names = batch_extract(real_dataset)

    print("\n计算相似度...")
    similarity = torch.mm(gen_feat, real_feat.T).numpy()

    print("\n生成检测结果...")
    max_sim = np.max(similarity, axis=0)
    labels = (max_sim >= TestConfig.similarity_threshold).astype(int)

    pd.DataFrame({
        "original_image": real_names,
        "label": labels
    }).to_csv(TestConfig.result_csv, index=False)
    print(f"结果保存至: {TestConfig.result_csv}")


if __name__ == "__main__":
    main()
