# precompute_similarity_fixed.py
import os

# ========== 必须首行设置 ==========
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载OpenMP
os.environ["OMP_NUM_THREADS"] = "1"  # 限制OpenMP线程数

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


class Config:
    """配置参数（需修改路径）"""
    gen_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/generated"  # 生成图像路径
    used_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_used"  # 被使用的原图路径
    output_path = "F:/image clef2025/sim_matrix.npy"  # 输出相似度矩阵路径

    batch_size = 128  # 根据内存调整，建议使用2的幂次
    num_workers = 0  # Windows必须设为0，Linux可用1
    device = "cpu"  # 强制使用CPU模式避免冲突


class PreprocessDataset(Dataset):
    """数据预处理模块（包含鲁棒性优化）"""

    def __init__(self, img_dir):
        self.img_paths = []
        for f in os.listdir(img_dir):
            full_path = os.path.join(img_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.img_paths.append(full_path)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_paths[idx]).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"错误加载图像 {self.img_paths[idx]}: {e}")
            # 返回空白图像替代，避免数据流中断
            return torch.zeros(3, 224, 224)


class FeatureExtractor(nn.Module):
    """优化的特征提取模型（基于ResNet50）"""

    def __init__(self):
        super().__init__()
        origin_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(origin_model.children())[:-1],  # 移除全连接层
            nn.Flatten()
        )

    def forward(self, x):
        return self.feature_extractor(x)


def robust_extract_features(extractor, dataloader):
    """
    稳定版特征提取流程
    - 强制使用CPU模式
    - 禁用梯度计算
    - 防止内存泄漏的上下文管理
    """
    extractor.eval()
    features = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="特征提取进度"):
            # 强制使用CPU并转换为32位精度
            batch = batch.to(torch.device(Config.device), dtype=torch.float32)
            feats = extractor(batch).cpu().numpy()  # 立即转回CPU
            features.append(feats)

    return np.concatenate(features, axis=0)


def compute_cosine_similarity(feat1, feat2):
    """余弦相似度计算的优化实现"""
    # 双精度计算提高数值稳定性
    feat1 = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True, ord=2)
    feat2 = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True, ord=2)
    return np.dot(feat1, feat2.T)


def main():
    # ========== 初始化配置 ==========
    torch.set_num_threads(1)  # 显式限制PyTorch线程数
    os.makedirs(os.path.dirname(Config.output_path), exist_ok=True)

    # ========== 特征提取 ==========
    extractor = FeatureExtractor().to(Config.device)
    extractor.train(False)  # 禁用dropout和BN的train模式

    # 生成图像特征
    gen_dataset = PreprocessDataset(Config.gen_dir)
    gen_loader = DataLoader(
        gen_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,  # Windows保持为0
        pin_memory=False  # 关闭内存锁页避免冲突
    )
    gen_features = robust_extract_features(extractor, gen_loader)

    # 被使用的原图特征
    used_dataset = PreprocessDataset(Config.used_dir)
    used_loader = DataLoader(
        used_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=False
    )
    used_features = robust_extract_features(extractor, used_loader)

    # ========== 计算相似度矩阵 ==========
    sim_matrix = compute_cosine_similarity(gen_features, used_features)

    # ========== 保存结果 ==========
    np.save(Config.output_path, sim_matrix.astype(np.float32))  # 保存32位浮点节省空间
    print(f"成功生成相似度矩阵！保存路径：{Config.output_path}")
    print(f"矩阵维度：生成图像数={sim_matrix.shape[0]}, 原图数量={sim_matrix.shape[1]}")


if __name__ == "__main__":
    main()
