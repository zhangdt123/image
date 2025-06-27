import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from MOE混合+注意力3\tran31 import MOEContrastiveModel  # 假设这是你的对比学习模型

class Config:
    """配置参数"""
    gen_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/generated"  # 生成图像路径
    used_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_used"  # 被使用的原图路径
    unused_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_not_used"  # 未使用的原图路径
    output_path = "path_to_save_features.npy"  # 保存特征的路径
    batch_size = 128  # 批量大小，根据显存调整
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 使用 GPU 或 CPU


class PreprocessDataset:
    """数据预处理（将图像加载并进行转换）"""
    def __init__(self, img_dir):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return len(self.img_paths)


def extract_features(model, dataloader):
    """提取图像特征"""
    model.eval()  # 切换到评估模式
    features = []
    with torch.no_grad():  # 禁用梯度计算
        for batch in tqdm(dataloader, desc="提取特征"):
            batch = batch.to(Config.device)
            _, feat = model(batch)  # 获取特征，假设模型返回的第二部分是特征
            features.append(feat.cpu().numpy())  # 转回CPU并存储
    return np.concatenate(features, axis=0)  # 合并所有特征


def compute_cosine_similarity(feat1, feat2):
    """计算余弦相似度"""
    feat1 = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True, ord=2)  # 归一化特征
    feat2 = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True, ord=2)  # 归一化特征
    return np.dot(feat1, feat2.T)  # 计算余弦相似度矩阵


def main():
    # 初始化模型
    model = MOEContrastiveModel().to(Config.device)  # 假设模型结构为 MOEContrastiveModel
    model.eval()  # 确保处于评估模式

    # 生成图像特征
    gen_dataset = PreprocessDataset(Config.gen_dir)  # 加载生成图像数据集
    gen_loader = DataLoader(gen_dataset, batch_size=Config.batch_size, shuffle=False)
    gen_features = extract_features(model, gen_loader)  # 提取生成图像特征

    # 被使用的原图特征
    used_dataset = PreprocessDataset(Config.used_dir)  # 加载被使用的原图数据集
    used_loader = DataLoader(used_dataset, batch_size=Config.batch_size, shuffle=False)
    used_features = extract_features(model, used_loader)  # 提取原图特征

    # 计算相似度矩阵
    sim_matrix = compute_cosine_similarity(gen_features, used_features)

    # 保存相似度矩阵
    np.save(Config.output_path, sim_matrix.astype(np.float32))  # 保存为32位浮点格式
    print(f"相似度矩阵已保存，路径: {Config.output_path}")
    print(f"矩阵维度：生成图像数={sim_matrix.shape[0]}, 原图数量={sim_matrix.shape[1]}")


if __name__ == "__main__":
    main()
