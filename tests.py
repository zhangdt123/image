import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as models

class TestConfig:
    model_path = "matching_model_best.pth"  # 训练好的模型权重路径
    test_gen_dir = "imageclef data/generated"  # 测试集生成图像目录
    test_orig_dir = "imageclef data/real_unknown"  # 测试集原始图像目录
    output_file = "run.csv"  # 输出结果文件

    # 模型参数（与训练一致）
    feature_dim = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 验证参数
    batch_size = 32
    similarity_threshold = 0.65  # 根据验证集最佳表现调整



class FeatureExtractor(nn.Module):
    """特征提取模型（与训练结构一致）"""

    def __init__(self):
        super().__init__()
        backbone = models.resnet50(pretrained=False)
        backbone.fc = nn.Identity()
        self.encoder = nn.Sequential(
            backbone,
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )

    def forward(self, x):
        return self.encoder(x)


class TestDataset(Dataset):
    """测试数据集"""

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = sorted(os.listdir(img_dir))
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, self.img_list[idx]  # 返回文件名用于匹配



def load_model():
    """加载训练好的模型"""
    model = FeatureExtractor().to(TestConfig.device)
    checkpoint = torch.load(TestConfig.model_path, map_location=TestConfig.device)
    new_checkpoint = {
        k.replace("encoder.0.", "0."): v  # 去除父模块名
        for k, v in checkpoint.items()
    }

    backbone = models.resnet50(pretrained=True)
    backbone.fc = nn.Identity()
    model = nn.Sequential(
        backbone,
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.LayerNorm(512)
    ).to(TestConfig.device)

    # 加载适配后的参数
    missing_keys, unexpected_keys = model.load_state_dict(
        new_checkpoint, strict=False
    )

    model.eval()
    return model


def batch_extract_features(model, dataset):
    """批量特征提取"""
    loader = DataLoader(dataset,
                        batch_size=TestConfig.batch_size,
                        num_workers=4,
                        shuffle=False)

    all_features = []
    all_filenames = []
    with torch.no_grad():
        for imgs, filenames in loader:
            imgs = imgs.to(TestConfig.device)
            features = model(imgs)
            all_features.append(features.cpu())
            all_filenames.extend(filenames)

    return torch.cat(all_features, dim=0), all_filenames


def main_test():
    # 1. 初始化模型
    model = load_model()

    # 2. 准备测试数据
    print("正在加载测试数据...")
    # 生成图像数据
    gen_dataset = TestDataset(TestConfig.test_gen_dir)
    # 原始图像数据
    orig_dataset = TestDataset(TestConfig.test_orig_dir)

    # 3. 提取特征
    print("提取生成图像特征...")
    gen_features, gen_filenames = batch_extract_features(model, gen_dataset)
    print("提取原始图像特征...")
    orig_features, orig_filenames = batch_extract_features(model, orig_dataset)

    # 4. 计算相似度矩阵
    print("计算相似度矩阵...")
    gen_features = F.normalize(gen_features, p=2, dim=1)
    orig_features = F.normalize(orig_features, p=2, dim=1)
    similarity_matrix = torch.mm(gen_features, orig_features.T)

    # 5. 为每个生成图像找到最佳匹配
    print("匹配最佳原始图像...")
    max_similarities, max_indices = torch.max(similarity_matrix, dim=1)

    # 6. 构建匹配结果
    print("生成结果文件...")
    with open(TestConfig.output_file, "w") as f:
        f.write("原始图像文件名,是否匹配\n")

        # 记录原始图像的匹配状态（这里输出所有可能的原始图像）
        matched_origins = set()
        for idx in max_indices.numpy():
            matched_origins.add(orig_filenames[idx])

        # 输出所有原始图像的判断结果
        for orig_file in orig_filenames:
            is_matched = 1 if orig_file in matched_origins and \
                              max_similarities[max_indices.eq(orig_filenames.index(orig_file))].max() \
                              > TestConfig.similarity_threshold else 0
            f.write(f"{orig_file},{is_matched}\n")

    print(f"测试完成！结果已保存到 {TestConfig.output_file}")


if __name__ == "__main__":
    main_test()

