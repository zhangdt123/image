import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.models as models

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置（示例如下）
    train_gen_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/generated"  # 生成图像
    train_used_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_used"  # 被用到的原始图像
    train_unused_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_not_used"  # 未被使用的原始图像

    val_gen_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/val-generated"
    val_orig_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/val-originals"

    batch_size = 32
    learning_rate = 3e-4
    epochs = 10

    # 对比学习参数
    temperature = 0.1
    num_pos_samples = 3  # 每个生成图像使用的正样本数
    num_neg_samples = 10  # 每个生成图像使用的负样本数


class ContrastiveDataset(Dataset):
    def __init__(self, gen_dir, used_dir, unused_dir, num_pos, num_neg):
        # 加载所有文件路径
        self.gen_paths = sorted([os.path.join(gen_dir, f) for f in os.listdir(gen_dir)])
        self.used_paths = sorted([os.path.join(used_dir, f) for f in os.listdir(used_dir)])
        self.unused_paths = sorted([os.path.join(unused_dir, f) for f in os.listdir(unused_dir)])

        self.num_pos = num_pos
        self.num_neg = num_neg

        # 数据增强
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.gen_paths)

    def __getitem__(self, idx):
        # 生成图像作为anchor
        gen_img = self.transform(Image.open(self.gen_paths[idx]).convert('RGB'))

        # 随机采样正样本（来自used）
        pos_indices = np.random.choice(len(self.used_paths), self.num_pos, replace=False)
        pos_imgs = [self.transform(
            Image.open(self.used_paths[i]).convert('RGB'))
            for i in pos_indices]

        # 采样负样本（来自unused）
        neg_indices = np.random.choice(len(self.unused_paths), self.num_neg, replace=False)
        neg_imgs = [self.transform(
            Image.open(self.unused_paths[i]).convert('RGB'))
            for i in neg_indices]

        return (gen_img,
                torch.stack(pos_imgs),
                torch.stack(neg_imgs))


class MultiSampleContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp

    def forward(self, anchors, positives, negatives):
        """
        输入尺寸：
        anchors: (B, D)
        positives: (B, P, D)
        negatives: (B, N, D)
        """
        B, P, D = positives.shape
        N = negatives.shape[1]

        # 扩展anchor做多对多比较
        anchors_exp = anchors.unsqueeze(1).expand(-1, P, -1)  # (B, P, D)

        # 合并正负样本
        all_samples = torch.cat([positives, negatives], dim=1)  # (B, P+N, D)

        # 计算所有相似度
        sim_matrix = torch.cosine_similarity(
            anchors_exp.unsqueeze(2),  # (B, P, 1, D)
            all_samples.unsqueeze(1),  # (B, 1, P+N, D)
            dim=-1
        ) / self.temp  # (B, P, P+N)

        # 创建标签：前P个是正样本
        labels = torch.zeros(B, P, dtype=torch.long).to(anchors.device)
        labels[:, :P] = torch.arange(P).to(anchors.device)

        # 对比损失
        loss = -torch.nn.functional.log_softmax(sim_matrix, dim=-1)
        loss = torch.gather(loss, 2, labels.unsqueeze(-1)).mean()

        return loss


# ---------- 验证模块 ----------
class ValidationDataset(Dataset):
    """用于验证的模糊匹配测试数据集"""

    def __init__(self, gen_dir, orig_dir):
        self.gen_paths = sorted([os.path.join(gen_dir, f) for f in os.listdir(gen_dir)])
        self.orig_paths = sorted([os.path.join(orig_dir, f) for f in os.listdir(orig_dir)])
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.gen_paths)

    def __getitem__(self, idx):
        gen_img = self.transform(Image.open(self.gen_paths[idx]).convert('RGB'))
        rand_orig = self.transform(
            Image.open(np.random.choice(self.orig_paths)).convert('RGB'))
        return gen_img, rand_orig


def evaluate_model(model, gen_dir, orig_dir, num_test=500):
    """批量验证匹配准确率"""
    model.eval()
    gen_features = []
    orig_features = []

    # 提取生成图像特征
    gen_loader = DataLoader(ValidationDataset(gen_dir, orig_dir),
                            batch_size=16, shuffle=False)
    with torch.no_grad():
        for gen, _ in tqdm(gen_loader, desc="提取生成特征"):
            gen_features.append(model(gen.to(Config.device)))
    gen_features = torch.cat(gen_features, dim=0)

    actual_test_num = min(num_test,len(gen_features))  # 保证采样数量不超过实际总数
    print(f"实际测试样本数：{actual_test_num}（可用数：{len(gen_features)}）")        # 修改原来的num_test为actual_test_num ↓
    rnd_idx = np.random.choice(len(gen_features), actual_test_num, replace=False)
    # 提取原始图像特征
    orig_loader = DataLoader(ValidationDataset(orig_dir, orig_dir),
                             batch_size=32, shuffle=False)
    with torch.no_grad():
        for _, orig in tqdm(orig_loader, desc="提取原始特征"):
            orig_features.append(model(orig.to(Config.device)))
    orig_features = torch.cat(orig_features, dim=0)

    # 计算余弦相似度矩阵
    sim_matrix = torch.mm(
        torch.nn.functional.normalize(gen_features, dim=1),
        torch.nn.functional.normalize(orig_features, dim=1).t()
    )

    # 随机抽样评估
    rnd_idx = np.random.choice(len(gen_features), num_test, replace=False)
    scores = []
    for idx in rnd_idx:
        sorted_sims = sim_matrix[idx].argsort(descending=True)
        # 假设前N%的被使用过（根据实际情况调整）
        possible_positives = sorted_sims[:int(len(sorted_sims) * 0.01)]
        scores.append(len(possible_positives) >0 )

    accuracy = sum(scores) / len(scores)
    return accuracy

# ---------- 训练流程 ----------
def main():
    # 初始化模型
    backbone = models.resnet50(pretrained=True)
    backbone.fc = nn.Identity()  # 使用默认特征
    model = nn.Sequential(
        backbone,
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.LayerNorm(512)
    ).to(Config.device)

    # 准备数据
    train_dataset = ContrastiveDataset(
        Config.train_gen_dir,
        Config.train_used_dir,
        Config.train_unused_dir,
        num_pos=Config.num_pos_samples,
        num_neg=Config.num_neg_samples
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=Config.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    # 优化配置
    criterion = MultiSampleContrastiveLoss(temp=Config.temperature)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=Config.learning_rate,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 训练循环
    best_acc = 0.0
    for epoch in range(Config.epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            anchors = batch[0].to(Config.device)
            positives = batch[1].to(Config.device)
            negatives = batch[2].to(Config.device)

            # 获取批尺寸
            B = anchors.size(0)
            P = Config.num_pos_samples
            N = Config.num_neg_samples

            # 前向传播（展开所有样本）
            a_features = model(anchors)  # [B, D]

            # 处理正样本（形状：[B, P, C, H, W] -> [B*P, C, H, W]）
            p_all = positives.view(-1, *positives.shape[2:])
            p_features = model(p_all).view(B, P, -1)

            # 处理负样本（形状：[B, N, C, H, W] -> [B*N, C, H, W]）
            n_all = negatives.view(-1, *negatives.shape[2:])
            n_features = model(n_all).view(B, N, -1)

            # 计算损失
            loss = criterion(a_features, p_features, n_features)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * anchors.size(0)

        scheduler.step()
        epoch_loss = epoch_loss / len(train_dataset)

        # 验证评估
        if (epoch + 1) % 1 == 0:
            accuracy = evaluate_model(model, Config.val_gen_dir, Config.val_orig_dir)
            print(f"EPOCH {epoch + 1} | Loss: {epoch_loss:.4f} | Acc: {accuracy:.2%}")

            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(model.state_dict(), "matching_model_best.pth")
                print("✅ 保存新最佳模型")

    #print(f"训练完成，最佳准确率：{best_acc:.2%}")

if __name__ == "__main__":
    main()