import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from copy import deepcopy


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置（需自定义）
    train_gen_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/generated"
    train_used_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_used"
    train_unused_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_not_used"
    num_neg_samples = 5
    batch_size = 128
    lr = 3e-4
    epochs = 50
    queue_size = 65536  # 负样本队列容量

    # 对比学习参数
    sim_threshold = 0.6
    min_pos_samples = 2
    proj_hidden_dims = [1024, 512, 256]
    momentum = 0.999  # 动量系数
    loss_alpha = 0.7  # 在线与动量损失权重


class ContrastiveDataset(Dataset):
    """支持动态正样本匹配的改良版数据集"""

    def __init__(self, gen_dir, used_dir, unused_dir):
        # 静态数据校验
        self._validate_dirs(gen_dir, used_dir, unused_dir)

        # 加载预计算的相似度矩阵
        self.sim_matrix = np.load("sim_matrix.npy")
        assert self.sim_matrix.shape[0] == len(os.listdir(gen_dir)), "相似度矩阵与生成图像数量不匹配"

        self.gen_paths = sorted(os.path.join(gen_dir, f) for f in os.listdir(gen_dir))
        self.used_paths = sorted(os.path.join(used_dir, f) for f in os.listdir(used_dir))
        self.unused_paths = sorted(os.path.join(unused_dir, f) for f in os.listdir(unused_dir))

        # 构建正样本候选池
        self.positive_pools = self._build_positive_pools(Config.sim_threshold, Config.min_pos_samples)

        # 数据增强
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _validate_dirs(self, *dirs):
        for d in dirs:
            if not os.path.exists(d):
                raise FileNotFoundError(f"数据目录不存在: {d}")

    def _build_positive_pools(self, threshold, min_samples):
        pools = []
        for i in range(len(self.gen_paths)):
            valid_ids = np.where(self.sim_matrix[i] >= threshold)[0]
            if len(valid_ids) < min_samples:
                valid_ids = np.argsort(self.sim_matrix[i])[-min_samples:]
            pools.append(valid_ids)
        return pools

    def __len__(self):
        return len(self.gen_paths)

    def __getitem__(self, idx):
        gen_img = self._load_image(self.gen_paths[idx], is_gen=True)
        pos_ids = np.random.choice(self.positive_pools[idx], Config.min_pos_samples, replace=False)
        positives = [self._load_image(self.used_paths[i]) for i in pos_ids]
        negs = self._get_random_negatives(Config.num_neg_samples)

        return (
            gen_img,
            torch.stack(positives),
            torch.stack(negs)
        )

    def _load_image(self, path, is_gen=False):
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def _get_random_negatives(self, num):
        ids = np.random.choice(len(self.unused_paths), num, replace=False)
        return [self._load_image(self.unused_paths[i]) for i in ids]


class DynamicProjection(nn.Module):
    """可适应不同特征层级的投影头"""

    def __init__(self, in_dim=512, hidden_dims=[1024, 512, 256], use_bn=True):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim

        # 动态温度参数
        self.temp_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()  # 温度限幅在(0.05, 0.2)
        )
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embeddings = self.mlp(x)
        temperature = 0.05 + 0.15 * self.temp_layer(embeddings)
        return embeddings, temperature

class AdvancedContrastiveModel(nn.Module):
    """集成动量编码器与特征队列的完整模型"""

    def __init__(self):
        super().__init__()
        # 骨干网络
        self.backbone = self._build_enhanced_backbone()

        # 动态投影头
        self.projector = DynamicProjection(2048, Config.proj_hidden_dims)

        # 动量编码系统
        self.momentum_encoder = self._build_momentum_encoder()

        # 初始化负样本队列
        self.register_buffer("queue", torch.randn(Config.proj_hidden_dims[-1], Config.queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_ptr = 0

    def _build_enhanced_backbone(self):
        base = torchvision.models.resnet50(pretrained=True)
        return nn.Sequential(*list(base.children())[:-1])  # 示例简版

    def _build_momentum_encoder(self):
        mom_encoder = deepcopy(self.backbone)
        mom_projector = deepcopy(self.projector)
        for param in mom_encoder.parameters():
            param.requires_grad = False
        for param in mom_projector.parameters():
            param.requires_grad = False
        return nn.Sequential(mom_encoder, mom_projector)

    def forward(self, x):
        # 在线特征
        online_feat = self.backbone(x)
        online_emb, temp = self.projector(online_feat.flatten(1))
        online_emb = F.normalize(online_emb, dim=1)

        # 动量特征
        with torch.no_grad():
            mom_feat = self.momentum_encoder[0](x)
            mom_emb, _ = self.momentum_encoder[1](mom_feat.flatten(1))
            mom_emb = F.normalize(mom_emb, dim=1)

        return online_emb, mom_emb, temp

    @torch.no_grad()
    def update_momentum_encoder(self, m=Config.momentum):
        # 动量参数更新
        for (online_param, mom_param) in zip(self.backbone.parameters(),
                                             self.momentum_encoder[0].parameters()):
            mom_param.data = mom_param.data * m + online_param.data * (1 - m)
        for (online_param, mom_param) in zip(self.projector.parameters(),
                                             self.momentum_encoder[1].parameters()):
            mom_param.data = mom_param.data * m + online_param.data * (1 - m)

    @torch.no_grad()
    def enqueue_dequeue(self, keys):
        # 更新队列
        batch_size = keys.shape[0]
        ptr = self.queue_ptr
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr = (ptr + batch_size) % Config.queue_size


class HybridContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, online_emb, mom_emb, queue, temp):
        # 正样本相似度
        pos_sim = torch.einsum('nd,nd->n', online_emb, mom_emb)

        # 负样本相似度
        neg_sim = torch.einsum('nd,dk->nk', online_emb, queue)

        # 综合对比损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temp
        labels = torch.zeros(online_emb.size(0), dtype=torch.long).to(online_emb.device)
        loss_online = F.cross_entropy(logits, labels)
        # 动量编码器视角下的对比（可选）
        logits_momentum = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temp.detach()
        loss_momentum = F.cross_entropy(logits_momentum, labels)
         # 混合损失
        return Config.loss_alpha * loss_online + (1 - Config.loss_alpha) * loss_momentum


def main():
    # 加载数据
    train_set = ContrastiveDataset(
        Config.train_gen_dir, Config.train_used_dir, Config.train_unused_dir
    )
    train_loader = DataLoader(
        train_set,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 初始化模型和优化器
    model = AdvancedContrastiveModel().to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-4)
    criterion = HybridContrastiveLoss()
    best_loss = 1111111
    # 训练循环
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0


        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.epochs}"):
            anchors, pos, neg = [t.to(Config.device) for t in batch]

            # 前向传播
            online_emb, mom_emb, temp = model(anchors)

            # 合并负样本
            neg_batch = neg.view(-1, neg.size(2), neg.size(3), neg.size(4))  # [B*num_neg, C, H, W]
            with torch.no_grad():
                neg_emb = model.momentum_encoder[0](neg_batch)
                neg_emb = model.momentum_encoder[1](neg_emb.flatten(1))[0].detach()
                neg_emb = F.normalize(neg_emb, dim=1)

            # 计算损失
            current_queue = torch.cat([model.queue, neg_emb.T], dim=1)[:, :Config.queue_size]
            loss = criterion(online_emb, mom_emb, current_queue, temp.mean())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新动量编码器和队列
            model.update_momentum_encoder()
            model.enqueue_dequeue(mom_emb)

            total_loss += loss.item() * anchors.size(0)

        # 保存模型
        avg_loss = total_loss / len(train_set)
        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")
        if avg_loss<best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"model_epoch{3}.pth")


if __name__ == "__main__":
    main()
