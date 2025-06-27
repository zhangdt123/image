import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0, vit_b_16
from PIL import Image
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from itertools import chain

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_gen_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/generated"
    train_used_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_used"
    train_unused_dir = "imageclef data/ImageCLEF25_GAN_Detect_Training_Data_Usage_training-dataset/real_not_used"
    num_neg_samples = 1
    batch_size = 128
    lr = 3e-4
    epochs = 50
    queue_size = 65536

    sim_threshold = 0.65
    min_pos_samples = 1
    proj_hidden_dims = [1024, 512, 256]
    momentum = 0.999
    loss_alpha = 0.7

class ContrastiveDataset(Dataset):
    def __init__(self, gen_dir, used_dir, unused_dir):
        self._validate_dirs(gen_dir, used_dir, unused_dir)

        self.sim_matrix = np.load("sim_matrix1.npy")
        self.gen_paths = sorted(os.path.join(gen_dir, f) for f in os.listdir(gen_dir))
        self.used_paths = sorted(os.path.join(used_dir, f) for f in os.listdir(used_dir))
        self.unused_paths = sorted(os.path.join(unused_dir, f) for f in os.listdir(unused_dir))

        self.positive_pools = self._build_positive_pools(Config.sim_threshold, Config.min_pos_samples)

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
        gen_img = self._load_image(self.gen_paths[idx])
        pos_ids = np.random.choice(self.positive_pools[idx], Config.min_pos_samples, replace=False)
        positives = [self._load_image(self.used_paths[i]) for i in pos_ids]
        negs = self._get_random_negatives(Config.num_neg_samples)
        return gen_img, torch.stack(positives), torch.stack(negs)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def _get_random_negatives(self, num):
        ids = np.random.choice(len(self.unused_paths), num, replace=False)
        return [self._load_image(self.unused_paths[i]) for i in ids]

class DynamicProjection(nn.Module):
    def __init__(self, in_dim=512, hidden_dims=[1024, 512, 256], use_bn=True):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim


        self.temp_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
        nn.ReLU(),
        nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        embeddings = self.mlp(x)
        temperature = 0.05 + 0.15 * self.temp_layer(embeddings)
        return embeddings, temperature

class MOEContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 各个专家
        self.resnet = nn.Sequential(*list(resnet50(pretrained=True).children())[:-1])
        self.efficientnet = nn.Sequential(*list(efficientnet_b0(pretrained=True).children())[:-1])
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()

        # 用线性层统一各专家输出到2048维
        self.resnet_fc = nn.Identity()
        self.efficientnet_fc = nn.Linear(1280, 2048)
        self.vit_fc = nn.Linear(768, 2048)

        self.num_experts = 3

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(2048 * 3, 512),  # [修改] 增加隐藏层
            nn.GELU(),                # [修改] 使用GELU激活
            nn.Linear(512, 3),
            nn.Softmax(dim=1)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=2048, num_heads=8)  # [新增]
        self.projector = DynamicProjection(2048, Config.proj_hidden_dims)
        self.momentum_encoder = deepcopy(self)
        self._freeze_momentum()


        self.register_buffer("queue", torch.randn(Config.proj_hidden_dims[-1], Config.queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_ptr = 0

    def _freeze_momentum(self):
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        resnet_feat = self.resnet(x).flatten(1)
        efficientnet_feat = self.efficientnet(x).flatten(1)
        vit_feat = self.vit(x)

        resnet_feat = self.resnet_fc(resnet_feat)
        efficientnet_feat = self.efficientnet_fc(efficientnet_feat)
        vit_feat = self.vit_fc(vit_feat)

        resnet_feat = F.normalize(resnet_feat, dim=1)         # [新增]
        efficientnet_feat = F.normalize(efficientnet_feat, dim=1)  # [新增]
        vit_feat = F.normalize(vit_feat, dim=1)              # [新增]

        expert_feats = torch.stack([resnet_feat, efficientnet_feat, vit_feat], dim=1)
        gate_input = torch.cat([resnet_feat, efficientnet_feat, vit_feat], dim=1)
        gate_scores = self.gate(gate_input)

        # [新增] 交叉注意力融合
        attn_feats = expert_feats.permute(1, 0, 2)  # [S, B, D]
        attn_out, _ = self.cross_attn(attn_feats, attn_feats, attn_feats)  # [新增]
        attn_out = attn_out.permute(1, 0, 2)        # [B, S, D]

        fused_feat = torch.sum(gate_scores.unsqueeze(-1) * attn_out, dim=1)  # [修改]
        online_emb, temp = self.projector(F.normalize(fused_feat, dim=1))

        with torch.no_grad():
            mom_emb, _ = self.momentum_forward(x)

        return online_emb, mom_emb, temp, gate_scores  # [修改] 返回gate_scores

    @torch.no_grad()
    def momentum_forward(self, x):
        resnet_feat = self.momentum_encoder.resnet(x).flatten(1)
        efficientnet_feat = self.momentum_encoder.efficientnet(x).flatten(1)
        vit_feat = self.momentum_encoder.vit(x)

        resnet_feat = self.momentum_encoder.resnet_fc(resnet_feat)
        efficientnet_feat = self.momentum_encoder.efficientnet_fc(efficientnet_feat)
        vit_feat = self.momentum_encoder.vit_fc(vit_feat)

        expert_feats = torch.stack([resnet_feat, efficientnet_feat, vit_feat], dim=1)
        gate_input = torch.cat([resnet_feat, efficientnet_feat, vit_feat], dim=1)
        gate_scores = self.momentum_encoder.gate(gate_input)

        fused_feat = torch.sum(gate_scores.unsqueeze(-1) * expert_feats, dim=1)
        emb, temp = self.momentum_encoder.projector(F.normalize(fused_feat, dim=1))
        return F.normalize(emb, dim=1), temp

    @torch.no_grad()
    def update_momentum_encoder(self):
        base_m = Config.momentum
        for (name_q, param_q), (name_k, param_k) in zip(self.named_parameters(),
                                                       self.momentum_encoder.named_parameters()):            # [新增] 分层动量更新策略
            m = base_m if 'projector' in name_q else 0.999  # 骨干网络更新更慢
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def enqueue_dequeue(self, keys):
        keys = keys.detach()
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= Config.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            end_len = Config.queue_size - ptr
            self.queue[:, ptr:] = keys[:end_len].T
            self.queue[:, :batch_size - end_len] = keys[end_len:].T

        ptr = (ptr + batch_size) % Config.queue_size
        self.queue_ptr = ptr

class HybridContrastiveLoss(nn.Module):
    def forward(self, online_emb, mom_emb, queue, temp, gate_scores):
        pos_sim = torch.einsum('nd,nd->n', online_emb, mom_emb)
        neg_sim = torch.einsum('nd,dk->nk', online_emb, queue)

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temp
        labels = torch.zeros(online_emb.size(0), dtype=torch.long).to(online_emb.device)
        loss_online = F.cross_entropy(logits, labels)

        logits_momentum = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temp.detach()
        loss_momentum = F.cross_entropy(logits_momentum, labels)

        entropy = -torch.mean(torch.sum(gate_scores * torch.log(gate_scores + 1e-10), dim=1))
        total_loss = Config.loss_alpha * loss_online + (1 - Config.loss_alpha) * loss_momentum + 0.1 * entropy
        return total_loss

def main():
    train_set = ContrastiveDataset(Config.train_gen_dir, Config.train_used_dir, Config.train_unused_dir)
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = MOEContrastiveModel().to(Config.device)
    optimizer = torch.optim.AdamW([
        {'params': model.resnet.parameters(), 'lr': Config.lr/10},     # [新增]
        {'params': model.efficientnet.parameters(), 'lr': Config.lr/10},  # [新增]
        {'params': model.vit.parameters(), 'lr': Config.lr/10},        # [新增]
        {'params': chain(model.gate.parameters(),
                         model.projector.parameters(),
                        model.cross_attn.parameters()),  # [新增]
         'lr': Config.lr}
    ], weight_decay=1e-4)
    criterion = HybridContrastiveLoss()

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0
        best_loss = 1e8

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.epochs}"):
            anchors, pos, neg = [t.to(Config.device) for t in batch]

            online_emb, mom_emb, temp, gate_scores = model(anchors)  # [修改] 接收gate_scores

            neg_batch = neg.view(-1, neg.size(2), neg.size(3), neg.size(4))
            with torch.no_grad():
                neg_emb, _ = model.momentum_forward(neg_batch)

            current_queue = torch.cat([model.queue, neg_emb.T], dim=1)[:, :Config.queue_size]
            loss = criterion(online_emb, mom_emb, current_queue, temp.mean(), gate_scores)  # [修改]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update_momentum_encoder()
            model.enqueue_dequeue(mom_emb)

            total_loss += loss.item() * anchors.size(0)

        avg_loss = total_loss / len(train_set)
        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"moe_model_best5.pth")

if __name__ == "__main__":
    main()