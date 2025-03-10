import os
import json
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import timm  
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents Tkinter error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tqdm import tqdm
import wandb

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, image_root, annotation_root, transform=None,n_sample=3):
        self.image_root = image_root
        self.annotation_root = annotation_root
        self.transform = transform
        self.data = self.load_data()
        self.n_sample=n_sample

    def load_data(self):
        self.data = []
        self.annot = {}
        self.img = {}
        for seq in os.listdir(self.image_root):
            self.annot[seq]=defaultdict(list)
            self.img[seq]={}

            ann_path = os.path.join(self.annotation_root, seq, "thermal", "COCO", "annotations.json")
            
            if not os.path.exists(ann_path):
                continue

            with open(ann_path, 'r') as f:
                annotations = json.load(f)

            for img_info in annotations.get("images", []):
                img_id = img_info['id']
                self.img[seq][img_id]=img_info

            for anno_info in annotations.get("annotations", []):
                track_id = anno_info['track_id']
                self.annot[seq][track_id].append(anno_info)
            
            for track_id in self.annot[seq].keys():
                self.data.append((seq,track_id))
                
        
        return self.data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq, track_id = self.data[idx]
        annots = self.annot[seq][track_id]
        num_annots = len(annots)

        # 샘플 개수가 부족한 경우 처리
        if num_annots < self.n_sample:
            # 부족한 경우, 반복해서 샘플링
            sample_ids = np.tile(np.arange(num_annots), int(np.ceil(self.n_sample / num_annots)))[:self.n_sample]
        else:
            # 랜덤한 시작 인덱스에서 연속 샘플링
            start_idx = np.random.randint(0, num_annots - self.n_sample + 1)
            sample_ids = np.arange(start_idx, start_idx + self.n_sample)

        sample_img = []
        sample_tag_id = []
        
        for sample_id in sample_ids:
            annot_info = annots[sample_id]

            img_id = annot_info['image_id']
            img_info = self.img[seq][img_id]

            img_path = os.path.join(self.image_root, seq, 'thermal', img_info['file_name'])

            w, h = img_info['width'], img_info['height']
            bbox = annot_info['bbox']

            image = Image.open(img_path)
            x, y, bw, bh = bbox
            cropped_image = image.crop((x, y, x + bw, y + bh))
            if self.transform:
                cropped_image = self.transform(cropped_image)

            sample_img.append(cropped_image[[0]])  # 채널 선택
            sample_tag_id.append(idx)
        
        return torch.stack(sample_img), idx

# 임베딩 모델 정의
class EmbeddingModel(nn.Module):
    def __init__(self,  model_name='resnet18',
                        embedding_dim=128, 
                        use_pretrained_model=True,):
        super(EmbeddingModel, self).__init__()
        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=use_pretrained_model)
        if model_name == 'resnet50':
            base_model = models.resnet50(pretrained=use_pretrained_model)
        elif model_name == 'efficientnet_b0':
            base_model = timm.create_model('efficientnet_b0', pretrained=use_pretrained_model)
        elif model_name == 'efficientnet_b4':
            base_model = timm.create_model('efficientnet_b4', pretrained=use_pretrained_model)
        elif model_name == 'efficientnet_b7':
            base_model = timm.create_model('efficientnet_b7', pretrained=use_pretrained_model)
        elif model_name == 'dinov2_small':
            base_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        elif model_name == 'dinov2_base':
            base_model = timm.create_model("facebookresearch/dinov2_vitb14", pretrained=use_pretrained_model)
        elif model_name == 'dinov2_large':
            base_model = timm.create_model("facebookresearch/dinov2_vitl14", pretrained=use_pretrained_model)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # ResNet 기반 모델인 경우
        if model_name in ['resnet18', 'resnet50']:
            in_channels = 1
            out_channels = base_model.conv1.out_channels
            kernel_size = base_model.conv1.kernel_size
            stride = base_model.conv1.stride
            padding = base_model.conv1.padding
            bias = base_model.conv1.bias is not None

            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # 마지막 FC layer 제거
            
            # conv1의 입력을 1채널로 변경
            self.feature_extractor[0] = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                                  stride=stride, padding=padding, bias=bias)
            # conv1 가중치 초기화
            with torch.no_grad():
                if base_model.conv1.weight.shape[1] == 3:
                    self.feature_extractor[0].weight[:] = base_model.conv1.weight.mean(dim=1, keepdim=True)
                else:
                    nn.init.kaiming_normal_(self.feature_extractor[0].weight, mode='fan_out', nonlinearity='relu')

            feature_dim = base_model.fc.in_features  # ResNet의 마지막 feature 크기

        # EfficientNet models
        elif "efficientnet" in model_name:
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # Remove last FC layer
            feature_dim = base_model.num_features  # EfficientNet feature dimension

        # DINOv2 기반 모델인 경우
        else:
            self.base_model = base_model
            feature_dim = base_model.embed_dim  # DINOv2 feature dimension

        self.embedding = nn.Linear(feature_dim, embedding_dim)

    def forward(self, x):
        
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # [B, 1, H, W] → [B, 3, H, W]

        # Use correct feature extraction method for DINOv2
        if hasattr(self, "feature_extractor"):
            features = self.feature_extractor(x)
        else:
            with torch.no_grad():
                dino_features = self.base_model.forward_features(x)  # For DINOv2
                features = dino_features["x_norm_clstoken"]  # Use CLS token

        features = torch.flatten(features, 1)
        embedding = self.embedding(features)
        return F.normalize(embedding, p=2, dim=1)  # L2 정규화 추가

# Triplet Loss 정의
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin)
    
    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

def shuffle_without_same_position(tensor):
    while True:
        shuffled = tensor[torch.randperm(tensor.size(0))]  # 무작위 섞기
        if not torch.any(shuffled == tensor):  # 같은 위치 유지된 경우 확인
            return shuffled  # 조건 충족 시 반환
  
# 검증 단계에서 PCA + Clustering 시각화 추가
def visualize_embeddings(features, tags, save_path="pca_result.png"):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    # num_clusters = len(set(tags))
    # kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(reduced_features)
    cluster_labels = tags#kmeans.labels_
    
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='tab20', alpha=0.7)
    plt.colorbar()
    # plt.legend()
    plt.title("PCA Clustering Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(save_path)
    plt.close()

# t-SNE 시각화 함수 추가
def visualize_tsne_embeddings(features, tags, save_path="tsne_result.png"):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, max_iter=1000)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=tags, cmap='tab20', alpha=0.7)
    plt.colorbar()
    plt.title("t-SNE Clustering Visualization")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.savefig(save_path)
    plt.close()


# 학습 및 검증 루프
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10,use_wandb=False):
    model.train()
    if config.use_wandb:
        wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, tags in tqdm(train_loader):
        
            # ==================================
            B, NS, C, H, W = images.shape
            images = images.to(device)

            features = model(images.view(B*NS, C, H, W)).view(B, NS,-1) # (0,0,... 1,1,... 2,...)

            anchor_batch = features[:,0,:].repeat_interleave(NS, dim=0)
            positive_batch = features.view(B*NS, -1)

            negative_ids = shuffle_without_same_position(torch.arange(len(tags)))
            negative_batch = features[negative_ids].view(B*NS, -1)
            
            loss = criterion(anchor_batch, positive_batch, negative_batch)

            # ==================================
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        all_features=[]
        all_tags = []
        with torch.no_grad():
            for e, (images, tags) in tqdm(enumerate(val_loader)):
                # ==================================
                B, NS, C, H, W = images.shape
                images = images.to(device)

                features = model(images.view(B*NS, C, H, W)).view(B, NS,-1) # (0,0,... 1,1,... 2,...)

                anchor_feat = features[:,0,:].repeat_interleave(NS, dim=0)
                positive_feat = features.view(B*NS, -1)

                negative_ids = shuffle_without_same_position(torch.arange(len(tags)))
                negative_feat = features[negative_ids].view(B*NS, -1)
                
                val_loss += criterion(anchor_feat, positive_feat, negative_feat)

                # ==================================
                if e<2:
                    all_features.append(features.view(B*NS, -1).cpu().numpy().astype(np.float16))
                    all_tags.extend(tags.repeat_interleave(NS, dim=0).numpy())
                
        all_features = np.vstack(all_features)
        all_tags = [int(n) for n in all_tags]
        visualize_tsne_embeddings(all_features, all_tags, save_path="./result/tsne_result.png")
        visualize_embeddings(all_features, all_tags, save_path="./result/pca_result.png")
        if epoch%10==0 and epoch!=0:
            visualize_tsne_embeddings(all_features, all_tags, save_path=f"./result/tsne_result_e{epoch:04d}.png")
            visualize_embeddings(all_features, all_tags, save_path=f"./result/pca_result_e{epoch:04d}.png")


        if config.use_wandb:
            wandb.log({
                "Epoch": epoch + 1,
                "Train Loss": running_loss / len(train_loader),
                "Validation Loss": val_loss / len(val_loader),
                "PCA Visualization": wandb.Image("./result/pca_result.png"),
                "t-SNE Visualization": wandb.Image("./result/tsne_result.png"),
            })

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}", end= "\t")
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        model.train()

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Triplet Loss Clustering Configuration")

    parser.add_argument("--model_name", type=str, default='efficientnet_b0', help="Model name")
    parser.add_argument("--device", type=str, default='cuda', help="Device")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimension of embedding vector")
    parser.add_argument("--loss_fn", type=str, default='triplet', help="Loss function: ['triplet','cosine']")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for Triplet Loss")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--use_pretrained", action="store_true", help="Use Pretrained weights")
    

    args = parser.parse_args(["--use_wandb","--use_pretrained"])
    return args

base_path = "./tmot_dataset_challenge/tmot_dataset_challenge/"

# 설정값 가져오기
config = get_args()
if config.use_wandb:
    wandb.init(project="Thermal-MOT-CVPR-2025", name=f"{config.model_name}-embedding{config.embedding_dim}", config=config)


# 모델 초기화 및 학습 설정
device = torch.device(config.device if torch.cuda.is_available() else "cpu")
model = EmbeddingModel(model_name=config.model_name,
                       embedding_dim=config.embedding_dim,
                       use_pretrained_model=config.use_pretrained).to(device)
criterion = TripletLoss(margin=config.margin)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)


# 데이터 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터 로더 설정
train_dataset = CustomDataset(os.path.join(base_path,"images/train"), os.path.join(base_path,"annotations/train"), transform, n_sample=16)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

val_dataset = CustomDataset(os.path.join(base_path,"images/val"), os.path.join(base_path,"annotations/val"), transform, n_sample=10)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4)

# 학습 실행
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=config.epochs, use_wandb=config.use_wandb)

# 모델 저장
ckpt = {'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'epoch':config.epochs}
torch.save(ckpt, "checkpoints.pth")
