import os
import re
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ====================================
# CONFIG
# ====================================
DATA_DIR = "./images"
IMG_SIZE = 48
PATCH_SIZE = 6
NUM_CLASSES = 40
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "vit48_best.pth"

# ====================================
# DATASET
# ====================================
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith(".png")]
        self.transform = transform
        # Extract label from filename pattern
        self.pattern = re.compile(r"_label(\d+)\.png$")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        match = self.pattern.search(filename)
        if not match:
            raise ValueError(f"Filename {filename} does not match pattern")
        label = int(match.group(1))
        path = os.path.join(self.root_dir, filename)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, filename

# ====================================
# TRANSFORMS
# ====================================
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# ====================================
# PATCH EMBEDDING + VIT BLOCK
# ====================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=48, patch_size=6, in_chans=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class ViTBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=48, patch_size=6, num_classes=40, embed_dim=256, depth=6, num_heads=4, drop=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio=4.0, drop=drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

# ====================================
# TRAINING & EVALUATION
# ====================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct = 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)

# ====================================
# ATTENTION HEATMAP
# ====================================
def get_attention_map(model, img_tensor):
    model.eval()
    with torch.no_grad():
        x = model.patch_embed(img_tensor)
        B = x.size(0)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + model.pos_embed
        x = model.pos_drop(x)
        for blk in model.blocks[:-1]:
            x = blk(x)
        # last block with attn
        normed = model.blocks[-1].norm1(x)
        attn_out, attn_weights = model.blocks[-1].attn(normed, normed, normed, need_weights=True)
        attn_map = attn_weights.mean(1)[0, 0, 1:]
        num_patches = attn_map.shape[0]
        patch_size = int(np.sqrt(num_patches))
        attn_map = attn_map.reshape(patch_size, patch_size).cpu().numpy()
        attn_map -= attn_map.min()
        attn_map /= attn_map.max()
        attn_map = cv2.resize(attn_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return attn_map

def show_attention(img_tensor, attn_map):
    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 0.5 + 0.5).clip(0, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = (0.5 * img_np + 0.5 * heatmap).clip(0, 1)
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

# ====================================
# MAIN
# ====================================
if __name__ == "__main__":
    dataset = CustomImageDataset(DATA_DIR, transform=train_transform)
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ViT(IMG_SIZE, PATCH_SIZE, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT)

        # Show attention maps for 3 random validation samples
        samples = random.sample(range(len(test_dataset)), min(3, len(test_dataset)))
        for idx in samples:
            img, label, fname = test_dataset[idx]
            attn_map = get_attention_map(model, img.unsqueeze(0).to(DEVICE))
            print(f"Sample: {fname} | Label: {label}")
            show_attention(img.unsqueeze(0), attn_map)
