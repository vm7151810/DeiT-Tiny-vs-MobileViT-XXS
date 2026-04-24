import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import yaml

from src.models.multitask_vit import MultiTaskViT
from src.utils.dataset import prepare_dataframe, EverydayDataset

def get_args():
    parser = argparse.ArgumentParser(description="Train Multi-Task ViT")
    parser.add_argument("--backbone", type=str, default="deit_tiny_patch16_224", help="Timm backbone name")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_root", type=str, default="data/metadata")
    parser.add_argument("--image_root", type=str, default="data/images")
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Prepare Data
    data_path = Path(args.data_root)
    df_proc, classes, attr_names, attr_value2idx = prepare_dataframe(
        data_path / "labels.csv",
        data_path / "attributes.yaml",
        data_path / "classes.txt",
        Path(args.image_root)
    )

    img_size = 224
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = EverydayDataset(df_proc, "train", transform=train_tfms, attr_names=attr_names)
    val_ds = EverydayDataset(df_proc, "val", transform=val_tfms, attr_names=attr_names)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. Build Model
    attr_sizes = {a: len(v) for a, v in attr_value2idx.items()}
    model = MultiTaskViT(args.backbone, len(classes), attr_sizes)
    model.to(device)

    # 3. Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_attr = nn.CrossEntropyLoss()

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for imgs, labels, attrs in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, labels, attrs = imgs.to(device), labels.to(device), attrs.to(device)
            
            optimizer.zero_grad()
            cls_logits, attr_logits, _ = model(imgs)
            
            loss_cls = criterion_cls(cls_logits, labels)
            loss_attr = 0
            for i, a in enumerate(attr_names):
                loss_attr += criterion_attr(attr_logits[a], attrs[:, i])
            
            loss = loss_cls + loss_attr
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")
        
        torch.save({
            'model_state': model.state_dict(),
            'classes': classes,
            'attr_value2idx': attr_value2idx,
        }, f"{args.output_dir}/last.pt")

if __name__ == "__main__":
    main()
