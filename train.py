import os
import time
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import wandb
from models.blocks import UNet
from dataset import get_train_val_dataloaders
from models.losses import DiceLoss
from models.losses import CombinedLoss

# He初始化
def init_weights_he(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# 保存CSV
def save_log_csv(path, history):
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_acc",
            "epoch_time"
        ])
        writer.writerows(history)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc="Train")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device).long()

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


# Validation
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Val")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device).long()

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        correct += (preds == masks).sum().item()
        total += masks.numel()

        loop.set_postfix(loss=loss.item())

    acc = correct / total
    return total_loss / len(loader), acc


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb 初始化
    wandb.init(
        project="unet-oxford-pet",
        name=f"lrdec_bs{args.batch_size}_lr{args.lr}_img{args.image_size}_crossentrophyloss",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "image_size": args.image_size
        }
    )

    print("Loading dataset...")

    train_loader, val_loader = get_train_val_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=0
    )

    print("Dataset ready")

    model = UNet(in_channels=3, num_classes=3).to(device)
    model.apply(init_weights_he)

    criterion = nn.CrossEntropyLoss()
    # criterion = DiceLoss()
    # criterion = CombinedLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    os.makedirs(args.save_dir, exist_ok=True)

    history = []

    #  Early Stopping 参数
    best_val_acc = 0.0       
    no_improve_count = 0
    patience = 10
    min_delta = 1e-4


    print("Start training...")

    for epoch in range(1, args.epochs + 1):

        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - start_time


        print(f"""
            Epoch [{epoch}/{args.epochs}]
            Train Loss: {train_loss:.4f}
            Val Loss  : {val_loss:.4f}
            Val Acc   : {val_acc:.4f}
            Time      : {epoch_time:.2f}s
        """)

        # log记录
        history.append([
            epoch,
            train_loss,
            val_loss,
            val_acc,
            epoch_time
        ])


        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch,
            "epoch_time": epoch_time,
            "lr": optimizer.param_groups[0]["lr"]
        })

        scheduler.step()

        #  save last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, os.path.join(args.save_dir, "last.pth"))



        # Early Stopping
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            no_improve_count = 0

            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, "best.pth")
            )
            print(f"[INFO] Val Acc improved → saving best.pth")

        else:
            no_improve_count += 1
            print(f"[INFO] No improvement in Val Acc: {no_improve_count}/{patience}")

        # stop condition
        if no_improve_count >= patience:
            print(f"\n[Early Stopping] No improvement in Val Acc for {patience} epochs → stop training\n")
            break

        # periodic save
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f"unet_epoch_{epoch}.pth")
            )

        # save csv
        save_log_csv(
            os.path.join(args.save_dir, "log.csv"),
            history
        )

    wandb.finish()

    print("Training Finished!")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    args = parser.parse_args()

    train(args)