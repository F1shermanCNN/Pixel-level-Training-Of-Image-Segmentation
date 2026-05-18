import os
import csv
import torch
import numpy as np
from tqdm import tqdm

from models.blocks import UNet
from dataset import get_train_val_dataloaders, get_test_dataloader


class SegmentationEvaluator:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, pred, target):
        mask = (target >= 0) & (target < self.num_classes)
        label = self.num_classes * target[mask].astype(int) + pred[mask].astype(int)
        count = np.bincount(label, minlength=self.num_classes**2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)

    def get_results(self):
        cm = self.confusion_matrix
        total_correct = np.diag(cm).sum()
        total_pixels = cm.sum()
        pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0.0

        intersection = np.diag(cm)
        union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
        with np.errstate(divide='ignore', invalid='ignore'):
            ious = intersection / union
        valid_indices = (union > 0)
        miou = np.mean(ious[valid_indices]) if np.any(valid_indices) else 0.0

        return pixel_acc, miou

    def reset(self):
        self.confusion_matrix.fill(0)

# evaluate
@torch.no_grad()
def evaluate(model, loader, device, num_classes=3, return_miou=False):
    model.eval()
    evaluator = SegmentationEvaluator(num_classes=num_classes)

    acc_list = []

    for images, masks in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        if return_miou:
            evaluator.update(preds.cpu().numpy(), masks.cpu().numpy())
        else:
            acc = (preds == masks).sum().item() / masks.numel()
            acc_list.append(acc)

    if return_miou:
        val_acc, val_miou = evaluator.get_results()
        return val_acc, val_miou
    else:
        return np.mean(acc_list)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders
    train_loader, val_loader = get_train_val_dataloaders(
        batch_size=16,
        image_size=256,
        num_workers=0
    )

    test_loader = get_test_dataloader(
        batch_size=16,
        image_size=256,
        num_workers=0
    )

    experiments = {
        # "new_full_run": "checkpoints/entrophyloss/best.pth",
        "full_run": "checkpoints/full_run/best.pth",
        "combined_loss": "checkpoints/full_run_combinedloss/best.pth",
        "dice_loss": "checkpoints/full_run_diceloss/best.pth"
    }

    results = []

    for name, ckpt_path in experiments.items():
        if not os.path.exists(ckpt_path):
            print(f"\n[Warning] {ckpt_path} 不存在，跳过该实验。")
            continue

        print(f"\nEvaluating {name} ...")

        # model
        model = UNet(in_channels=3, num_classes=3).to(device)

        # 加载权重
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        # 验证集：计算 mIoU
        val_acc, val_miou = evaluate(model, val_loader, device, num_classes=3, return_miou=True)
        # 测试集：只计算 pixel accuracy
        test_acc = evaluate(model, test_loader, device, num_classes=3, return_miou=False)

        print(f"""
            {name}
            Val Acc : {val_acc:.4f}
            Val mIoU: {val_miou:.4f}
            Test Acc: {test_acc:.4f}
        """)

        results.append([name, val_acc, val_miou, test_acc])

    # save CSV
    os.makedirs("checkpoints", exist_ok=True)
    csv_path = "checkpoints/evaluation_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "val_acc", "val_mIoU", "test_acc"])
        writer.writerows(results)

    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()