import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from models.blocks import UNet
from dataset import get_train_val_dataloaders

def calculate_iou(pred, target, num_classes=3):
    """计算单张图像的平均 Intersection over Union (mIoU)"""
    ious = []
    pred = pred.flatten()
    target = target.flatten()
    for cls in range(num_classes):
        intersection = np.logical_and(pred == cls, target == cls).sum()
        union = np.logical_or(pred == cls, target == cls).sum()
        if union == 0:
            ious.append(1.0) 
        else:
            ious.append(intersection / union)
    return np.mean(ious)


def evaluate_and_analyze(weight_path, val_loader, device, num_classes=3, total_cases=10):
    model = UNet(in_channels=3, num_classes=num_classes) 
    
    print(f"Loading best weights from: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    case_studies = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            images_np = images.cpu().numpy()

            for b in range(images_np.shape[0]):
                p = preds_np[b]
                t = targets_np[b]
                img = images_np[b]

                mask = (t >= 0) & (t < num_classes)
                all_preds.append(p[mask])
                all_targets.append(t[mask])


                miou = calculate_iou(p, t, num_classes)
                
                case_studies.append({
                    'image': img,
                    'target': t,
                    'pred': p,
                    'miou': miou,
                    'id': f"batch_{idx}_idx_{b}"
                })

    # 混淆矩阵
    print("Generating Confusion Matrix...")
    flat_preds = np.concatenate(all_preds)
    flat_targets = np.concatenate(all_targets)
    
    cm = confusion_matrix(flat_targets, flat_preds, labels=list(range(num_classes)))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    class_names = ['Pet Body', 'Background', 'Contour'] 
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".4f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix (Combined Loss)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    
    os.makedirs('evaluation_results', exist_ok=True)
    cm_path = 'evaluation_results/confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # 选 10 张错例
    print(f"Sampling {total_cases} cases across different error severities...")
    # 按照 mIoU 从低到高排序
    case_studies.sort(key=lambda x: x['miou'])
    
    total_sampled = len(case_studies)
    sampled_indices = np.linspace(0, total_sampled - 1, total_cases, dtype=int)
    
    for rank, idx in enumerate(sampled_indices):
        case = case_studies[idx]
        miou_val = case['miou']
        
        if rank < 3:
            severity = "Worst (Severe Error)"
        elif rank < 7:
            severity = "Median (Average Flaw)"
        else:
            severity = "Minor (Fine Detail Error)"

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 恢复图像显示格式 [C, H, W] -> [H, W, C]
        img_show = np.transpose(case['image'], (1, 2, 0))
        img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)
        

        axes[0].imshow(img_show)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
    
        axes[1].imshow(case['target'], cmap='tab10', vmin=0, vmax=9)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(case['pred'], cmap='tab10', vmin=0, vmax=9)
        axes[2].set_title(f"Prediction (mIoU: {miou_val:.4f})")
        axes[2].axis('off')
        
        plt.suptitle(f"Case {rank+1}/10 [{severity}] - ID: {case['id']}", fontsize=14)
        
        # 保存文件
        error_path = f'evaluation_results/case_{rank+1:02d}_{severity.split()[0].lower()}.png'
        plt.savefig(error_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved Case {rank+1:02d} | Severity: {severity:<25} | mIoU: {miou_val:.4f} -> {error_path}")

    print("\nAll tasks completed! Check the 'evaluation_results' folder.")


if __name__ == '__main__':
    WEIGHT_PATH = "checkpoints/full_run_combinedloss/best.pth"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading validation dataloader...")
    _, val_loader = get_train_val_dataloaders(batch_size=4)
    
    evaluate_and_analyze(
        weight_path=WEIGHT_PATH, 
        val_loader=val_loader, 
        device=DEVICE, 
        num_classes=3, 
        total_cases=10
    )