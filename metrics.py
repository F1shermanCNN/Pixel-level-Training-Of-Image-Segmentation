import torch


def compute_miou(preds, targets, num_classes, smooth=1e-6):
    """
    preds: [B, C, H, W] (raw logits)
    targets: [B, H, W]
    """

    # convert logits -> predicted class
    preds = torch.argmax(preds, dim=1)

    iou_per_class = []

    for cls in range(num_classes):

        pred_cls = (preds == cls)
        target_cls = (targets == cls)

        intersection = (pred_cls & target_cls).sum().float()

        union = (pred_cls | target_cls).sum().float()

        iou = (intersection + smooth) / (union + smooth)

        iou_per_class.append(iou)

    miou = torch.mean(torch.tensor(iou_per_class))

    return miou.item()