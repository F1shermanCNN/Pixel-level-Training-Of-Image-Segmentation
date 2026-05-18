import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        ## 防止数学错误
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: [B, C, H, W] (raw logits)
        targets: [B, H, W] (ground truth labels)
        """
        num_classes = preds.shape[1]
         # Step 1: Softmax
        preds = F.softmax(preds, dim=1)

        # Step 2: One-hot encode target
        targets_one_hot = F.one_hot(targets, num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Step 3: Flatten batch & spatial dims
        preds_flat = preds.contiguous().view(preds.shape[0], preds.shape[1], -1)
        targets_flat = targets_one_hot.contiguous().view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)

        # Step 4: Compute per-class Dice
        intersection = (preds_flat * targets_flat).sum(-1)
        union = preds_flat.sum(-1) + targets_flat.sum(-1)

        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score  # Dice loss per class

        return dice_loss.mean()
    
class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(CombinedLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, preds, targets):

        ce = self.ce_loss(preds, targets)

        dice = self.dice_loss(preds, targets)

        total_loss = (
            self.weight_ce * ce
            + self.weight_dice * dice
        )

        return total_loss