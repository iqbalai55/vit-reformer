import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute cross-entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute probability of correct class
        pt = torch.exp(-ce_loss)  # pt is the probability of correct class

        # Compute the focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha factor if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_factor = self.alpha[targets]
            else:
                alpha_factor = self.alpha
            focal_loss = alpha_factor * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
