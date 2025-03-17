import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self,pos_weight=5.):
        super().__init__()
        self.pos_weight=pos_weight
    def __call__(self, preds, targets):
        bce=binary_crossentropy(torch.sigmoid(preds), targets,pos_weight=self.pos_weight)
        dice=sigmoid_dice_loss(preds, targets)
        return 2 * bce + dice
    
def binary_crossentropy(pr, gt, eps=1e-7, pos_weight=1., neg_weight=1.):
    pr = torch.clamp(pr, eps, 1. - eps)
    gt = torch.clamp(gt, eps, 1. - eps)
    loss = - pos_weight * gt * torch.log(pr) -  neg_weight * (1 - gt) * torch.log(1 - pr)
    return torch.mean(loss)

#modified from detrex losses:
def sigmoid_dice_loss(
    preds,
    targets,
    eps: float = 1e-5,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        preds (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor):
            A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-4.

    Return:
        torch.Tensor: The computed dice loss.
    """
    preds=torch.sigmoid(preds)
    preds = preds.flatten(1)#第二个维度及以后的所有元素都被展平为一个维度
    targets = targets.flatten(1).float()
    numerator = 2 * torch.sum(preds * targets, 1) + eps
    denominator = torch.sum(preds, 1) + torch.sum(targets, 1) + eps
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()
