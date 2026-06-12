import numpy as np

def iou_score(output, target):
    # output shape: [N, C, H, W]
    # target shape: [N, C, H, W]

    # Convert to boolean tensors
    output = output.bool()
    target = target.bool()

    intersection = (output & target).float().sum((1, 2))  # Sum over (H, W)

    # Compute Union
    union = (output | target).float().sum((1, 2))  # Sum over (H, W)

    # Compute IoU
    iou = (intersection + 1e-6) / (union + 1e-6) 
    iou_sum = iou.sum()
    return {'sum' : iou_sum} # .mean()  # Return mean IoU score for batch
