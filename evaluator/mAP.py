import numpy as np

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2g - x1g + 1) * (y2g - y1g + 1)
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou

def average_precision(recalls, precisions):
    """
    Compute the average precision, given the recall and precision curves.
    """
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def compute_map(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Compute the mean Average Precision (mAP) given the ground truth and predicted bounding boxes.
    """
    all_ap = []
    for class_id in range(len(gt_boxes)):
        gt = gt_boxes[class_id]
        pred = pred_boxes[class_id]
        
        if len(gt) == 0:
            all_ap.append(0)
            continue
        
        tp = np.zeros(len(pred))
        fp = np.zeros(len(pred))
        
        matched = np.zeros(len(gt), dtype=bool)
        
        pred = sorted(pred, key=lambda x: x[1], reverse=True)
        
        for pred_idx, (pred_box, score) in enumerate(pred):
            iou_max = 0.0
            gt_match_idx = -1
            for gt_idx, gt_box in enumerate(gt):
                iou = compute_iou(pred_box, gt_box)
                if iou > iou_max:
                    iou_max = iou
                    gt_match_idx = gt_idx
            
            if iou_max >= iou_threshold and not matched[gt_match_idx]:
                tp[pred_idx] = 1
                matched[gt_match_idx] = True
            else:
                fp[pred_idx] = 1
        
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        
        recalls = tp / len(gt)
        precisions = tp / (tp + fp)
        
        ap = average_precision(recalls, precisions)
        all_ap.append(ap)
    
    return np.mean(all_ap)

# Example usage
gt_boxes = [
    [[50, 50, 150, 150]],  # Ground truth boxes for class 0
    [[30, 30, 70, 70]],    # Ground truth boxes for class 1
]

pred_boxes = [
    [([55, 55, 145, 145], 0.9)],  # Predicted boxes for class 0 with confidence scores
    [([25, 25, 75, 75], 0.8)],    # Predicted boxes for class 1 with confidence scores
]

map_score = compute_map(gt_boxes, pred_boxes)
return (f"mAP: {map_score:.4f}")
