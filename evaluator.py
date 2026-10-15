import numpy as np
import cv2

def calculate_iou(box1, box2):
    """计算两个边界框的 IoU，输入格式 [xmin, ymin, xmax, ymax]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def calculate_dice(pred_mask, gt_mask):
    """计算分割掩膜的 Dice 系数"""
    # 确保掩膜是二值化的 (0 或 1)
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    
    if union == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
        
    return 2.0 * intersection / union