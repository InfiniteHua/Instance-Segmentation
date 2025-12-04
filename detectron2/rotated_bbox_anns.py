import json
import numpy as np
import cv2
import os

# 输入 COCO 格式 annotation 文件
input_json = "flowers/train/_annotations.coco.json"
output_json = "flowers/train/_annotations_rotated.coco.json"

with open(input_json, "r") as f:
    coco = json.load(f)

for ann in coco["annotations"]:
    # 获取 mask 多边形 points
    # segmentation 是 list of list，例如 [[x1,y1,x2,y2,...]]
    segs = ann["segmentation"]
    all_points = []
    for poly in segs:
        points = np.array(poly).reshape(-1, 2)
        all_points.append(points)
    if not all_points:
        print(f"Warning: empty segmentation for annotation id {ann['id']}, skipping...")
        continue
    all_points = np.concatenate(all_points, axis=0).astype(np.float32)  # shape (N,2)

    # 计算最小外接旋转矩形
    rect = cv2.minAreaRect(all_points)  # ((cx,cy),(w,h),angle)
    (cx, cy), (w, h), theta = rect

    # COCO RotatedBoxes 一般要求 w>=h，angle 逆时针旋转
    if w < h:
        w, h = h, w
        theta = theta + 90.0

    theta = -theta
    theta = theta % 360

    # 更新 annotation 的 bbox 为 [cx, cy, w, h, theta]
    ann["bbox"] = [float(cx), float(cy), float(w), float(h), float(theta)]

# 保存新的 annotation 文件
with open(output_json, "w") as f:
    json.dump(coco, f, indent=2)

print(f"Rotated bbox annotations saved to {output_json}")

