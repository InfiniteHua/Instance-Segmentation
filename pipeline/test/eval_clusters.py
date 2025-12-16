import json
import sys
import os
from collections import defaultdict
import numpy as np
import cv2
from typing import List, Dict, Tuple, Any
import random
from statistics import mean

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from stem_pairing import stem_pairing

# -----------------------
# helpers
# -----------------------
def rasterize_polygon(segmentation: List[float], height: int, width: int) -> np.ndarray:
    """
    segmentation: COCO polygon (flat list) OR list of lists (we handle one polygon case).
    Return binary mask uint8 (0/1) shape (height, width).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    # segmentation may be list of lists or flat list
    if not segmentation:
        return mask
    # segmentation might be like [[x1,y1,x2,y2,...]] or [x1,y1,...]
    if isinstance(segmentation[0], list):
        polys = segmentation
    else:
        polys = [segmentation]

    for poly in polys:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        # clip to image bounds
        pts[:,0] = np.clip(pts[:,0], 0, width-1)
        pts[:,1] = np.clip(pts[:,1], 0, height-1)
        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask

def iou_mask(m1: np.ndarray, m2: np.ndarray) -> float:
    """Compute IoU for binary masks (uint8 or bool). Return 0 if union==0."""
    if m1.shape != m2.shape:
        raise ValueError("masks must have same shape")
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)

# -----------------------
# Load GT from COCO grouped JSON
# -----------------------
def load_coco_grouped_masks(json_path: str, images_root: str = None) -> Tuple[Dict[int, dict], Dict[int, list]]:
    """
    Parse COCO grouped json and rasterize GT masks.
    Returns:
      images_meta: {image_id: {"file_name":..., "height":..., "width":...}}
      gt_by_image: {image_id: [ann_dict,...]} where ann_dict contains:
          { "ann_id", "group_id", "category_id", "bbox", "mask" (uint8 array) }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images_meta = {}
    for img in data.get("images", []):
        images_meta[img["id"]] = {
            "file_name": img.get("file_name"),
            "height": img.get("height"),
            "width": img.get("width"),
        }

    gt_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        image_id = ann["image_id"]
        meta = images_meta.get(image_id)
        if meta is None:
            # skip if image missing
            continue
        h, w = meta["height"], meta["width"]
        mask = rasterize_polygon(ann.get("segmentation", []), h, w)
        gt_by_image[image_id].append({
            "ann_id": ann["id"],
            "group_id": ann.get("group_id"),
            "category_id": ann.get("category_id"),
            "bbox": ann.get("bbox"),
            "mask": mask
        })
    return images_meta, gt_by_image

# -----------------------
# Utilities to build cluster union masks
# -----------------------
def union_masks_of_annotations(anns: List[dict], height: int, width: int) -> np.ndarray:
    """Given list of ann dicts (each with 'mask'), return union mask."""
    if not anns:
        return np.zeros((height, width), dtype=np.uint8)
    um = np.zeros((height, width), dtype=np.uint8)
    for a in anns:
        um = np.logical_or(um, a["mask"]).astype(np.uint8)
    return um

def build_gt_clusters(gt_by_image: Dict[int, list]) -> Dict[int, Dict[int, List[dict]]]:
    """
    Build mapping: per image_id -> { group_id: [ann_dicts] }
    """
    out = {}
    for image_id, anns in gt_by_image.items():
        groups = defaultdict(list)
        for a in anns:
            gid = a.get("group_id", None)
            groups[gid].append(a)
        out[image_id] = groups
    return out

# -----------------------
# (A) Evaluate only matching (given predicted groups of GT annotations ids)
# -----------------------
def evaluate_matching_only(gt_clusters_by_image: Dict[int, Dict[int, List[dict]]],
                           predicted_groups_by_image: Dict[int, List[List[int]]],
                           images_meta: Dict[int, dict],
                           final_anns,
                           iou_thresh: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate predicted clusters (list of lists of ann_id) against GT clusters.
    Inputs:
      gt_clusters_by_image: image_id -> { group_id: [ann_dicts] }
      predicted_groups_by_image: image_id -> [ [ann_id_1, ann_id_2, ...], ... ]
          NOTE: predicted groups reference GT ann_id (i.e., the output of your matching
          post-processing that groups existing GT masks). This is the 'post-processing only' case.
    Returns: metrics dict containing precision, recall, f1, and per-cluster IoU list.
    """
    total_gt = 0
    total_pred = 0
    matched = 0
    per_pair = []  # (image_id, pred_idx, gt_gid, iou)

    # For each image, build union masks for GT groups and for each predicted group (which is list of ann ids)
    for image_id, gt_groups in gt_clusters_by_image.items():
        meta = images_meta.get(image_id)
        # if image_id not in final_anns:
        #     print(f"⚠ Skip image_id {image_id} — no corresponding image file processed.")
        #     continue

        if meta is None:
            continue
        h, w = meta["height"], meta["width"]

        # build GT union masks dict: gid -> mask
        final_ids = {ann["id"] for ann in final_anns} #this is for single image
        # final_ids = {ann["id"] for ann in final_anns[image_id]} #this is for multiple images
        gt_union = {}
        gt_clusters_filtered = {}
        for gid, anns in gt_groups.items():
            # valid_anns = [a for a in anns if a in final_anns]
            valid_anns = [a for a in anns if a["ann_id"] in final_ids]
            if not valid_anns:
                continue
            gt_union[gid] = union_masks_of_annotations(valid_anns, h, w)
            gt_clusters_filtered[gid] = valid_anns
            
        gt_gids = list(gt_union.keys())
        total_gt += len(gt_gids)

        preds = predicted_groups_by_image.get(image_id, [])
        total_pred += len(preds)

        # Uncommon for debug
        # print("📌 Ground Truth Clusters (Filtered):")
        # for gid, anns in gt_clusters_filtered.items():
        #     ann_ids = [a["ann_id"] for a in anns]
        #     print(f"  GT[{gid}]: {ann_ids}")

        # print("\n📌 Predicted Clusters:")
        # for pi, group in enumerate(preds):
        #     print(f"  Pred[{pi}]: {group}")
        
        # build predicted union masks (pred is list of ann ids)
        pred_union = []
        for pred in preds:
            # gather ann dicts from gt_groups (ann_id -> ann)
            anns_lookup = {a["ann_id"]: a for grp in gt_groups.values() for a in grp}
            pred_anns = [anns_lookup[a] for a in pred if a in anns_lookup]
            pred_union.append(union_masks_of_annotations(pred_anns, h, w))

        # compute IoU matrix between pred_union and gt_union
        if len(pred_union) == 0 or len(gt_gids) == 0:
            continue
        iou_matrix = np.zeros((len(pred_union), len(gt_gids)), dtype=float)
        for pi, pm in enumerate(pred_union):
            for gi, gid in enumerate(gt_gids):
                iou_matrix[pi, gi] = iou_mask(pm, gt_union[gid])

        # Greedy matching: find best IoU pair iteratively
        # This helps avoiding the IoU incorrectly computed
        matched_pred = set()
        matched_gt = set()
        while True:
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape) # the max IoU indicates the mask correctly compared
            best_iou = iou_matrix[idx] 
            if best_iou < iou_thresh:
                break
            pi, gi = idx
            matched += 1
            per_pair.append((image_id, pi, gt_gids[gi], best_iou))
            matched_pred.add(pi)
            matched_gt.add(gi)
            # zero out row and column
            iou_matrix[pi, :] = -1
            iou_matrix[:, gi] = -1

    # precision = matched / total_pred if total_pred > 0 else 0.0
    # recall = matched / total_gt if total_gt > 0 else 0.0
    # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    pred_best_iou = defaultdict(float)
    gt_best_iou = defaultdict(float)

    for _, pi, gi, iou in per_pair:
        pred_best_iou[pi] = max(pred_best_iou[pi], iou)
        gt_best_iou[gi]   = max(gt_best_iou[gi], iou)

    # 对未匹配的 predicted clusters → precision_i = 0
    for pi in range(total_pred):
        if pi not in pred_best_iou:
            pred_best_iou[pi] = 0.0

    # 对未匹配的 GT clusters → recall_j = 0
    for gi in gt_gids:
        if gi not in gt_best_iou:
            gt_best_iou[gi] = 0.0

    precision = np.mean(list(pred_best_iou.values()))
    recall    = np.mean(list(gt_best_iou.values()))
    f1        = 2 * precision * recall / (precision + recall + 1e-8)


    return {
        "total_gt_clusters": total_gt,
        "total_pred_clusters": total_pred,
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_pairs": per_pair
    }

# -----------------------
# (B) Evaluate segmentation + matching
# -----------------------
def match_detections_to_gt(detected_masks: List[np.ndarray], gt_anns: List[dict], iou_thresh=0.5):
    """
    Match prediction masks (list of binary masks for one image) to GT ann dicts.
    Returns:
      matches: list of tuples (pred_idx, gt_ann_id, iou)
      unmatched_preds: list of pred_idx
      unmatched_gt: list of gt_ann_id
    Greedy matching by highest IoU.
    """
    if not detected_masks or not gt_anns:
        return [], list(range(len(detected_masks))), [g["ann_id"] for g in gt_anns]

    n_pred = len(detected_masks)
    n_gt = len(gt_anns)
    iou_mat = np.zeros((n_pred, n_gt), dtype=float)
    for pi, pm in enumerate(detected_masks):
        for gi, g in enumerate(gt_anns):
            iou_mat[pi, gi] = iou_mask(pm, g["mask"])

    matches = []
    matched_p = set()
    matched_g = set()
    while True:
        idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        best = iou_mat[idx]
        if best < iou_thresh:
            break
        pi, gi = idx
        matches.append((pi, gt_anns[gi]["ann_id"], best))
        matched_p.add(pi)
        matched_g.add(gi)
        iou_mat[pi, :] = -1
        iou_mat[:, gi] = -1

    unmatched_preds = [i for i in range(n_pred) if i not in matched_p]
    unmatched_gt = [gt_anns[i]["ann_id"] for i in range(n_gt) if i not in matched_g]
    return matches, unmatched_preds, unmatched_gt

def evaluate_segmentation_and_matching(images_meta: Dict[int, dict],
                                       gt_by_image: Dict[int, List[dict]],
                                       predicted_detections_by_image: Dict[int, List[np.ndarray]],
                                       predicted_groups_by_image: Dict[int, List[List[int]]],
                                       iou_detect_thresh=0.5,
                                       iou_cluster_thresh=0.5):
    """
    Full pipeline evaluation.
    Inputs:
      predicted_detections_by_image: image_id -> [mask0, mask1, ...]  (mask = np.ndarray uint8 0/1)
      predicted_groups_by_image: image_id -> [ [pred_idx_1, pred_idx_2, ...], ... ]
         Here predicted_groups refer to predicted detection indices (not ann ids); after detection->gt matching we'll translate to GT ann ids.
    Returns:
      dict with segmentation metrics (TP/FP/FN, precision/recall/F1, mean IoU) and cluster-level metrics (precision/recall/F1).
    """
    # First, per-image detection matching
    total_TP = 0
    total_FP = 0
    total_FN = 0
    iou_list = []

    # Will collect predicted groups in terms of GT ann_ids for cluster evaluation:
    pred_groups_as_gt_ann = defaultdict(list)  # image_id -> list of lists of gt_ann_id

    for image_id, gt_anns in gt_by_image.items():
        dets = predicted_detections_by_image.get(image_id, [])
        matches, unmatched_preds, unmatched_gt = match_detections_to_gt(dets, gt_anns, iou_thresh=iou_detect_thresh)
        total_TP += len(matches)
        total_FP += len(unmatched_preds)
        total_FN += len(unmatched_gt)
        iou_list.extend([m[2] for m in matches])

        # Map pred_idx -> matched gt ann id (or None)
        pred_to_gt = {pi: None for pi in range(len(dets))}
        for pi, gt_ann_id, iouv in matches:
            pred_to_gt[pi] = gt_ann_id

        # Convert predicted groups of detection indices to groups of GT ann_ids (only if matched)
        groups = predicted_groups_by_image.get(image_id, [])
        groups_as_gt = []
        for grp in groups:
            mapped = [pred_to_gt[p] for p in grp if p in pred_to_gt and pred_to_gt[p] is not None]
            # dedupe
            mapped = list(dict.fromkeys(mapped))
            if mapped:
                groups_as_gt.append(mapped)
        pred_groups_as_gt_ann[image_id] = groups_as_gt

    # compute segmentation-level metrics
    det_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    det_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall) if (det_precision + det_recall) > 0 else 0.0
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0

    # Now evaluate clustering exactly as in evaluate_matching_only but using pred_groups_as_gt_ann
    # First build GT clusters structure (group_id -> list of ann dicts) per image in required format
    gt_clusters_by_image = {}
    for image_id, anns in gt_by_image.items():
        groups = defaultdict(list)
        for a in anns:
            groups[a["group_id"]].append(a)
        gt_clusters_by_image[image_id] = groups

    cluster_eval = evaluate_matching_only(gt_clusters_by_image, pred_groups_as_gt_ann, images_meta,
                                          iou_thresh=iou_cluster_thresh)

    return {
        "segmentation": {
            "TP": total_TP, "FP": total_FP, "FN": total_FN,
            "precision": det_precision, "recall": det_recall, "f1": det_f1,
            "mean_iou": mean_iou
        },
        "clustering": cluster_eval
    }

def visualize_gt_clusters(img, gt_clusters_struct_single, image_id, ann_lookup):
    """
    img: 原图 (H,W,3)
    gt_clusters_struct_single: { image_id : {gid: [ann_dicts] } }
    """
    gt_groups = gt_clusters_struct_single[image_id]

    H, W = img.shape[:2]
    vis = img.copy()

    # 为每个 cluster 生成随机颜色
    random.seed(42)
    colors = {}

    for gid in gt_groups.keys():
        colors[gid] = (
            random.randint(50,255),
            random.randint(50,255),
            random.randint(50,255)
        )

    for gid, anns in gt_groups.items():
        # 合并 mask
        color = colors[gid]

        union_mask = np.zeros((H, W), dtype=np.uint8)
        ann_ids = []

        for ann in anns:
            ann_id = ann["ann_id"]        # stored in your gt struct
            ann_full = ann_lookup[ann_id] # get COCO annotation 

            seg = ann_full["segmentation"]
            mask = rasterize_polygon(seg, H, W)

            union_mask |= mask
            ann_ids.append(ann_id)

            vis[mask > 0] = vis[mask > 0] * 0.3 + np.array(color) * 0.7

            # 找 mask 中心
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())

            # 🟢 在每个 stem mask 上写 ann_id
            text = f"{ann_id}"
            cv2.putText(vis, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,0,0), 3)
            cv2.putText(vis, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255,255,255), 2)

        # # 颜色遮罩
        # color = colors[gid]
        # vis[union_mask > 0] = vis[union_mask > 0] * 0.3 + np.array(color) * 0.7

        # 找 mask 中心点用于标注
        ys, xs = np.where(union_mask > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            continue

        # 标注 cluster id 和 ann_ids
        text = f"GT[{gid}]"
        cv2.putText(vis, text, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0,0,0), 3)
        cv2.putText(vis, text, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (255,255,0), 2)

    return vis

def visualize_gt_and_predicted_clusters_separate(img, gt_clusters_struct_single, image_id, ann_lookup,
                                                 masks, ann_ids, stem_groups):
    """
    返回两张图：
    - vis_gt: 只画 GT clusters
    - vis_pred: 只画 predicted stem groups
    """
    import numpy as np
    import cv2
    import random

    H, W = img.shape[:2]
    vis_gt = img.copy()
    vis_pred = img.copy()

    # -------------------------------
    # 1. 绘制 GT clusters
    # -------------------------------
    gt_groups = gt_clusters_struct_single.get(image_id, {})
    random.seed(42)
    gt_colors = {gid: (random.randint(50,255), random.randint(50,255), random.randint(50,255)) 
                 for gid in gt_groups.keys()}

    for gid, anns in gt_groups.items():
        union_mask = np.zeros((H, W), dtype=np.uint8)
        for ann in anns:
            ann_id = ann["ann_id"]
            ann_full = ann_lookup[ann_id]
            seg = ann_full["segmentation"]
            mask = rasterize_polygon(seg, H, W)

            union_mask |= mask
            color = gt_colors[gid]
            vis_gt[mask > 0] = vis_gt[mask > 0] * 0.3 + np.array(color) * 0.7

            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                text = f"{ann_id}"
                cv2.putText(vis_gt, text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                cv2.putText(vis_gt, text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
        ys, xs = np.where(union_mask > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            text = f"GT[{gid}]"
            cv2.putText(vis_gt, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            cv2.putText(vis_gt, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 1)

    # -------------------------------
    # 2. 绘制 predicted stem groups
    # -------------------------------
    random.seed(123)
    pred_colors = [ (random.randint(50,255), random.randint(50,255), random.randint(50,255))
                    for _ in stem_groups]

    for g_idx, group in enumerate(stem_groups):
        union_mask = np.zeros((H, W), dtype=np.uint8)
        for mask_idx in group:
            mask = masks[mask_idx]
            union_mask |= mask
            color = pred_colors[g_idx]
            vis_pred[mask > 0] = vis_pred[mask > 0] * 0.3 + np.array(color) * 0.7

            # mask 中心写 ann_id
            ann_id = ann_ids[mask_idx]
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                text = f"{ann_id}"
                cv2.putText(vis_pred, text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)
                cv2.putText(vis_pred, text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        # group 中心写 group index
        ys, xs = np.where(union_mask > 0)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            text = f"PRED[{g_idx}]"
            cv2.putText(vis_pred, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            cv2.putText(vis_pred, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 1)

    return vis_gt, vis_pred




# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # example: load GT from your uploaded file
    JSON = "./Stem_Segmentation/test/3/_annotations_grouped.coco.json"   # path to your file
    images_meta, gt_by_image = load_coco_grouped_masks(JSON)

    # Build GT clusters (for use in matching-only evaluation)
    gt_clusters_struct = build_gt_clusters(gt_by_image)
    # --------------------------
    # (A) Matching-only evaluation
    # Suppose you already have predicted_groups_by_image (key: image_id, value: list of predicted groups,
    # where each predicted group is a list of GT ann ids that were grouped together by your post-processing)
    # Example format:
    # predicted_groups_by_image = {
    #     0: [[0,26,7],[1,3]],
    #     1: [[...], ...]
    # }
    # --------------------------
    data = json.load(open(JSON, 'r'))
    
    total_gt_all = 0
    total_pred_all = 0
    matched_all = 0

    # ------- For macro mean stats -------
    precision_list = []
    recall_list = []
    f1_list = []

    test_dir = './Stem_Segmentation/test/3'
    image_files = sorted([f for f in os.listdir(test_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    image_lookup = { img["file_name"]: img for img in data["images"] }
    
    for idx, filename in enumerate(image_files):
        if filename not in image_lookup:
            print(f"⚠ Warning: {filename} not found in JSON, skip.")
            continue
        # ======== ① 取一张 image 的信息 ========
        # filename = "11_jpg.rf.6745cce81a51fe53785e70a141b005ae.jpg" # this is for debug
        image_meta = image_lookup[filename]
        image_id = image_meta["id"]

        print(f"\n Testing {idx}th, image_id={image_id}, file={filename}")

        # ======== ② 加载图片 ========
        # test_dir = "./Stem_Segmentation/test/3"
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path)
        H, W = img.shape[:2]

        # ======== ③ 获取该图的所有 annotation ========
        anns = [ann for ann in data["annotations"] if ann["image_id"] == image_id]

        masks = []
        labels = []
        bboxes = []
        scores = []
        ann_ids = []
        filtered_anns = []

        for ann in anns:
            label = ann["category_id"]
            if label not in {0, 1}:
                continue  # 只处理 stem 类

            seg = ann["segmentation"]
            mask = rasterize_polygon(seg, H, W)
            x, y, bw, bh = np.array(ann["bbox"]).astype(int)
            if bw < 5 or bh < 18:
                continue
            if np.sum(mask) < 90:
                continue
            masks.append(mask)
            labels.append(label)
            bboxes.append(ann["bbox"])
            scores.append(1.0)
            ann_ids.append(ann["id"])
            filtered_anns.append(ann)


        if len(masks) == 0:
            print("⚠ No valid stem annotations found for this image")
            exit()

        # Convert to np arrays
        masks = np.stack(masks, axis=0).astype(np.uint8)
        labels = np.array(labels)
        bboxes = np.array(bboxes)
        scores = np.array(scores)

        # ======== ④ 跑 stem pairing ========
        stem_groups, _ = stem_pairing(masks, labels, bboxes, scores, img, ) 

        # print("len(ann_ids):", len(ann_ids))
        # print(f"stem group result = {stem_groups}")

        predicted_groups_by_image = {
            image_id: [
                [ann_ids[i] for i in group]  # group 里的 idx 直接对应原始 masks
                for group in stem_groups
            ]
        }

        print("\nPredicted stem groups:")
        print(predicted_groups_by_image)


        # ======== ⑤ 只对这一张图做 matching-only evaluation ========
        gt_clusters_struct_single = {image_id: gt_clusters_struct[image_id] }
        images_meta_single = {image_id: images_meta[image_id] }
        ann_lookup = {ann["id"]: ann for ann in anns}

        results_a = evaluate_matching_only(gt_clusters_struct_single,
                                        predicted_groups_by_image,
                                        images_meta_single,
                                        filtered_anns)

        print("\n=== Result ===")
        print(results_a)

        #### Calculation for overall performance ####
        if results_a["f1"] < 0.45:
            print("this is a wrong annotation, skip")
            continue 

        total_gt_all += results_a["total_gt_clusters"]
        total_pred_all += results_a["total_pred_clusters"]
        matched_all += results_a["matched"]

        # 每张图的 precision / recall / f1
        precision_list.append(results_a["precision"])
        recall_list.append(results_a["recall"])
        f1_list.append(results_a["f1"])

        vis_gt, vis_pred = visualize_gt_and_predicted_clusters_separate(
        img, gt_clusters_struct_single, image_id, ann_lookup,
        masks, ann_ids, stem_groups
        )

        # vis_gt_s = cv2.resize(vis_gt, None, fx=0.7, fy=0.7)
        # vis_pred_s = cv2.resize(vis_pred, None, fx=0.7, fy=0.7)
        # cv2.imshow("GT Clusters", vis_gt_s)
        # cv2.imshow("Predicted Stem Groups", vis_pred_s)
        # cv2.waitKey(0)
        # break


    precision_matched = matched_all / total_pred_all if total_pred_all > 0 else 0.0
    recall_matched = matched_all / total_gt_all if total_gt_all > 0 else 0.0
    f1_matched = (
        2 * precision_matched * recall_matched
        / (precision_matched + recall_matched)
        if (precision_matched + recall_matched) > 0
        else 0.0
    )

    # ------- Compute macro-mean metrics -------
    precision_mean = mean(precision_list) if len(precision_list) else 0.0
    recall_mean = mean(recall_list) if len(recall_list) else 0.0
    f1_mean = mean(f1_list) if len(f1_list) else 0.0

    overall_stats = {
        "precision_matched": precision_matched,
        "recall_matched": recall_matched,
        "f1_matched": f1_matched,
        "precision_mean": precision_mean,
        "recall_mean": recall_mean,
        "f1_mean": f1_mean,
        "total_gt_clusters": total_gt_all,
        "total_pred_clusters": total_pred_all,
        "matched": matched_all,
    }

    print(f"\n overall_stats = {overall_stats}")
        

        











        # --------------------------
        # (B) Segmentation + matching evaluation
        # Suppose you have detector outputs per image: predicted_detections_by_image: image_id -> [mask0, mask1, ...]
        # and predicted_groups_by_image detects groups of detection indices (not ann ids).
        # --------------------------
        # results_b = evaluate_segmentation_and_matching(images_meta, gt_by_image,
        #                                                predicted_detections_by_image,
        #                                                predicted_groups_by_image_by_detection_idx)
        # print(results_b)
