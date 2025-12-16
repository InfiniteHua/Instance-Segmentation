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

from top_selection import read_coco, process_stems_skeleton, count_crossings_masks, draw_points_on_image, draw_segment_groups, count_crossings_times, build_pred_segments

from eval_clusters import rasterize_polygon, load_coco_grouped_masks

from pick_point import get_pick_point, draw_pp

import mmcv
import mmengine
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from mmengine.visualization import Visualizer

from mmdet.utils import register_all_modules

from mmengine.config import Config 

import torch, gc

def compute_iou(mask1, mask2):
    """Compute IoU between two boolean masks."""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def match_predictions_to_annotations(pred_masks, gt_masks, gt_ann_ids, iou_threshold=0.1):
    """
    Match each predicted mask to the closest COCO annotation via IoU.
    
    Args:
        pred_masks: (N_pred, H, W) binary predicted masks
        gt_masks:   (N_gt, H, W) binary annotation masks
        gt_ann_ids: list of ann_id, length N_gt
        iou_threshold: minimum IoU to accept a match
    
    Returns:
        matched_ann_ids: length N_pred list
            matched_ann_ids[i] = the annotation id matching predicted mask i
            or None if no match >= threshold
    """
    matched_ann_ids = []

    for i, pm in enumerate(pred_masks):
        best_iou = 0.0
        best_ann_id = None

        for gm, ann_id in zip(gt_masks, gt_ann_ids):
            iou = compute_iou(pm, gm)
            if iou > best_iou:
                best_iou = iou
                best_ann_id = ann_id

        if best_iou >= iou_threshold:
            matched_ann_ids.append(best_ann_id)
        else:
            matched_ann_ids.append(None)  # false positive mask

    return matched_ann_ids

def convert_predicted_groups(predicted_groups_by_image, image_id):
    groups_list = predicted_groups_by_image[image_id]  # [[anns], [anns], ...]

    pred_groups = {i: g for i, g in enumerate(groups_list)}
    return pred_groups


if __name__ == "__main__":
    # load pretrained mask-rcnn model
    register_all_modules()
    cfg = Config.fromfile('../stem_configs/mask-rcnn_r50-caffe_fpn_ms-poly-3x_Stem_tinytest.py')
    checkpoint_file = 'tutorial_exps/epoch_150.pth'
    model = init_detector(cfg, checkpoint_file, device='cuda:0')
    model.dataset_meta = cfg.metainfo
    visualizer_now = VISUALIZERS.build(model.cfg.visualizer)
    visualizer_now.dataset_meta = model.dataset_meta

    # example: load GT from your uploaded file
    JSON = "./Stem_Segmentation/test/4/_annotations_grouped.coco.json"   # path to your file
    images_meta, gt_by_image = load_coco_grouped_masks(JSON)

    # Build GT clusters (for use in matching-only evaluation)
    # gt_clusters_struct = build_gt_clusters(gt_by_image)
    data = json.load(open(JSON, 'r'))

    test_dir = './Stem_Segmentation/test/temp'
    image_files = sorted([f for f in os.listdir(test_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    image_lookup = { img["file_name"]: img for img in data["images"] }

    acc = 0
    count = 0
    for idx, filename in enumerate(image_files):
        print(f"filename: {filename}")
        if filename not in image_lookup:
            print(f"⚠ Warning: {filename} not found in JSON, skip.")
            continue

        # read image data
        # filename = "11_jpg.rf.e0f41ea72c62db6f5b30d5ad1c07bfc6.jpg"
        image_meta = image_lookup[filename]
        image_id = image_meta["id"]

        gt_segments, gt_groups, img = read_coco(r"./Stem_Segmentation/test/4", r"_annotations_pickable.coco.json", image_id)
        if gt_segments == None:
            print(f"id: {idx} incorrect - image_{filename} has no segments")
            continue
        
        # load the image
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path)
        H, W = img.shape[:2]

        img_bgr = mmcv.imread(img_path, channel_order='bgr')

        # model inference
        result = inference_detector(model, img_bgr)

        # obtain all the annotations 
        anns = [ann for ann in data["annotations"] if ann["image_id"] == image_id]

        gt_masks = []
        gt_labels = []
        gt_bboxes = []
        gt_scores = []
        gt_ann_ids = []

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
            gt_masks.append(mask)
            gt_labels.append(label)
            gt_bboxes.append(ann["bbox"])
            gt_scores.append(1.0)
            gt_ann_ids.append(ann["id"])

        if len(gt_masks) == 0:
            print("⚠ No valid stem annotations found for this image")
            exit()
        

        # Convert to np arrays
        gt_masks = np.stack(gt_masks, axis=0).astype(np.uint8)
        gt_labels = np.array(gt_labels)
        gt_bboxes = np.array(gt_bboxes)
        gt_scores = np.array(gt_scores)

        # get predicted results
        pred_masks = result.pred_instances.masks.cpu().numpy()
        # print(f" pred mask len = {len(pred_masks)}")
        pred_labels = result.pred_instances.labels.cpu().numpy()
        pred_bboxes = result.pred_instances.bboxes.cpu().numpy()
        pred_scores = result.pred_instances.scores.cpu().numpy()

        # ann_ids = match_predictions_to_annotations(pred_masks, gt_masks, gt_ann_ids)

        # run stem pairing
        _ = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        stem_groups, pred_masks_final = stem_pairing(pred_masks, pred_labels, pred_bboxes, pred_scores, img) 

        sys.stdout.close()
        sys.stdout = _
        
        # print(f"stem_groups = {stem_groups}")
        # print(f" pred_masks_final has {len(pred_masks_final)} masks")

        pred_segments = build_pred_segments(pred_masks_final)
        pred_groups = [[] for i in stem_groups] 

        for seg in pred_segments:   
            for gidx, g in enumerate(stem_groups):
                if seg["index"] in g:
                    pred_groups[gidx].append(seg)
                    break
        # print(f" pred_groups has {len(pred_groups)} groups")
        #### project predictions to annotations ####
        ann_ids = match_predictions_to_annotations(pred_masks_final, gt_masks, gt_ann_ids)
        predicted_groups_by_image = {
            image_id: [
                [ann_ids[i] for i in group]  # group 里的 idx 直接对应原始 masks
                for group in stem_groups
            ]
        }

        pred_groups_anns = convert_predicted_groups(predicted_groups_by_image, image_id)

        gt_groups = [[] for i in pred_groups_anns]   # 和 gt 逻辑一致

        for seg in gt_segments:                 # seg["index"] 是 annotation id
            for gidx, g in enumerate(pred_groups_anns.values()):
                if seg["index"] in g:
                    gt_groups[gidx].append(seg)
                    break

        ##### use skeleton to extract trend line
        dense_stems = process_stems_skeleton(pred_groups, 5)
        # for i, ds in enumerate(dense_stems):
        #     if not isinstance(ds, np.ndarray):
        #         print("❌ dense_stems[{}] is not ndarray".format(i), ds)
        #     elif ds.ndim != 2:
        #         print("❌ dense_stems[{}] ndim !=2, shape = {}".format(i, ds.shape))
        #     elif ds.shape[1] != 2:
        #         print("❌ dense_stems[{}] shape != (*,2), shape = {}".format(i, ds.shape))

        losses = np.array(count_crossings_masks(dense_stems, pred_groups))
        # crosses = np.array(count_crossings_times(dense_stems))
        min_loss = np.min(losses)
        candidates = np.where(losses == min_loss)[0]
        if len(candidates) == 1:
            top_idx = candidates[0]
        else:
            # cross_filtered = [i for i in candidates if crosses[i] <= 1] # look for the candidates with the least crossing(0 or 1)
            # if len(cross_filtered) == 1:
            #     top_idx = cross_filtered[0]
            # elif len(cross_filtered) > 1:
            #     # print(f"There are {len(cross_filtered)} non-crossing candidates: {candidates}")
            #     candidate_group = [groups[i] for i in cross_filtered]
            #     candidate_areas = [np.sum([s['area'] for s in g]) for g in candidate_group]
            #     top_idx = cross_filtered[np.argmax(candidate_areas)]
            # else:
            # Compare their areas and pick the largest one
            candidate_group = [pred_groups[i] for i in candidates]
            candidate_areas = [np.sum(np.array([s['area'] for s in g])) for g in candidate_group]
            top_idx = candidates[np.argmax(candidate_areas)]

            print(f"There are {len(candidates)} candidates: {candidates}")

        pick_point = get_pick_point(pred_groups[top_idx], dense_stems[top_idx]) 
        # img = draw_points_on_image(img, dense_stems[top_idx])
        img = draw_pp(img, pick_point)
        # for can in candidates:
        #     img = draw_points_on_image(img, dense_stems[can])

        # draw_segment_groups(img, pred_groups, None, save=True, save_dir=r"./Stem_Segmentation/test/4_result/pipeline/pick_point", save_idx=idx)
        draw_segment_groups(img, pred_groups, None, save=True, save_dir=r"./Stem_Segmentation/test/temp/masks", save_idx=idx)
        group = gt_groups[top_idx]
        pickable_flags = [s['pickable'] for s in group]
        num_pickable = sum(pickable_flags)
        total = len(group)
        if total == 0:
            print("total stems = 0, wrong annotation")
            continue
        ratio = num_pickable / total
        if ratio >= 0.5:
            acc += 1
            print(f"id: {idx} correct — pickable ratio: {ratio:.2f} ({num_pickable}/{total}) \n")
        else:
            print(f"id: {idx} wrong — pickable ratio: {ratio:.2f} ({num_pickable}/{total}) \n")

        count +=1

        del result
        torch.cuda.empty_cache()
        gc.collect()
        # break

    print(f"overall accuracy: {acc}/{count} = {acc/count}")