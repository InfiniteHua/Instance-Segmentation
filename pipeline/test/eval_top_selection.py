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

from top_selection import read_coco, process_stems_skeleton, count_crossings_masks, draw_points_on_image, draw_segment_groups

from eval_clusters import rasterize_polygon, load_coco_grouped_masks


def convert_predicted_groups(predicted_groups_by_image, image_id):
    groups_list = predicted_groups_by_image[image_id]  # [[anns], [anns], ...]

    pred_groups = {i: g for i, g in enumerate(groups_list)}
    return pred_groups

if __name__ == "__main__":
    # example: load GT from your uploaded file
    JSON = "./Stem_Segmentation/test/4/_annotations_grouped.coco.json"   # path to your file
    images_meta, gt_by_image = load_coco_grouped_masks(JSON)

    # Build GT clusters (for use in matching-only evaluation)
    # gt_clusters_struct = build_gt_clusters(gt_by_image)
    data = json.load(open(JSON, 'r'))

    test_dir = './Stem_Segmentation/test/4'
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
        image_meta = image_lookup[filename]
        image_id = image_meta["id"]

        segments, gt_groups, img = read_coco(r"./Stem_Segmentation/test/4", r"_annotations_pickable.coco.json", image_id)
        if segments == None:
            print(f"id: {idx} incorrect - image_{filename} has no segments")
            continue

        # load the image
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path)
        H, W = img.shape[:2]

        # obtain all the annotations 
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

        # run stem pairing
        _ = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        stem_groups = stem_pairing(masks, labels, bboxes, scores, img) 

        sys.stdout.close()
        sys.stdout = _

        predicted_groups_by_image = {
            image_id: [
                [ann_ids[i] for i in group]  # group 里的 idx 直接对应原始 masks
                for group in stem_groups
            ]
        }

        pred_groups = convert_predicted_groups(predicted_groups_by_image, image_id)

        groups = [[] for i in pred_groups]   # 和 gt 逻辑一致

        for seg in segments:                 # seg["index"] 是 annotation id
            for gidx, g in enumerate(pred_groups.values()):
                if seg["index"] in g:
                    groups[gidx].append(seg)
                    break
        
        ##### use skeleton to extract trend line
        dense_stems = process_stems_skeleton(groups, 5)
        # for i, ds in enumerate(dense_stems):
        #     if not isinstance(ds, np.ndarray):
        #         print("❌ dense_stems[{}] is not ndarray".format(i), ds)
        #     elif ds.ndim != 2:
        #         print("❌ dense_stems[{}] ndim !=2, shape = {}".format(i, ds.shape))
        #     elif ds.shape[1] != 2:
        #         print("❌ dense_stems[{}] shape != (*,2), shape = {}".format(i, ds.shape))

        losses = np.array(count_crossings_masks(dense_stems, groups))
        min_loss = np.min(losses)
        candidates = np.where(losses == min_loss)[0]
        if len(candidates) == 1:
            top_idx = candidates[0]
        else:
            # Compare their areas and pick the largest one
            candidate_group = [groups[i] for i in candidates]
            candidate_areas = [np.sum(np.array([s['area'] for s in g])) for g in candidate_group]
            top_idx = candidates[np.argmax(candidate_areas)]
            print(f"\n There are {len(candidates)} candidates: {candidates}")
        img = draw_points_on_image(img, dense_stems[top_idx])

        ## !!! this need to be further developed into pick point selection and candidate secondary selection
        # if all([s['pickable'] for s in groups[top_idx]]):
        #     acc +=1
        #     print("id:", idx, " correct")
        # else:
        #     print("id:", idx, )
        group = groups[top_idx]
        pickable_flags = [s['pickable'] for s in group]
        num_pickable = sum(pickable_flags)
        total = len(group)
        ratio = num_pickable / total
        if ratio >= 0.5:
            acc += 1
            print(f"id: {idx} correct — pickable ratio: {ratio:.2f} ({num_pickable}/{total})")
        else:
            print(f"id: {idx} wrong — pickable ratio: {ratio:.2f} ({num_pickable}/{total})")

        draw_segment_groups(img, groups, None, save=True, save_dir=r"./Stem_Segmentation/test/4_result/p_s", save_idx=idx)
        count +=1

    print(f"overall accuracy: {acc}/{count} = {acc/count}")