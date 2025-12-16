import os
import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
import math

SIZE_THRESH = 150

def read_coco(coco_base, coco_filename, img_id = 0):
    coco_dir = os.path.join(coco_base, coco_filename)
    # coco_dir = coco_base + coco_filename
    with open(coco_dir, 'r') as f:
        coco_data = json.load(f)

    total_len = len(coco_data['images'])
    if img_id>=total_len:
        return None, None, None
    
    segment_info = []
    group_info = {}
    # Image size is needed for RLE masks
    image_id_to_size = {img['id']: (img['height'], img['width']) for img in coco_data['images']}
    image_path = [img['file_name'] for img in coco_data['images']]
    valid_annotations = [i for i in coco_data['annotations'] if i["image_id"] == img_id and i["category_id"] == 1]

    img_p = image_path[img_id]
    for idx, ann in enumerate(valid_annotations):
        segmentation = ann['segmentation']
        image_id = ann['image_id']
        height, width = image_id_to_size[image_id]
        
        # Convert segmentation to binary mask
        if isinstance(segmentation, list):
            # Polygon format
            mask = np.zeros((height, width), dtype=np.uint8)
            for seg in segmentation:
                pts = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        else:
            # RLE format
            rle = segmentation
            if isinstance(rle['counts'], list):
                rle = mask_utils.frPyObjects(rle, height, width)
            mask = mask_utils.decode(rle)

        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)

        # Compute area
        area = cv2.contourArea(cnt)
        if area<SIZE_THRESH:
            continue



        # Compute direction using PCA
        data_pts = cnt.reshape(-1, 2).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(data_pts, mean=np.array([]))
        direction = eigenvectors[0]  # Principal direction (unit vector)

        # Project all points onto the principal axis to find start and end
        # Project each point: dot product with principal direction
        projections = np.dot(data_pts - mean[0], direction)
        
        # Compute center
        threshold = 0.1 * (projections.max() - projections.min())
        near_zero_indices = np.where(np.abs(projections) < threshold)[0]

        if len(near_zero_indices) > 0:
            center = data_pts[near_zero_indices].mean(axis=0)
        else:
            # fallback if no near-zero points (e.g., very small contour)
            center = mean[0]

        center = tuple(center.astype(int))

        sorted_indices = np.argsort(projections)

        k = 5  # number of points to average
        k = min(k, len(data_pts)//2)  # safety

        # if ann['id'] == 4740:
        #     print("here!")
        # Average the top-k and bottom-k points in projection order
        start_idx = sorted_indices[:k]
        end_idx   = sorted_indices[-k:]

        start = data_pts[start_idx].mean(axis=0)
        end   = data_pts[end_idx].mean(axis=0)

        if end[-1]< start[-1]:
            temp = end
            end = start
            start = temp
            direction = -direction

        vec_up = center-start
        direction_up = vec_up/np.linalg.norm(vec_up)
        vec_down = end-center
        direction_down = vec_down/np.linalg.norm(vec_down)



        # Store contour as list of (x, y)
        contour = cnt.squeeze().tolist()
        if isinstance(contour[0], int):
            contour = [contour]  # Single point edge case


        if 'group_id' in ann.keys():
            gt_group = ann['group_id']
            if gt_group not in group_info.keys():
                group_info[gt_group] = [ann['id']]
            else:
                group_info[gt_group].append(ann['id'])
        else:
            continue
            
        

        segment_info.append({
            'index': ann['id'],
            'area': float(area),
            'center': center,
            'direction': direction,
            'direction_up': direction_up,
            'direction_down': direction_down,
            'start': start,
            'end': end,
            'contour': cnt,
            'pickable': ann['pickable'],
        })

    if len(segment_info)==0:
        return None, None, None

    # Example output
    # print(segment_info[0] if segment_info else "No segments found.")
    return segment_info, group_info, cv2.imread(os.path.join(coco_base, img_p))

def build_pred_segments(pred_masks):
    segments = []

    for i in range(len(pred_masks)):
        mask = pred_masks[i].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        # PCA direction
        data_pts = cnt.reshape(-1, 2).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(data_pts, mean=np.array([]))
        direction = eigenvectors[0]

        # project
        projections = np.dot(data_pts - mean[0], direction)
        sorted_indices = np.argsort(projections)

        # Compute center
        threshold = 0.1 * (projections.max() - projections.min())
        near_zero_indices = np.where(np.abs(projections) < threshold)[0]

        if len(near_zero_indices) > 0:
            center = data_pts[near_zero_indices].mean(axis=0)
        else:
            # fallback if no near-zero points (e.g., very small contour)
            center = mean[0]

        center = tuple(center.astype(int))

        k = 5
        k = min(k, len(data_pts)//2)

        start = data_pts[sorted_indices[:k]].mean(axis=0)
        end   = data_pts[sorted_indices[-k:]].mean(axis=0)

        if end[-1]< start[-1]:
            temp = end
            end = start
            start = temp
            direction = -direction

        vec_up = center-start
        direction_up = vec_up/np.linalg.norm(vec_up)
        vec_down = end-center
        direction_down = vec_down/np.linalg.norm(vec_down)

        

        contour = cnt.squeeze().tolist()
        if isinstance(contour[0], int):
            contour = [contour]

        segments.append({
            'index': i,                # predicted index
            'area': float(area),
            'center': tuple(center),
            'direction': direction,
            'direction_up': direction_up,
            'direction_down': direction_down,
            'start': start,
            'end': end,
            'contour': cnt,
        })

    return segments


def grouping_accuracy(gt, pred):
    # Build a mapping from object -> group id for ground truth
    gt_map = {}
    for g_id, objs in gt.items():
        for o in objs:
            gt_map[o] = g_id

    # Build a mapping from object -> group id for prediction
    pred_map = {}
    
    htp = 0
    tp = fp = fn = 0
    for g_id, objs in enumerate(pred):
        group_idx = []
        for o in objs:
            group_idx.append(o['index'])
            pred_map[o['index']] = g_id

        gt_group = [] # ground truth of predicted segments
        for i in group_idx:
            gt_group.append(gt_map[i])

        unique_values = np.unique(gt_group)
        if len(unique_values)==1 or (len(unique_values)==2 and unique_values.min()==-1):
            htp +=1 
            if len(gt[gt_group[0]]) == len(gt_group) or len(gt[gt_group[0]]) == len(gt_group)+1: #LHS: ground truth segments
                tp += 1

    accuracy = tp/(g_id+1)
    half_accuracy = htp/(g_id+1) # htp means all segments in pred group do belongs to one gt group. But there are some missing segments.

    return {
        # 'precision': precision,
        # 'recall': recall,
        # 'f1': f1,
        "accuracy" : accuracy
    }

def draw_segment_groups(image, groups, lines = None, draw_directions=False, scale=50, save=False, save_dir=None, save_idx=0):
    """
    Draws grouped segments with unique colors for each group.

    Parameters:
    - image: color image to draw on
    - groups: list of grouped segments (output of match_segments)
    - draw_directions: if True, draw direction arrows
    - scale: length of direction arrows
    """
    # print("Num of groups:", len(groups))
    vis_img = image.copy()
    height, width = vis_img.shape[:2]

    # Generate random colors for groups
    rng = np.random.default_rng(114514)
    if lines==None:
        colors = rng.integers(0, 255, size=(len(groups), 3), dtype=np.uint8)
    else:
        colors = rng.integers(0, 255, size=(max(len(groups), len(lines)), 3), dtype=np.uint8)
    # colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0, 255, 255), (255,0,255), (128,128,128), (0,128,255),(0,255,128),(128,0,255),(255,0,128),(128,255,0),(255,128,0)]

    for group_idx, group in enumerate(groups):
        color = tuple(int(c) for c in colors[group_idx])
        # vis_img = image.copy()
        for seg in group:
            # Draw contour
            cv2.drawContours(vis_img, [seg['contour']], -1, color, 2)

            # Draw center
            center = tuple(map(int, seg['center']))
            # cv2.circle(vis_img, center, 3, (0, 0, 255), -1)
            # cv2.circle(vis_img, tuple(map(int, seg['start'])), 4, (0, 0, 255), -1)
            # cv2.circle(vis_img, tuple(map(int, seg['end'])), 4, (0, 0, 255), -1)

            # Draw direction
            if draw_directions:
                direction = seg['direction']
                end_point = (
                    int(center[0] + direction[0] * scale),
                    int(center[1] + direction[1] * scale)
                )
                cv2.arrowedLine(vis_img, center, end_point, (0, 255, 0), 2, tipLength=0.2)

    if lines is not None:
        for line_idx, _ in enumerate(lines):
            color = tuple(int(c) for c in colors[line_idx])
            m, b = lines[line_idx]
            h, w = image.shape[:2]
            x0 = 0
            x1 = w - 1
            y0 = int(m * x0 + b)
            y1 = int(m * x1 + b)
            
            pt1 = (x0, y0)
            pt2 = (x1, y1)

            cv2.line(vis_img, pt1, pt2, color, 2)

            
        # cv2.imshow('Segment Groups', vis_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # return vis_img

    if save:
        if save_dir == None:
            cv2.imwrite('result.png', vis_img)
        else:
            cv2.imwrite(os.path.join(save_dir, f'result_{save_idx}.png'), vis_img)
    else:
        cv2.imshow('Segment Groups', vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def _angle_between(v1, v2):
    """Compute angle between two vectors in degrees."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    degrees = math.degrees(math.acos(dot_product))
    if degrees > 90:
        degrees = 180-degrees
    return degrees

def _compute_trendline(group):
    """
    Compute a linear trendline (y = mx + b) for a group of segments.
    Uses all contour points from all segments for a robust fit.

    Args:
        group (List[Dict]): A list of segment dictionaries.

    Returns:
        Tuple[float, float] or None: The slope (m) and intercept (b) of the line,
                                     or None if a line cannot be fit.
    """
    points = []
    for seg in group:
        # Assumes seg['contours'] is a list/array of [x, y] points
        points.extend(np.array(seg['contour']))

    if len(points) < 2:
        return None  # Not enough points to fit a line

    points = np.array(points)
    xs, ys = points[:,0, 0], points[:,0, 1]

    # Use a robust fitting method in case of vertical lines
    if np.all(xs == xs[0]):  # Perfectly vertical line
        return float('inf'), xs[0] # Represent as (infinite slope, x-intercept)
        
    # m, b = np.polyfit(xs, ys, 1)
    X = np.stack((xs, ys), axis=1).astype(np.float32)

    mean, eigenvectors = cv2.PCACompute(X, mean=np.array([]))
    
    direction = eigenvectors[0]
    if direction[0] == 0:
        direction[0] = 1e-5
    m = direction[1]/direction[0]
    b = mean[0,1] - m * mean[0,0]
    if m == None:
        print("error in PCA")
    return m, b

# true means no overlap
def _check_overlap(group, segment):
    sorted_indices = sorted(range(len(group)), key=lambda i: group[i]['start'][1])
    group = [group[i] for i in sorted_indices]
    for s in group:
        if (s['start'][1] > segment['start'][1] and s['start'][1] < segment['end'][1]) or \
        (s['end'][1] > segment['start'][1] and s['end'][1] < segment['end'][1]) or \
        (s['start'][1] < segment['start'][1] and s['end'][1] > segment['end'][1]) :
            return False
    
    return True

def _check_group_overlap(group1, group2):
    sorted_indices = sorted(range(len(group1)), key=lambda i: group1[i]['start'][1])
    group1 = [group1[i] for i in sorted_indices]

    sorted_indices = sorted(range(len(group2)), key=lambda i: group2[i]['start'][1])
    group2 = [group2[i] for i in sorted_indices]

    for seg in group1:
        value = _check_overlap(group2, seg)
        if value == False:
            return False
        
    return True