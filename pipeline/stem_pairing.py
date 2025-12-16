import os
import mmcv
import mmengine
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from mmengine.visualization import Visualizer

from mmdet.utils import register_all_modules

from mmengine.config import Config 

import numpy as np
import cv2

import math
import time
import random


### !!!!!!!!!!!!! ###
# when run with anns 'preprossesing_stems' shall be adjusted comparing to running with test file

def prepossesing_stems(masks, labels, bboxes, scores):

    valid_masks = []
    valid_labels = []
    valid_bboxes = []
    valid_scores = []
    valid_areas = []
    valid_indices = []  # 保留原始索引方便打印日志

    for i, (mask, label, bbox, score) in enumerate(zip(masks, labels, bboxes, scores)):
    #for i in range(num_masks):
        STEM_CLASS_IDS = {0} # for prediction only 0 is stem
        #STEM_CLASS_IDS = {0, 1} # for annotations boyh 0 and 1 are stem
        if label not in STEM_CLASS_IDS:
            continue    
        ### cancel this when using anns
        # if model.dataset_meta['classes'][label] != 'Stem':
        #     continue

        if score < 0.4:
            continue

        # this is after model detection
        x1, y1, x2, y2 = bbox.astype(int)
        w, h = x2 - x1, y2 - y1

        # this is using anns
        # x, y, w, h = bbox.astype(int)

        if w < 5 or h < 18:
            continue

        ys, xs = np.where(mask > 0)
        if len(xs) <= 90:  
            continue

        valid_masks.append(mask)
        valid_labels.append(label)
        valid_bboxes.append(bbox)
        valid_scores.append(score)
        valid_areas.append(np.sum(mask > 0))
        valid_indices.append(i)  
    
    num_valid = len(valid_masks)
    removed = [False] * num_valid

    for i in range(num_valid):
        if removed[i]:
            continue
        for j in range(i + 1, num_valid):
            if removed[j]:
                continue

            inter = np.logical_and(valid_masks[i] > 0, valid_masks[j] > 0)
            inter_area = np.sum(inter)

            smaller_area = min(valid_areas[i], valid_areas[j])
            overlap_ratio = inter_area / smaller_area if smaller_area > 0 else 0

            if overlap_ratio > 0.2:  # 可以调整阈值
                if valid_areas[i] >= valid_areas[j]:
                    removed[j] = True
                    print(f"stem_{valid_indices[j]} is removed by stem_{valid_indices[i]} "
                        f"(area={smaller_area}, overlap={overlap_ratio:.3f})")
                else:
                    removed[i] = True
                    print(f"stem_{valid_indices[i]} is removed by stem_{valid_indices[j]} "
                        f"(area={smaller_area}, overlap={overlap_ratio:.3f})")
                    break  # 当前i已被删除，无需继续

    # 只保留未删除的mask
    masks = [valid_masks[i] for i in range(num_valid) if not removed[i]]
    labels = [valid_labels[i] for i in range(num_valid) if not removed[i]]
    bboxes = [valid_bboxes[i] for i in range(num_valid) if not removed[i]]
    scores = [valid_scores[i] for i in range(num_valid) if not removed[i]]

    return masks, labels, bboxes, scores

def candidate_visual(img_bgr, centers, directions, stem_lengths, stem_masks, length_conf, img_path, indx):
    colored = img_bgr.copy()

    base_center = centers[indx]
    base_dir = directions[indx]
    l1 = stem_lengths[indx]

    #上色 base stem (红色)
    mask_base = stem_masks[indx]
    pixel_count = np.sum(mask_base > 0)
    print(f"Stem {indx} has pixels: {pixel_count}")

    colored[mask_base > 0] = [0, 0, 255]   # BGR 红色

    start_point = (int(base_center[0]), int(base_center[1]))
    end_point = (
        int(base_center[0] + -base_dir[0] * l1[1]*-50),  # 方向可适当放大
        int(base_center[1] + -base_dir[1] * l1[1]*-50)
    )

    # 画线（红色）
    cv2.line(colored, start_point, end_point, (0, 0, 255), 2)  # BGR 红色，粗细=2

    # 可选：在终点画一个箭头
    cv2.arrowedLine(colored, start_point, end_point, (0, 0, 255), 2, tipLength=0.2)

    upper_scores = []
    lower_scores = []

    # 遍历其他 stem
    for j, (c2, d2, mask_j, l2, conf2) in enumerate(zip(centers, directions, stem_masks, stem_lengths, length_conf)):
        if j == indx:
            continue
        res = in_sector(base_center, base_dir, c2, l1, l2, d2)
        if res is not None:
            region, side = res
            if direction_condition(base_dir, d2, region, side, base_center, c2):
                score = compute_score(base_center, base_dir, c2, d2, l2, conf2)
                if region == "upper":
                    upper_scores.append((score, c2))
                else:
                    lower_scores.append((score, c2))
                colored[mask_j > 0] = [0, 255, 0]

                    # 在 stem 中心标注 score
                center_xy = tuple(c2.astype(int))  # 转整数像素坐标
                text = f"{score:.2f}"             # 保留两位小数
                cv2.putText(
                        colored, text, center_xy,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 0),  # 蓝色文字
                        thickness=1,
                        lineType=cv2.LINE_AA
                    )   

    # 显示结果
    save_path = os.path.splitext(img_path)[0] + "_colored.jpg"
    cv2.imwrite(save_path, colored)

    print(f"✅ 已保存到 {save_path}")


def PCA_feature_cluster(all_coords):
    mean, eigenvectors = cv2.PCACompute(all_coords, mean=None)

    principal_axis = eigenvectors[0]  
    projected = all_coords @ principal_axis 
    min_proj, max_proj = projected.min(), projected.max()
    mid_proj = (min_proj + max_proj) / 2 
    center = mean[0] + principal_axis * (mid_proj - projected.mean())

    direction = normalize_direction(eigenvectors[0]) # put all upward
    projections = np.dot(all_coords - center, direction)
    length_u = projections.max()
    length_l = -projections.min()
    # length = projections.max() - projections.min()
    # print(direction)
    return center, direction, length_u, length_l

def PCA_feature(coords):
    mean, eigenvectors = cv2.PCACompute(coords, mean=None)
    center = mean[0]
    direction = eigenvectors[0] # the direction is randomized by PCA, not necesssarily upward or downword
    direction = normalize_direction(eigenvectors[0]) # put all upward
    projections = np.dot(coords - center, direction)
    length_u = projections.max()
    length_l = -projections.min()
    # length = projections.max() - projections.min()
    # print(direction)
    return center, direction, length_u, length_l

  
def angle_between(v1, v2):
    """返回向量 v1 到 v2 的夹角（弧度），范围 [-pi, pi], attention that in pixel coordinate, a positive angle means on the right side"""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    ang = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
    if ang > math.pi:
        ang -= 2*math.pi
    if ang < -math.pi:
        ang += 2*math.pi
    return ang

def in_sector(base_center, base_dir, other_center, base_length, other_length, other_dir, max_angle=20*np.pi/180):
    """
    判断 other_center 是否落在 base_center 以 base_dir/反向为中轴的扇形区域。
    返回 ("upper"/"lower", "left"/"right") 或 None
    """
    # if abs(angle_between(base_dir, other_dir)) > math.pi/2: 
    #     other_dir = -other_dir # fix direction problem
    
    vec = other_center - base_center

    base_top = base_center + base_dir * (base_length[0] )
    base_bottom = base_center - base_dir * (base_length[1] )

    other_top = other_center + other_dir * (other_length[0] )
    other_bottom = other_center - other_dir * (other_length[1] )

    # if np.linalg.norm(vec) < 1e-8:
    #     return None

    # 判断上下：看 vec 在 base_dir 上的投影正负
    proj = np.dot(vec, base_dir)
    # print(f"vec = {vec}")
    if proj >= 0:
        region = "upper"
        axis = base_dir
    else:
        region = "lower"
        axis = -base_dir   # ✅ 下半部分用反方向

    # 左右：通过 cross product 判断
    perp = np.cross(base_dir, vec)
    side = "left" if perp > 0 else "right"

    # use abs ang estimate if belong to the sector
    cos_ang = np.dot(axis, vec) / (np.linalg.norm(axis) * np.linalg.norm(vec))
    cos_ang = np.clip(cos_ang, -1.0, 1.0)  
    ang = np.arccos(cos_ang)
    if ang > max_angle:
        return None
    
    if region == "upper":    #     # other 必须真的在 base 之上
        if np.dot(other_bottom + other_dir*0 - base_top, base_dir) < 0:  # add a tolerance of 10
            return None
    else:  # lower
        # other 必须真的在 base 之下
        if np.dot(other_top - other_dir*0 - base_bottom, base_dir) > 0:
            return None
        
    return region, side
    
    
def direction_condition(base_dir, other_dir, region, side, base_center, other_center, tol1=np.deg2rad(20), tol2=np.deg2rad(15), tol3=np.deg2rad(10)):
    """
    check those in-section stems' angle condition
    old: tol1(12.5), tol2(23), tol3(10)
    """
    # if abs(angle_between(base_dir, other_dir)) > math.pi/2: 
    #     other_dir = -other_dir # fix direction problem
    ang = angle_between(base_dir, other_dir)
    
    vec = other_center - base_center

    valid = False 

    if region == "upper":
        cos_ang_pos = np.dot(base_dir, vec) / (np.linalg.norm(base_dir) * np.linalg.norm(vec))
        cos_ang_pos = np.clip(cos_ang_pos, -1.0, 1.0)  
        ang_pos = np.arccos(cos_ang_pos)

        if side == "left":
            if 0 > ang:
                if ang_pos - ang < tol1:
                    valid = True
            # elif tol2 > ang >= 0:
            else:
                if ang-ang_pos < tol2:
                    valid = True

        else:  # right
            if 0 < ang:
                if ang_pos + ang < tol1:
                    valid = True
            # elif -tol2 < ang <= 0:
            else:
                if abs(ang)-ang_pos <tol2:
                    valid = True

    else:  # lower
        cos_ang_pos = np.dot(-base_dir, vec) / (np.linalg.norm(base_dir) * np.linalg.norm(vec))
        cos_ang_pos = np.clip(cos_ang_pos, -1.0, 1.0)  
        ang_pos = np.arccos(cos_ang_pos)
        if side == "left":
            if 0 < ang:
                if ang_pos + ang < tol1:
                    valid = True
            # elif -tol2 < ang <= 0:
            else:
                if abs(ang)-ang_pos <tol2:
                    valid = True
        else:  # right
            if 0 > ang:
                if ang_pos - ang < tol1:
                    valid = True
            # elif tol2 > ang >= 0:
            else:
                if ang-ang_pos < tol2:
                    valid = True
    
    # print(f"ang_pos = {ang_pos} and ang = {ang} ")
    return valid
        
def line_intersection(p1, d1, p2, d2):
    """
    求两条直线 p1 + t*d1 与 p2 + s*d2 的交点
    如果平行或接近平行，返回 None
    """
    A = np.array([d1, -d2]).T  # 2x2
    b = p2 - p1
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t, s = np.linalg.solve(A, b)
    return p1 + t * d1

def compute_score(c1, d1, c2, d2, l2, conf2):
    """
    中心 stem (c1,d1) 与另一 stem (c2,d2) 的 score
    """
    inter = line_intersection(c1, d1, c2, d2)
    if inter is None:
        return float("inf")  # 平行时认为分数很大
    radius = abs(np.linalg.norm(inter - c2) - (l2[0]+l2[1])/4)/(1 + 1.5 * (conf2 ** 2))
    # radius = abs(np.linalg.norm(inter - c2))/(1 + 0.5 * (conf2 ** 2))
    theta = abs(angle_between(d1, d2))
    return radius * theta

def normalize_direction(d):
    """统一方向：强制 y>=0"""
    d = d / (np.linalg.norm(d) + 1e-6)
    if d[1] < 0:   # 如果朝下，翻转
        d = -d
    return d

def pick_best_candidate(i, candidates, score_map, diff_thresh = 35, alpha = 1):
    """
    Choose the candidate minimizing (score_ij + score_ji)
    """
    if not candidates:
        return None

    # 按 score_ij ascending order（smaller the better）
    candidates_sorted = sorted(candidates, key=lambda x: x[0])

    sum_score = []
    prev_scores = [] 

    for score_ij, c2, j in candidates_sorted:
        # search j->i 
        score_ji = score_map.get((j, i), float('inf'))
        # sum_score.append(score_ij + score_ji - alpha* score_ij * score_ji/ (score_ij + score_ji))
        sum_score.append(score_ij + score_ji + abs(score_ij - score_ji))
        
    # min_index = sum_score.index(min(sum_score))
    sorted_idx = np.argsort(np.array(sum_score))
    for idx in sorted_idx:
        j_best = candidates_sorted[idx][2]
        score_ji_best = score_map.get((j_best, i), float('inf'))
        score_ij_best = score_map.get((i, j_best), float('inf'))

        if abs(score_ji_best - score_ij_best) < diff_thresh and score_ij_best < 55:
            if (not prev_scores) or (score_ji_best + score_ij_best < min(prev_scores)):
                # print(f"stem {j_best} is picked by stem {i} for score_{i}_{j_best} = {candidates_sorted[idx][0]} and score_{j_best}_{i} is {score_map.get((j_best, i), float('inf'))}")
                return j_best
            else:
                print(f"stem {j_best} is unmatched with stem {i} for the fallback check: score_{i}_{j_best} = {candidates_sorted[idx][0]} and score_{j_best}_{i} is {score_map.get((j_best, i), float('inf'))}")
        else:
            print(f"stem {j_best} is unmatched with stem {i} for large score difference: score_{i}_{j_best} = {candidates_sorted[idx][0]} and score_{j_best}_{i} is {score_map.get((j_best, i), float('inf'))}")

        prev_scores.append(score_ji_best + score_ij_best)
            

def dfs(node, current_group, visited_clusters, adjacency):
        visited_clusters.add(node)
        current_group.add(node)
        for nb in adjacency[node]:
            if nb not in visited_clusters:
                dfs(nb, current_group, visited_clusters, adjacency)

def stem_scores(centers, directions, stem_lengths, length_conf):
    all_upper_scores = []  # 存每个 stem 的 upper_scores
    all_lower_scores = []  # 存每个 stem 的 lower_scores
    score_map = {}  # key: (i,j) -> score from i->j
    for i, (c1, d1, l1) in enumerate(zip(centers, directions, stem_lengths)):
        upper_scores = []
        lower_scores = []

        for j, (c2, d2, l2, conf2) in enumerate(zip(centers, directions, stem_lengths, length_conf)):
            if i == j:
                continue
            if abs(angle_between(d1, d2)) > math.pi/2: 
                if i < j:
                    d1 = -d1 # fix direction problem
                else:
                    d2 = -d2

            res = in_sector(c1, d1, c2, l1, l2, d2)
            if res is not None:
                region, side = res
                # print(f"region={region} and side = {side}")
                if direction_condition(d1, d2, region, side, c1, c2):
                    score = compute_score(c1, d1, c2, d2, l2, conf2)
                    # print(f"score={score}")
                    if region == "upper":
                        upper_scores.append((score, c2, j))
                    else:
                        lower_scores.append((score, c2, j))
                    
                    score_map[(i, j)] = score

        all_upper_scores.append(upper_scores)
        all_lower_scores.append(lower_scores)

    return all_upper_scores, all_lower_scores, score_map

def consistency_identification(num_stems, graph, directions, score_map, all_upper_scores, all_lower_scores, cross_thresh):
    for i in range(num_stems):
        neighbors = graph[i]

        # only when stem i contains upper and lower 
        types = [t for (_, t) in neighbors]
        if "upper" in types and "lower" in types:
            j_up = [x for x in neighbors if x[1] == "upper"][0][0]
            j_low = [x for x in neighbors if x[1] == "lower"][0][0]

            # get j_up、j_low vectors
            d_up = directions[j_up]
            d_low = directions[j_low]
            ang_consis = abs(angle_between(d_up, d_low))
            if ang_consis > np.deg2rad(90):
                ang_consis = np.deg2rad(180) - ang_consis 

            if abs(ang_consis) > cross_thresh:
                score_up = score_map.get((i, j_up), float('inf'))
                score_low = score_map.get((i, j_low), float('inf'))

                # delete the larger score
                if score_up > score_low:
                    print(f"Removing inconsistent upper pair ({i}, {j_up}) for large angle diff {math.degrees(ang_consis):.2f}°")
                    graph[i] = [(j, t) for (j, t) in graph[i] if j != j_up]
                    graph[j_up] = [(j, t) for (j, t) in graph[j_up] if j != i]
                    all_upper_scores[i] = [s for s in all_upper_scores[i] if s[2] != j_up]
                else:
                    print(f"Removing inconsistent lower pair ({i}, {j_low}) for large angle diff {math.degrees(ang_consis):.2f}°")
                    graph[i] = [(j, t) for (j, t) in graph[i] if j != j_low]
                    graph[j_low] = [(j, t) for (j, t) in graph[j_low] if j != i]
                    all_lower_scores[i] = [s for s in all_lower_scores[i] if s[2] != j_low]

    return all_upper_scores, all_lower_scores, graph


def clustering_by_stems(num_stems, all_upper_scores, all_lower_scores, directions, score_map):

    best_upper = {}
    best_lower = {}

    for i, (up, low) in enumerate(zip(all_upper_scores, all_lower_scores)):
        # first round clustering
        if up:
            j_upper = pick_best_candidate(i, up, score_map)
            if j_upper is not None:
                best_upper[i] = j_upper
        if low:
            j_lower = pick_best_candidate(i, low, score_map)
            if j_lower is not None:
                best_lower[i] = j_lower
    

    # neighbor list
    graph = {i: [] for i in range(num_stems)}

    for i in range(num_stems):
        if i in best_upper:
            j = best_upper[i]
            if j in best_lower and best_lower[j] == i:

                ## Appended Mechanism: intermediate check ##
                found_middle = False
                for k in range(num_stems):
                    if k == i or k == j:
                        continue
                    if (
                    best_upper.get(k) == j
                    and best_lower.get(k) == i
                    and score_map.get((k, i), float('inf')) <= 30
                    and score_map.get((k, j), float('inf')) <= 30
                ):
                        found_middle = True
                        graph[i].append((k, "upper"))
                        graph[k].append((i, "lower"))
                        graph[k].append((j, "upper"))
                        graph[j].append((k, "lower"))
                        print(f"Intermediate connection formed: {i} - {k} - {j}")

                    elif(best_upper.get(k) == j): # look one more downward
                        best_lower_k = best_lower.get(k)
                        if (
                        best_lower.get(best_lower_k) == i
                        and score_map.get((best_lower_k, i), float('inf')) <= 30):
                            found_middle = True
                            graph[i].append((best_lower_k, "upper"))
                            graph[best_lower_k].append((i, "lower"))
                            graph[k].append((j, "upper"))
                            graph[j].append((k, "lower"))
                            print(f"Intermediate connection formed: {i} - {best_lower_k} - {k} - {j}")
                    
                    elif(best_lower.get(k) == i): # look one more upward 
                        best_upper_k = best_upper.get(k)
                        if (
                        best_upper.get(best_upper_k) == j
                        and score_map.get((best_upper_k, j), float('inf')) <= 30):
                            found_middle = True
                            graph[i].append((k, "upper"))
                            graph[k].append((i, "lower"))
                            graph[best_upper_k].append((j, "upper"))
                            graph[j].append((best_upper_k, "lower"))
                            print(f"Intermediate connection formed: {i} - {k} - {best_upper_k} - {j}")
                    


                if not found_middle:
                    graph[i].append((j, "upper"))
                    graph[j].append((i, "lower"))

    
    for k in graph:
        graph[k] = list(set(graph[k]))
    # print(graph)

    #### Consistency Identification ####
    cross_thresh = np.deg2rad(22.5)
    all_upper_scores, all_lower_scores, graph = consistency_identification(num_stems, graph, directions, score_map, all_upper_scores, all_lower_scores, cross_thresh)

    # print("After consistency check:")
    # print(graph)

    ############ Multi-stages Match ###########
    changed = True
    iteration = 0

    while changed:
        iteration += 1
        changed = False
        print(f"\n--- Iteration {iteration} ---")

        # Step 1: 找出缺少 upper/lower 的 stems
        missing_matches = {}
        for i, neighbors in graph.items():
            existing_dirs = {tag for _, tag in neighbors}
            missing = []
            if 'upper' not in existing_dirs:
                missing.append('upper')
            if 'lower' not in existing_dirs:
                missing.append('lower')
            if missing:
                missing_matches[i] = missing

        # Step 2: 计算 fully matched stems
        all_stems = set(range(num_stems))
        fully_matched = all_stems - set(missing_matches.keys())

        # Step 3: 构造候选配对
        second_round_matches = {}
        for i, missing in missing_matches.items():
            second_round_matches[i] = []
            for need in missing:
                if need == "upper":
                    candidates = all_upper_scores[i]
                else:
                    candidates = all_lower_scores[i]

                # 去掉已经 fully matched 的
                candidates = [c for c in candidates if c[2] not in fully_matched]
                if not candidates:
                    continue

                filtered = []
                for score, c2, j in candidates:
                    j_missing = missing_matches.get(j, [])

                    # 只允许互补方向
                    if need == "upper" and "lower" in j_missing:
                        filtered.append((score, c2, j))
                    elif need == "lower" and "upper" in j_missing:
                        filtered.append((score, c2, j))
                best = None
                if filtered:           
                    best = pick_best_candidate(i, filtered, score_map)    
                    if best is not None:            
                        second_round_matches[i].append((best, need))

        print(f"Result from iteration-{iteration}:{second_round_matches}")

        # Step 4: 更新 graph，只添加 mutual 匹配
        for i, matches in second_round_matches.items():
            for j, need in matches:
                if j < i:
                    continue
                complement = 'lower' if need == 'upper' else 'upper'
                if j in second_round_matches:
                    for jj, nneed in second_round_matches[j]:
                        if jj == i and nneed == complement:

                            ## Appended Mechanism: intermediate check ##
                            found_middle = False
                            for k, kmatches in second_round_matches.items():
                                if k == i or k == j:
                                    continue
                                if len(kmatches) == 2:
                                    pair = {kmatches[0][0], kmatches[1][0]}
                                    if {i, j} == pair:
                                        s_ki = score_map.get((k, i), float('inf'))
                                        s_kj = score_map.get((k, j), float('inf'))
                                        if s_ki <= 20 and s_kj <= 20:
                                            found_middle = True
                                            graph[i].append((k, need))
                                            graph[k].append((i, complement))
                                            graph[k].append((j, need))
                                            graph[j].append((k, complement))
                                            print(f"Intermediate connection formed: {i} - {k} - {j}")
                                            break
                                    elif i in pair:
                                        m = (pair - {i}).pop() 
                                        matches = second_round_matches.get(m, [])
                                        if (j, need) in matches:
                                            found_middle = True
                                            graph[i].append((k, need))
                                            graph[k].append((i, complement))
                                            graph[m].append((j, need))
                                            graph[j].append((m, complement))
                                            print(f"Intermediate connection formed: {i} - {k} - {m} - {j}")
                                            break
                                    elif j in pair:
                                        m = (pair - {j}).pop() 
                                        matches = second_round_matches.get(m, [])
                                        if (i, complement) in matches:
                                            found_middle = True
                                            graph[i].append((m, need))
                                            graph[m].append((i, complement))
                                            graph[k].append((j, need))
                                            graph[j].append((k, complement))
                                            print(f"Intermediate connection formed: {i} - {m} - {k} - {j}")
                                            break


                            if not found_middle:
                                graph[i].append((j, need))
                                graph[j].append((i, complement))
                                changed = True  # 本轮有新匹配产生
                                # print(f"Matched: {i} ({need}) <-> {j} ({complement})")
                            else:
                                print(f"No changes")

        #### Consistency Identification ####
        all_upper_scores, all_lower_scores, graph = consistency_identification(num_stems, graph, directions, score_map, all_upper_scores, all_lower_scores, cross_thresh)
        # print("After consistency check:")
        # print(graph)
            
        # Step 5: 退出条件
        # 若没有新增匹配，或所有未匹配 stem 的候选都为空
        if not changed:
            # ⚠️ 重新计算最新的 missing_matches
            latest_missing = {}
            for i, neighbors in graph.items():
                existing_dirs = {tag for _, tag in neighbors}
                missing = []
                if 'upper' not in existing_dirs:
                    missing.append('upper')
                if 'lower' not in existing_dirs:
                    missing.append('lower')
                if missing:
                    latest_missing[i] = missing
            # print(f"Latest Missing: {latest_missing}")

            latest_fully_matched = set(range(num_stems)) - set(latest_missing.keys())

            # 找出仍未 fully matched 的 stems
            still_unmatched = set(range(num_stems)) - latest_fully_matched
            # print(f"Still Unmateched: {still_unmatched}")

            still_missing = []
            for i in still_unmatched:
                upper_empty = len(all_upper_scores[i]) == 0
                lower_empty = len(all_lower_scores[i]) == 0
                if upper_empty or lower_empty:
                    still_missing.append(i)
            # print(f"Bottom Top Stems: {still_missing}")

            break
    
    ############ Final Cluster ###############
    # print(f"graph = {graph}")
    final_clusters = []
    cluster_labels = [-1] * num_stems
    cluster_id = 0

    for i in range(num_stems):
        if cluster_labels[i] != -1:
            continue  # 已分类
        
        # 用栈/队列来扩展整个连通分量
        cluster_members = []
        stack = [i]
        while stack:
            node = stack.pop()
            if cluster_labels[node] != -1:
                continue
            cluster_labels[node] = cluster_id
            cluster_members.append(node)
            for nei in graph[node]:
                if cluster_labels[nei[0]] == -1:
                    stack.append(nei[0])

        # assign an id to classified stems
        for m in cluster_members:
            cluster_labels[m] = cluster_id
        
        final_clusters.append(cluster_members)
        cluster_id += 1

    # print(f"final_clusters:")
    # for idx, cluster in enumerate(final_clusters):
    #     print(f"cluster {idx} = {cluster}")
    
    return final_clusters


def overlapping_merge(final_merged_clusters, cluster_centers, cluster_dirs, flower_lengths, final_clusters, cluster_profiles, e_thres, s_thres):
    clusters_n = len(final_merged_clusters)
    merged = [set(c) for c in final_merged_clusters]
    merged_flags = [False] * clusters_n

    adjacency = {i: set() for i in range(clusters_n)}

    score_map_cluster = {}
    ang_diff_cluster = {}

    for i, cluster_1 in enumerate(final_merged_clusters):
        c_1 = np.mean([cluster_centers[mem] for mem in cluster_1], axis=0)
        dir_sum_1 = np.sum([cluster_dirs[mem] for mem in cluster_1], axis=0)
        dir_1 = dir_sum_1 / np.linalg.norm(dir_sum_1)
        profile_1 = [cluster_profiles[mem] for mem in cluster_1]

        for j, cluster_2 in enumerate(final_merged_clusters):
            if j == i:
                continue
                
            c_2 = np.mean([cluster_centers[mem] for mem in cluster_2], axis=0)
            dir_sum_2 = np.sum([cluster_dirs[mem] for mem in cluster_2], axis=0)
            dir_2 = dir_sum_2 / np.linalg.norm(dir_sum_2)
            len_2 = np.sum([flower_lengths[mem] for mem in cluster_2], axis=0) # rough estimation
            
            e_1 = abs(angle_between(dir_1, dir_2))
            ang_diff_cluster[(i, j)] = e_1
            inter = line_intersection(c_1, dir_1, c_2, dir_2)
            radius = abs(np.linalg.norm(inter - c_2)) + len_2/2
            theta = abs(angle_between(dir_1, dir_2))
            s = radius*theta
            score_map_cluster[(i, j)] = s 
            # print(f"cluster {i} ↔ cluster {j} | "
            #           f"angle={np.rad2deg(e_1):.2f}°, score={s}")

    for i, cluster_1 in enumerate(final_merged_clusters):
        profile_1 = [cluster_profiles[mem] for mem in cluster_1]
        for j, cluster_2 in enumerate(final_merged_clusters):
            if j == i:
                continue
            profile_2 = [cluster_profiles[mem] for mem in cluster_2]
            not_overlap =  True
            for p_1 in profile_1:
                for b_1 in p_1:
                    low1, up1 = b_1[0], b_1[1]

                    for p_2 in profile_2:
                        for b_2 in p_2:
                            low2, up2 = b_2[0], b_2[1]
                            if not (up1 < low2 or up2 < low1):
                                not_overlap = False

            if not_overlap:
                s_i = score_map_cluster.get((i, j), float('inf'))
                # s_j = score_map_cluster.get((j, i), float('inf'))
                e = ang_diff_cluster.get((i, j), float('inf'))
                if e < e_thres and s_i < s_thres:
                    ## combine i and j cluster together
                    adjacency[i].add(j)
                    adjacency[j].add(i)
                    print(f"-> Merge overlapping candidate: cluster {i} <-> cluster {j} | "
                        f"angle={np.rad2deg(e):.2f}°, score={s_i}")
                    
                    ## check i, j, k merge conflict ##
                    neigh_i = list(adjacency[i])
                    neigh_j = list(adjacency[j])

                    check = None
                    neigh = None

                    if len(neigh_i) == 2:
                        check = i
                        neigh = neigh_i
                    elif len(neigh_j) == 2:
                        check = j
                        neigh = neigh_j
                    
                    if check is not None: 
                        k_1 = neigh[0]
                        k_2 = neigh[1]

                        # 如果 j↔k 不满足 merge 条件 → 冲突出现
                        s_k = min(
                            score_map_cluster.get((k_1, k_2), float('inf')),
                            score_map_cluster.get((k_2, k_1), float('inf'))
                        )
                        e_k = ang_diff_cluster.get((k_1, k_2), float('inf'))
                        ok_k = (e_k < e_thres and s_k < s_thres)

                        if not ok_k:
                            # 冲突：i-j-k 三角关系里 j-k 不匹配
                            # 需要在 (i,k) 和 (i,j) 中删除 score 较大的那条

                            s_k1 = min(
                                score_map_cluster.get((check, k_1), float('inf')),
                                score_map_cluster.get((k_1, check), float('inf'))
                            )

                            s_k2 = min(
                                score_map_cluster.get((check, k_2), float('inf')),
                                score_map_cluster.get((k_2, check), float('inf'))
                            )

                            # 找得分更差的那条边
                            if s_k1 > s_k2:
                                adjacency[check].discard(k_1)
                                adjacency[k_1].discard(check)
                                print(f"x Removed merge: cluster {check} <-> {k_1} due to conflict with {k_2}, "
                                    f"ang_{k_1}_{k_2} = {e_k} and score_{k_1}_{k_2} = {s_k}")
                            else:
                                adjacency[check].discard(k_2)
                                adjacency[k_2].discard(check)
                                print(f"x Removed merge: cluster {check} <-> {k_2} due to conflict with {k_1}, "
                                        f"ang_{k_1}_{k_2} = {e_k} and score_{k_1}_{k_2} = {s_k}")
                        else:    
                            print(f"Ternary merge: cluster {check} <-> {k_1} <-> {k_2}, "
                                    f"ang_{k_1}_{k_2} = {e_k} and score_{k_1}_{k_2} = {s_k}")

                        
    visited_clusters = set()
    merged_clusters = []

    for i in range(clusters_n):
        if i not in visited_clusters:
            group = set()
            dfs(i, group, visited_clusters, adjacency)

            merged_cluster = set()
            for idx in group:
                merged_cluster.update(merged[idx])
            merged_clusters.append(sorted(list(merged_cluster)))

    final_merged_clusters =  merged_clusters
    # print(f"After overlapping merge, final_merged_clusters = {final_merged_clusters}")


    #### merge ####
    final_clusters_1 = []
    for merged in final_merged_clusters:
        merged_stems = []
        for idx in merged:
            merged_stems.extend(final_clusters[idx])

        final_clusters_1.append(sorted(set(merged_stems)))  

    print(f"final_clusters_1 = {final_clusters_1}")

    return final_clusters_1

def final_cluster_visual(img_bgr, stem_masks, final_clusters_1, save_dir, filename):
    colored = img_bgr.copy()

    # 为每个 cluster 生成一个随机颜色
    cluster_colors = [
        [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        for _ in range(len(final_clusters_1))
    ]

    # 遍历每个 cluster
    for cluster_id, members in enumerate(final_clusters_1):
        # if len(members) <= 1:
        #     continue  # 跳过单 stem cluster
        color = cluster_colors[cluster_id]
        for idx in members:
            mask = stem_masks[idx]  # boolean 或 uint8 mask
            colored[mask > 0] = color
        
        #### stem line visual ####
        # all_coords = []
        # for idx in members:
        #     ys, xs = np.where(stem_masks[idx] > 0)
        #     coords = np.stack([xs, ys], axis=1).astype(np.float32)
        #     all_coords.append(coords)
        
        # if len(all_coords) == 0:
        #     continue

        # all_coords = np.concatenate(all_coords, axis=0)
        
        # # ---- 计算 PCA ----
        # mean, eigenvectors = cv2.PCACompute(all_coords, mean=None)
        # center = mean[0]
        # direction = eigenvectors[0]
        # direction = direction / np.linalg.norm(direction)

        # # ---- 确定方向长度（贯穿整个 cluster）----
        # projections = np.dot(all_coords - center, direction)
        # min_proj, max_proj = projections.min(), projections.max()
        # p1 = center + direction * min_proj * 1.5
        # p2 = center + direction * max_proj * 1.5

        # # ---- 绘制方向线 ----
        # # color = (0, 0, 0)  # 黑色主轴线（你可以改为红色或对应cluster颜色）
        # cv2.line(
        #     colored,
        #     tuple(p1.astype(int)),
        #     tuple(p2.astype(int)),
        #     color=color,
        #     thickness=2
        # )
    
    # Uncommon this for eval_cluster test
    # colored = cv2.resize(colored, None, fx=0.7, fy=0.7)
    # cv2.imshow("Predicted Stem Groups", colored)
    # cv2.waitKey(0)

    # 保存结果 Uncomment this for stem pairing test
    # save_path = os.path.splitext(img_path)[0] + "_clusters.jpg"
    # save_path = os.path.join(save_dir, filename)
    # cv2.imwrite(save_path, colored)
    # print("可视化结果已保存：", save_path)

def stem_pairing(masks, labels, bboxes, scores, img):

    centers, directions = [], []
    stem_masks = []
    stem_lengths = []
    lengths = []
    stem_bounds =[]

    #### Preprocessing ####
    masks, labels, bboxes, scores = prepossesing_stems(masks, labels, bboxes, scores)

    for mask, label, bbox, score in zip(masks, labels, bboxes, scores):

        # base on annotation, all masks sent to here is stem, no need to check
        # if model.dataset_meta['classes'][label] != 'Stem':
        #     continue
        if score < 0.4:
            continue

        x, y, w, h = bbox.astype(int)

        if w < 5 or h < 18:
            continue

        ys, xs = np.where(mask > 0)
        if len(xs) < 90:  
            continue
        
        top_y = np.min(ys)
        bottom_y = np.max(ys)
        stem_bounds.append([top_y, bottom_y])

        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        center, direction, length_u, length_l = PCA_feature(coords)
        centers.append(center)
        directions.append(direction)
        stem_masks.append(mask)
        stem_lengths.append([length_u, length_l])
        lengths.append(length_u + length_l)

    length_conf = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths) + 1e-6) #confidence is from 0-1

    ########## Score Counting ##########
    all_upper_scores, all_lower_scores, score_map = stem_scores(centers, directions, stem_lengths, length_conf)

    ############# Clustering ############
    num_stems = len(centers)
    final_clusters = clustering_by_stems(num_stems, all_upper_scores, all_lower_scores, directions, score_map)

    ###################### Final Cluster Merging ########################
    cluster_centers = []
    cluster_dirs = []
    cluster_lengths = []
    flower_lengths = []
    cluster_profiles = [] 
    #### Features Collection ####
    for cluster_id, members in enumerate(final_clusters):

        all_coords = []
        member_profile = []

        for idx in members:
            ys, xs = np.where(stem_masks[idx] > 0)
            coords = np.stack([xs, ys], axis=1)
            all_coords.append(coords)

            member_profile.append(stem_bounds[idx])

        all_coords = np.concatenate(all_coords, axis=0).astype(np.float32)

        center, direction, length_up, length_low = PCA_feature_cluster(all_coords)

        cluster_centers.append(center)
        cluster_dirs.append(direction)
        cluster_lengths.append([length_up, length_low])
        flower_lengths.append(length_up + length_low)
        cluster_profiles.append(member_profile)

    cluster_conf = (flower_lengths - np.min(flower_lengths)) / (np.max(flower_lengths) - np.min(flower_lengths) + 1e-6) #confidence is from 0-1

    #### Score Counting #### 
    final_upper_scores, final_lower_scores, score_map_final = stem_scores(cluster_centers, cluster_dirs, cluster_lengths, cluster_conf)

    #### clustering ####
    num_stems = len(cluster_centers)
    final_merged_clusters = clustering_by_stems(num_stems, final_upper_scores, final_lower_scores, cluster_dirs, score_map_final)
    
    #### Overlapping Merge ####
    e_thres = 20*np.pi/180
    s_thres = 30
    final_clusters_1 = overlapping_merge(final_merged_clusters, cluster_centers, cluster_dirs, flower_lengths, final_clusters, cluster_profiles, e_thres, s_thres)

    save_dir = ""
    filename = ""
    final_cluster_visual(img, stem_masks, final_clusters_1, save_dir, filename)
    return final_clusters_1, masks





######### Start Pairing !!! ##########

if __name__ == "__main__":

    ############ Initialize the model #############
    # register all modules in mmdet into the registries
    begin = time.time()
    register_all_modules()
    cfg = Config.fromfile('../stem_configs/mask-rcnn_r50-caffe_fpn_ms-poly-3x_Stem_tinytest.py')
    checkpoint_file = 'tutorial_exps/epoch_150.pth'
    model = init_detector(cfg, checkpoint_file, device='cuda:0')
    model.dataset_meta = cfg.metainfo
    visualizer_now = VISUALIZERS.build(model.cfg.visualizer)
    visualizer_now.dataset_meta = model.dataset_meta



    ############ Test the data #############
    test_dir = './Stem_Segmentation/test/temp'
    save_dir = os.path.join(test_dir, "results")  # 新建一个 results 文件夹

    os.makedirs(save_dir, exist_ok=True)  # 若不存在则创建
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    image_files.sort()

    for idx, filename in enumerate(image_files, start=1):
        img_path = os.path.join(test_dir, filename)
        img_bgr = mmcv.imread(img_path, channel_order='bgr')

        result = inference_detector(model, img_bgr)
        # middle = time.time()
        # print("Model Inference cost:", middle - begin, "s")

        masks = result.pred_instances.masks.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()

        centers, directions = [], []
        stem_masks = []
        stem_lengths = []
        lengths = []
        stem_bounds =[]

        out_file = os.path.join(test_dir, f'result_{idx}.jpg')
        img_vis = img_bgr.copy()
        print(f"Before preprocessing, there are {len(masks)} masks")
        #### Preprocessing ####
        masks, labels, bboxes, scores = prepossesing_stems(masks, labels, bboxes, scores)
        print(f"After preprocessing, there are {len(masks)} masks")
        for mask, label, bbox, score in zip(masks, labels, bboxes, scores):
            if model.dataset_meta['classes'][label] != 'Stem':
                continue
            if score < 0.4:
                continue

            x1, y1, x2, y2 = bbox.astype(int)
            w, h = x2 - x1, y2 - y1

            if w < 5 or h < 18:
                continue

            ys, xs = np.where(mask > 0)

            if len(xs) < 90:  
                continue

            top_y = np.min(ys)
            bottom_y = np.max(ys)
            stem_bounds.append([top_y, bottom_y])

            coords = np.stack([xs, ys], axis=1).astype(np.float32)
            center, direction, length_u, length_l = PCA_feature(coords)
            centers.append(center)
            directions.append(direction)
            stem_masks.append(mask)
            stem_lengths.append([length_u, length_l])
            lengths.append(length_u + length_l)

        length_conf = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths) + 1e-6) #confidence is from 0-1

        # print(f"directions={directions}")

        ######## visual test for sector filter ##########
        #candidate_visual(img_bgr, centers, directions, stem_lengths, stem_masks, length_conf, img_path, indx)

        ########## Score Counting ##########
        all_upper_scores, all_lower_scores, score_map = stem_scores(centers, directions, stem_lengths, length_conf)

        # print("all_upper_scores:")
        # for i, scores in enumerate(all_upper_scores):
        #     print(f"  Stem {i}:")
        #     for score, c2, j in scores:
        #         print(f"    -> candidate {j}, score={score:.3f}, center={c2}")

        # print("\nall_lower_scores:")
        # for i, scores in enumerate(all_lower_scores):
        #     print(f"  Stem {i}:")
        #     for score, c2, j in scores:
        #         print(f"    -> candidate {j}, score={score:.3f}, center={c2}")

        ############# Clustering ############
        num_stems = len(centers)
        final_clusters = clustering_by_stems(num_stems, all_upper_scores, all_lower_scores, directions, score_map)

        ###################### Final Cluster Merging ########################

        cluster_centers = []
        cluster_dirs = []
        cluster_lengths = []
        flower_lengths = []
        cluster_profiles = [] 
        #### Features Collection ####
        for cluster_id, members in enumerate(final_clusters):

            all_coords = []
            member_profile = []

            for idx in members:
                ys, xs = np.where(stem_masks[idx] > 0)
                coords = np.stack([xs, ys], axis=1)
                all_coords.append(coords)
   
                member_profile.append(stem_bounds[idx])

            all_coords = np.concatenate(all_coords, axis=0).astype(np.float32)

            center, direction, length_up, length_low = PCA_feature_cluster(all_coords)

            cluster_centers.append(center)
            cluster_dirs.append(direction)
            cluster_lengths.append([length_up, length_low])
            flower_lengths.append(length_up + length_low)
            cluster_profiles.append(member_profile)
        
        # for idx, profile in enumerate(cluster_profiles):
        #     print(f"cluster_{idx} has profile: {profile}")
        cluster_conf = (flower_lengths - np.min(flower_lengths)) / (np.max(flower_lengths) - np.min(flower_lengths) + 1e-6) #confidence is from 0-1

        #### Score Counting #### 
        final_upper_scores, final_lower_scores, score_map_final = stem_scores(cluster_centers, cluster_dirs, cluster_lengths, cluster_conf)

        # print("final_upper_scores:")
        # for i, scores in enumerate(final_upper_scores):
        #     print(f"  Cluster {i}:")
        #     for score, c2, j in scores:
        #             print(f"    -> candidate {j}, score={score:.3f}, center={c2}")

        # print("\nfinal_lower_scores:")
        # for i, scores in enumerate(final_lower_scores):
        #     print(f"  Cluster {i}:")
        #     for score, c2, j in scores:
        #             print(f"    -> candidate {j}, score={score:.3f}, center={c2}")

        #### clustering ####
        print(f"\n--- Here starts the final clusters merge ---")
        num_stems = len(cluster_centers)
        
        final_merged_clusters = clustering_by_stems(num_stems, final_upper_scores, final_lower_scores, cluster_dirs, score_map_final)
        
        #### Overlapping Merge ####
        e_thres = 20*np.pi/180
        s_thres = 30
        final_clusters_1 = overlapping_merge(final_merged_clusters, cluster_centers, cluster_dirs, flower_lengths, final_clusters, cluster_profiles, e_thres, s_thres)

        # end = time.time()
        # print("Stem Pairing cost:", end - start, "s")
        ######## Visual test for final cluster ########
        final_cluster_visual(img_bgr, stem_masks, final_clusters_1, save_dir, filename)
    
   

    




