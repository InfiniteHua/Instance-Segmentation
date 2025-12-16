import numpy as np
import cv2

def point_in_any_mask(x, y, segs, safe_margin=20):
    px, py = float(x), float(y)
    point = np.array([px, py], dtype=float)
    for seg in segs:
        cnt = np.array(seg["contour"])
        inside = cv2.pointPolygonTest(cnt, (px, py), measureDist=False)
        if inside >= 0:  # mask 内或边缘
            start = np.array(seg["start"], dtype=float)
            end   = np.array(seg["end"],   dtype=float)
            center = np.array(seg["center"], dtype=float)

            d_start = np.linalg.norm(point - start)
            d_end   = np.linalg.norm(point - end)

            final_point = point.copy()

            if d_start < safe_margin:
                direction = center - start
                dist = np.linalg.norm(direction)
                if dist > 1e-8:
                    unit = direction / dist
                    final_point = start + unit * min(safe_margin, dist)

            elif d_end < safe_margin:
                direction = center - end
                dist = np.linalg.norm(direction)
                if dist > 1e-8:
                    unit = direction / dist
                    final_point = end + unit * min(safe_margin, dist)
            return final_point
    return None

def get_offset(point, center, offset_px=20):
    # offset to mask center
    vec = center - point
    dist_to_center = np.linalg.norm(vec)
    unit = vec / dist_to_center

    if dist_to_center <= 1e-8:
        used_offset = 0.0
        return tuple(point)
    else:
        # set limit to the offset
        max_allowed = dist_to_center * 0.9
        used_offset = min(offset_px, max_allowed)
    final_point = point + used_offset * unit

    return final_point


def get_pick_point(segs, stems):
    """
    pred_groups: list of list[segments]   # 每组的segment列表
    dense_stems: list of list[(x,y)]      # 每组的dense stems
    top_idx: 选择的组
    """
    # 1/4 point along the stem
    y_min = stems[:, 1].min()
    y_max = stems[:, 1].max()
    target_y = y_min + 0.25 * (y_max - y_min)
    idx = np.argmin(np.abs(stems[:, 1] - target_y))
    pick_candidate = stems[idx]  # (x,y)
    px, py = pick_candidate

    # N = len(stems)
    # pick_idx = max(0, min(N - 1, int(N * 0.25)))  # 防止越界
    # pick_candidate = stems[pick_idx]              # (x, y)
    # px, py = pick_candidate

    safe_point = point_in_any_mask(px, py, segs)
    if safe_point is not None:
        return tuple(safe_point)
    
    # find the closest mask point
    mask_points = []
    owner_idx = []  # parallel list to track which seg each point came from
    for si, seg in enumerate(segs):
        cnt = np.array(seg["contour"])[:, 0, :]  # (n,1,2) → (n,2)
        mask_points.append(cnt)
        owner_idx.extend([si] * cnt.shape[0])

    mask_points = np.vstack(mask_points)  # (M, 2)
    owner_idx = np.array(owner_idx, dtype=int)  # (M,)

    dists = np.linalg.norm(mask_points - pick_candidate, axis=1)
    nearest_idx = np.argmin(dists)
    nearest_mask_point = mask_points[nearest_idx]
    seg_id = int(owner_idx[nearest_idx])
    seg_p = segs[seg_id]

    final_point = get_offset(nearest_mask_point, seg_p["center"])
    return tuple(final_point)


def draw_pp(img, point, size=8, color=(0,0,255), thickness=2):
    output = img.copy()
    x, y = map(int, point)

    cv2.line(output, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(output, (x - size, y + size), (x + size, y - size), color, thickness)
    return output