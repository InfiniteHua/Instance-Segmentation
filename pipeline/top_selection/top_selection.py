from .top_selection_util import read_coco, draw_segment_groups, _compute_trendline, build_pred_segments
import cv2
import numpy as np
from collections import defaultdict, deque
from matplotlib.path import Path
from skimage.draw import polygon
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

def draw_points_on_image(img, points, color=(0, 0, 255), radius=4, thickness=-1, show_text=True):
    # Copy image to avoid modifying the original
    output = img.copy()

    for i, (x, y) in enumerate(points):
        # Convert to int pixel coordinates
        px, py = int(round(x)), int(round(y))
        cv2.circle(output, (px, py), radius, color, thickness)

        # if show_text:
        #     text = f"P{i}:({x:.1f},{y:.1f})"
        #     cv2.putText(output, text, (px + 5, py - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return output


## ===========================================================================================================

def resample_fixed_distance(points, step=0.1):
    # --- Safety check: fewer than 2 points ---
    if len(points) < 2:
        # return original point or empty list depending on use
        return points.copy()
    
    deltas = np.diff(points, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    dist_along = np.concatenate(([0], np.cumsum(seg_lengths)))
    total_length = dist_along[-1]
    if total_length == 0:
        return points[:1]
    new_d = np.arange(0, total_length, step)
    new_x = np.interp(new_d, dist_along, points[:, 0])
    new_y = np.interp(new_d, dist_along, points[:, 1])
    return np.stack((new_x, new_y), axis=1)

def fit_quadratic_curve(start, end, contour, num_points=50):
    """
    Fit a quadratic curve y = a*x^2 + b*x + c passing approximately through the contour,
    while fixing the start and end points. Returns num_points samples along the curve.
    """

    contour = np.array(contour)
    contour = np.reshape(contour,(-1,2))
    x0, y0 = start
    x2, y2 = end

    # shift coordinates to start at 0
    x_shift = x0
    y_shift = y0
    X = contour[:,0] - x_shift
    Y = contour[:,1] - y_shift
    x_end = x2 - x_shift

    # Quadratic: y = a x^2 + b x
    # Constraint: f(0) = 0 -> automatically satisfied
    #             f(x_end) = y_end - y_shift -> y_end - y_shift = a*x_end^2 + b*x_end
    A = np.column_stack([X**2, X])
    Y_target = Y

    # Solve least squares
    coeff, *_ = np.linalg.lstsq(A, Y_target, rcond=None)
    a, b = coeff

    # sample points along x
    xs = np.linspace(0, x_end, num_points)
    ys = a*xs**2 + b*xs
    xs += x_shift
    ys += y_shift
    points = np.column_stack([xs, ys])
    points = points[np.argsort(points[:, 1])]
    return points

def connect_stem_segments_quadratic(stem, num_points=50):
    """
    Connect segments of a stem using quadratic curve fit per segment
    and interpolate between segments.
    """
    points = []
    for i, seg in enumerate(stem):
        contour = seg.get("contour", [])
        start = seg["start"]
        end = seg["end"]

        if len(contour) >= 3:
            midline = fit_quadratic_curve(start, end, contour, num_points)
        else:
            # fallback straight line
            midline = np.linspace(start, end, num_points)

        # avoid duplicate points
        if len(points) > 0 and np.allclose(points[-1], midline[0]):
            points.extend(midline[1:].tolist())
        else:
            points.extend(midline.tolist())

        # connect end to next segment start
        if i < len(stem) - 1:
            next_start = stem[i+1]["start"]
            connect_line = np.linspace(midline[-1], next_start, num_points)
            points.extend(connect_line[1:].tolist())

    return np.array(points)

def process_stems(image, step = 10):
    dense_stems = []
    for idx, stem in enumerate(image):
        sorted_indices = sorted(range(len(stem)), key=lambda i: stem[i]['center'][1])
        stem = [stem[i] for i in sorted_indices]
        # if idx>=4:
        #     print("here!")
        dense_points = connect_stem_segments_quadratic(stem, 50)
        dense_stems.append(resample_fixed_distance(dense_points, step))
    return dense_stems


def is_point_in_contour(point, contour, radius=4):
    """Return True if the point (x, y) is inside the given contour polygon."""
    # contour = np.reshape(contour, (-1,2))
    # path = Path(contour)
    # return path.contains_point(point)
    contour = np.reshape(contour, (-1, 2))
    path = Path(contour)
    
    # If point itself is inside
    if path.contains_point(point):
        return True

    # Otherwise, check a circle of nearby points around it
    angles = np.linspace(0, 2*np.pi, 36)  # 36 samples around the circle
    circle_points = np.array([
        [point[0] + radius * np.cos(a), point[1] + radius * np.sin(a)]
        for a in angles
    ])
    
    return np.any(path.contains_points(circle_points))

#### This version is to count skeleton line points in the stem contour ####
def count_crossings(dense_stems, image, radius=2):
    """
    Optimized version: Counts how many times each stem's points fall within other stems' contours.
    Uses vectorized point-in-polygon checks and !!! bounding box prefiltering !!!.
    """
    n_stems = len(image)
    losses = np.zeros(n_stems, dtype=int)

    # --- Precompute paths and bounding boxes ---
    contours_per_stem, bbox_per_stem = [], []
    for stem in image:
        stem_contours, stem_bboxes = [], []
        for seg in stem:
            contour = seg.get("contour")
            if contour is None or len(contour) < 3:
                continue
            contour = np.reshape(contour, (-1, 2))
            path = Path(contour)
            xmin, ymin = contour.min(axis=0)
            xmax, ymax = contour.max(axis=0)
            stem_contours.append(path)
            stem_bboxes.append((xmin, ymin, xmax, ymax)) # save as bounding box
        contours_per_stem.append(stem_contours)
        bbox_per_stem.append(stem_bboxes)

    # --- Precompute small set of offset directions ---
    offsets = radius * np.array([
        [0, 0],  # center
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1/np.sqrt(2), 1/np.sqrt(2)],
        [-1/np.sqrt(2), 1/np.sqrt(2)],
        [1/np.sqrt(2), -1/np.sqrt(2)],
        [-1/np.sqrt(2), -1/np.sqrt(2)]
    ])

    # --- Main loop ---
    for i, points in enumerate(dense_stems):
        for j in range(n_stems):
            if i == j:
                continue

            contours, bboxes = contours_per_stem[j], bbox_per_stem[j]
            for path, (xmin, ymin, xmax, ymax) in zip(contours, bboxes):
                # Expand bbox by radius to prefilter
                mask = (
                    (points[:, 0] >= xmin - radius) & (points[:, 0] <= xmax + radius) &
                    (points[:, 1] >= ymin - radius) & (points[:, 1] <= ymax + radius)
                )
                if not np.any(mask):
                    continue

                pts = points[mask]
                # Generate offset samples (8 directions + center)
                pts_expanded = np.concatenate([pts + off for off in offsets], axis=0)
                inside = path.contains_points(pts_expanded)
                # Reshape back to groups of len(offsets) per original point (9, M)
                inside_per_point = inside.reshape(len(offsets), -1).any(axis=0)
                losses[i] += np.count_nonzero(inside_per_point)

    return losses.tolist()

#### This version is to calculate the overlapping area of stem and skeleton line masks #### 
def count_crossings_masks(dense_stems, image, radius=2, img_size=(1280,1280)):
    """
    Optimized version: Counts how many times each stem's points fall within other stems' contours.
    Uses opencv common area calculation from masks and !!! bounding box prefiltering !!!.
    """
    H, W = img_size
    n_stems = len(image)
    # losses = np.zeros(n_stems, dtype=int)
    losses = [0] * len(dense_stems)
    # --- Precompute paths and bounding boxes ---
    contours_per_stem, bbox_per_stem = [], []
    stem_masks = [] # all stem masks
    for stem in image:
        stem_contours, stem_bboxes = [], []
        mask = np.zeros((H, W), dtype=np.uint8)
        for seg in stem:
            contour = seg.get("contour")
            if contour is None or len(contour) < 3:
                continue
            contour = np.reshape(contour, (-1, 2))
            xmin, ymin = contour.min(axis=0)
            xmax, ymax = contour.max(axis=0)
            stem_bboxes.append((xmin, ymin, xmax, ymax)) # save as bounding box
            cv2.fillPoly(mask, [contour], 255)

        bbox_per_stem.append(stem_bboxes)
        stem_masks.append(mask)

    # --- Main loop ---
    for i, points in enumerate(dense_stems):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            # treat as huge loss or skip
            losses[i] = (999999)
            continue
        # Create mask for skeleton[i]
        skel_mask = np.zeros((H, W), dtype=np.uint8)

        pts_int = points.astype(int)
        for (x, y) in pts_int:
            # Draw thick dots to approximate thick polyline
            cv2.circle(skel_mask, (x, y), radius, 255, -1)

        for j in range(n_stems):
            if i == j:
                continue

            bboxes = bbox_per_stem[j]
            passes_prefilter = False
            for (xmin, ymin, xmax, ymax) in bboxes:
                # Expand bbox by radius to prefilter
                mask_bb = (
                    (points[:, 0] >= xmin - radius) & (points[:, 0] <= xmax + radius) &
                    (points[:, 1] >= ymin - radius) & (points[:, 1] <= ymax + radius)
                )
                if np.any(mask_bb):
                    passes_prefilter = True
                    break  # one bbox passed → j is candidate
            
            if not passes_prefilter:
                continue  # skip expensive bitwise_and computation

            inter = cv2.bitwise_and(skel_mask, stem_masks[j])
            losses[i] += np.count_nonzero(inter)

    return losses

def segments_intersect(p1, p2, p3, p4):
    """Check whether segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and \
           (ccw(p1, p2, p3) != ccw(p1, p2, p4))

def count_crossings_times(dense_stems):
    n = len(dense_stems)
    crossings = [0] * n

    for i in range(n):
        pts_i = dense_stems[i]
        if pts_i is None or len(pts_i) < 2:
            continue
        p1 = pts_i[0]
        p2 = pts_i[-1]

        for j in range(n):
            if i == j:
                continue
            pts_j = dense_stems[j]
            if pts_j is None or len(pts_j) < 2:
                continue
            p3 = pts_j[0]
            p4 = pts_j[-1]

            # check segment intersections between stem i & j
            if segments_intersect(p1, p2, p3, p4):
                crossings[i] += 1

    return crossings

#########################################################################################################
# use skeleton
def process_stems_skeleton(image, step=10):
    dense_stems = []
    for idx, stem in enumerate(image):
        # for the stem segments in one cluster, order them following y axis
        sorted_indices = sorted(range(len(stem)), key=lambda i: stem[i]['center'][1])
        stem = [stem[i] for i in sorted_indices]
        # if idx>=4:
        #     print("here!")
        dense_points = connect_stem_segments_skeleton(stem, 50)
        dense_stems.append(resample_fixed_distance(dense_points, step))
    return dense_stems

# using the stem segments from one cluster to find the skeleton line of it
def connect_stem_segments_skeleton(stem, num_points=50): 
    points = []
    
    img = np.zeros((1280, 1280, 1), dtype=np.uint8)
    for i, seg in enumerate(stem):
        contour = seg.get("contour", [])

        cv2.fillPoly(img, [contour], 255)

    skeleton = skeletonize(img).astype(np.uint8)

    ys, xs, _ = np.nonzero(skeleton)
    coords = np.stack((xs, ys), axis=1)

    points.extend(coords.tolist())
        

    return np.array(points)

if __name__ == "__main__":
    idx = 0
    acc = 0
    while True:
        segments, gt_groups, img = read_coco(r"./Stem_Segmentation/test/4", r"_annotations_pickable.coco.json", idx)
        if segments == None:
            break

        # collect the seg info for each cluster
        print(f"gt_groups = {gt_groups}")
        groups = [[] for i in gt_groups]
        for seg in segments:
            for gidx, g in enumerate(gt_groups.values()):
                if seg["index"] in g:
                    groups[gidx].append(seg)
                    break

        lines = []
        for g in groups:
            trend = _compute_trendline(g)
            lines.append(trend)


        ##### use quadratic to extract trend line
        # dense_stems = process_stems(groups, 5)

        ##### use skeleton to extract trend line
        dense_stems = process_stems_skeleton(groups, 5)

        losses = np.array(count_crossings_masks(dense_stems, groups))

        # add the tolerance to loss in case of incorrect masks
        # tolerance = 10

        min_loss = np.min(losses)

        # effective_threshold = max(min_loss, tolerance)

        # candidates = np.where(losses <= effective_threshold)[0]
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
        if all([s['pickable'] for s in groups[top_idx]]):
            acc +=1
            print("id:", idx, " correct")
        else:
            print("id:", idx, )



        #### find relative up and down
        # non_intersected_idx = find_non_intersecting_lines_in_frame(lines)
        # if len(non_intersected_idx)>0:
        #     top_lines = [lines[id] for id in non_intersected_idx]
        #     top_segments = [groups[id] for id in non_intersected_idx]
        # else:
        #     topmost, relations = find_topmost_line(lines, groups)
        #     top_lines = [lines[topmost]]
        #     top_segments = [groups[topmost]]



        #### naive implementation, count num of intersections on mask
        # intersections = line_intersections(lines)
        # img = draw_points_on_image(img, intersections)

        # score = compute_group_scores(groups, intersections)

        # best_group, _, _ = select_best_group(groups, score)

        
        # draw_segment_groups(img, groups, [lines[top_idx]], save=True, save_dir=r"C:\Users\zihuan\Documents\9_post_processing\cv_images\test_output", save_idx=idx)
        draw_segment_groups(img, groups, None, save=True, save_dir=r"./Stem_Segmentation/test/4_result/1", save_idx=idx)
        idx +=1
        # break

    print(f"overall accuracy: {acc/idx}")