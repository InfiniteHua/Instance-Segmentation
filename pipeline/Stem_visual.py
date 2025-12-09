import os
import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector

from mmdet.utils import register_all_modules
from mmengine.config import Config 
import numpy as np
import cv2

# 将mmdet中的所有模块注册到注册表中
register_all_modules()

# --- 模型加载部分与您的代码相同 ---
cfg = Config.fromfile('../stem_configs/mask-rcnn_r50-caffe_fpn_ms-poly-3x_Stem_tinytest.py')
checkpoint_file = 'tutorial_exps/epoch_150.pth'
model = init_detector(cfg, checkpoint_file, device='cuda:0')
model.dataset_meta = cfg.metainfo

test_dir = './Stem_Segmentation/test/temp'
image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort()

# 创建一个新的文件夹来保存带有编号的结果
os.makedirs(test_dir, exist_ok=True)

# --- 新增：定义您想要绘制和编号的Stem索引 ---
# 使用集合(set)可以更快地进行查找
target_indices = {2, 4, 5, 6, 7, 9, 12, 13,14, 15, 18, 19, 25}

# 遍历所有测试图片
for idx, filename in enumerate(image_files, start=1):
    img_path = os.path.join(test_dir, filename)
    img_bgr = mmcv.imread(img_path, channel_order='bgr')
    img_vis = img_bgr.copy() # 创建一个副本用于绘制

    result = inference_detector(model, img_bgr)

    # --- 从这里开始是修改的核心逻辑 ---

    # 1. 提取所有检测实例的属性
    pred_instances = result.pred_instances
    labels = pred_instances.labels.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    masks = pred_instances.masks.cpu().numpy()
    bboxes = pred_instances.bboxes.cpu().numpy()

    # 2. 找到 'Stem' 类别对应的标签索引
    class_names = model.dataset_meta['classes']
    stem_label_index = -1
    if 'Stem' in class_names:
        stem_label_index = class_names.index('Stem')
    
    if stem_label_index == -1:
        print(f"警告: 数据集类别中未找到 'Stem'。跳过图片 {filename}")
        continue

    # 3. 筛选出所有 'Stem' 实例，并按置信度降序排序
    stem_detections = []
    for i in range(len(labels)):
        # 应用置信度阈值
        if scores[i] < 0.4:
            continue
        # 只保留 'Stem' 类别
        if labels[i] == stem_label_index:
            stem_detections.append({
                'score': scores[i],
                'mask': masks[i],
                'bbox': bboxes[i]
            })
    
    # 按置信度得分从高到低排序
    # stem_detections.sort(key=lambda x: x['score'], reverse=True)

    # 4. 遍历排序后的Stem，只绘制并标记指定编号的实例
    for stem_counter, detection in enumerate(stem_detections, start=0):
        # 检查当前stem的编号是否在我们想要的目标列表中
        
        mask = detection['mask']
        bbox = detection['bbox'].astype(int)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        if w < 5 or h < 18:
            continue

        ys, xs = np.where(mask > 0)
        if len(xs) <= 95:  
            print(f"stem_{stem_counter} is removed with {len(xs)} pixels")
            continue

            # 定义一个颜色来绘制掩码 (这里使用黄色 BGR)
        color = (0, 255, 255) 
            
        # 将掩码区域用半透明的颜色覆盖
        # img_vis[mask]会选中所有mask为True的像素点
        img_vis[mask] = img_vis[mask] * 0.5 + np.array(color) * 0.5
            
        # 在边界框的左上角添加编号文本
        label_text = str(stem_counter)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_color = (255, 255, 255)  # 白色

        # 1️⃣ 计算bbox中心点
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 2️⃣ 计算文字尺寸（用于居中对齐）
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        # 3️⃣ 调整文字位置，使其中心对齐bbox
        text_x = cx - text_width // 2
        text_y = cy + text_height // 2

        # 4️⃣ 绘制文字
        cv2.putText(img_vis, label_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # 5. 保存手动绘制好的图片
    out_file = os.path.join(test_dir, f'labeled_stem_{filename}')
    cv2.imwrite(out_file, img_vis)

print(f"所有 {len(image_files)} 张图片的结果已保存至: {test_dir}")