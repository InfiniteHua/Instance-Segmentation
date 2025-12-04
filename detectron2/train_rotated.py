# train_flowers_rotated.py
import os
import copy
import json
import cv2
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.structures import RotatedBoxes, Instances, BoxMode, PolygonMasks
import detectron2.data.transforms as T

# ---------- 1) Dataset Converter ----------
def get_rotated_flower_dicts(img_dir, ann_json):
    """
    Load COCO-style annotations where bbox is [cx, cy, w, h, angle].
    Angle should be in degrees. 
    Detectron2 coordinate system: angle increases counter-clockwise (CCW).
    """
    with open(ann_json, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annos_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        annos_by_image.setdefault(img_id, []).append(ann)

    dataset_dicts = []
    for img_id, img_info in images.items():
        record = {}
        # Ensure path is correct
        file_name = os.path.join(img_dir, img_info["file_name"]) if not os.path.isabs(img_info["file_name"]) else img_info["file_name"]
        record["file_name"] = file_name
        record["height"] = img_info.get("height", None)
        record["width"] = img_info.get("width", None)
        # If strict, you might want to read shapes here, but for speed we skip if json has it
        if record["height"] is None or record["width"] is None:
             img_temp = cv2.imread(file_name)
             if img_temp is not None:
                 record["height"], record["width"] = img_temp.shape[:2]
        
        record["image_id"] = img_id
        record["annotations"] = []

        anns = annos_by_image.get(img_id, [])
        for ann in anns:
            bbox = ann.get("bbox", None)
            if bbox is None or len(bbox) != 5:
                continue
            # Detectron2 需要 0=Stem, 1=Flower
            original_id = ann.get("category_id", 0)
            if original_id == 1:
                new_id = 0
            elif original_id == 2:
                new_id = 1
            else:
                # 如果遇到其他奇怪的 ID，可以选择跳过或者归为 0
                # print(f"Warning: unknown category_id {original_id}, skipping")
                continue
            # Note: Detectron2 RotatedBox expects [cx, cy, w, h, angle_degrees]
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWHA_ABS, # Use Rotated Absolute Mode
                "category_id": new_id,
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": ann.get("segmentation", [])
            }
            record["annotations"].append(obj)
        dataset_dicts.append(record)
    return dataset_dicts

def register_rotated_flowers():
    # Update these paths to your actual paths
    train_img_dir = "flowers/train"
    val_img_dir = "flowers/valid"
    train_ann = os.path.join(train_img_dir, "_annotations_rotated.coco.json")
    val_ann = os.path.join(val_img_dir, "_annotations_rotated.coco.json")

    # Use try-except to avoid re-register errors if run multiple times in notebook/script
    try:
        DatasetCatalog.register("flowers_rotated_train", lambda: get_rotated_flower_dicts(train_img_dir, train_ann))
        MetadataCatalog.get("flowers_rotated_train").set(thing_classes=["Stem", "flower"], evaluator_type="coco")
    except AssertionError:
        pass

    try:
        DatasetCatalog.register("flowers_rotated_val", lambda: get_rotated_flower_dicts(val_img_dir, val_ann))
        MetadataCatalog.get("flowers_rotated_val").set(thing_classes=["Stem", "flower"], evaluator_type="coco")
    except AssertionError:
        pass

# ---------- 2) Custom Mapper for Rotated Boxes ----------
class SimpleRotatedMapper:
    """
    Reads image, converts to Tensor (C, H, W), and sets up RotatedBoxes.
    """
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, dataset_dict):
        # 1. Deep copy to prevent side-effects on the global dataset list
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # 2. Read Image
        image = cv2.imread(dataset_dict["file_name"])
        if image is None:
            raise FileNotFoundError(f"Image not found: {dataset_dict['file_name']}")
        
        # 3. Check/Update Size if missing (optional safety)
        h, w = image.shape[:2]
        dataset_dict["height"] = h
        dataset_dict["width"] = w

        # 4. Convert Image to Tensor for Detectron2 Model
        # Format: (C, H, W), Float type
        # Transpose from (H, W, C) -> (C, H, W)
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        # 5. Process Annotations
        annos = dataset_dict.get("annotations", [])
        instances = Instances((h, w))
        
        boxes = []
        classes = []
        masks = []
        
        for ann in annos:
            # Remap Logic (Safe here because we deep copied)
            cat_id = ann["category_id"]
            boxes.append(ann["bbox"])
            classes.append(cat_id)
            masks.append(ann["segmentation"])

        if len(boxes) > 0:
            # Create RotatedBoxes
            instances.gt_boxes = RotatedBoxes(torch.tensor(boxes, dtype=torch.float32))
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            instances.gt_masks = PolygonMasks(masks)
        else:
            instances.gt_boxes = RotatedBoxes(torch.zeros((0, 5), dtype=torch.float32))
            instances.gt_classes = torch.tensor([], dtype=torch.int64)

        return {
            "image": image_tensor, 
            "instances": instances,
            "height": h,
            "width": w,
            "image_id": dataset_dict["image_id"]
        }

# ---------- 3) Trainer ----------
class RotatedTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # Use our clean mapper
        return build_detection_train_loader(cfg, mapper=SimpleRotatedMapper(is_train=True))

# ---------- 4) Main Configuration & Training ----------
def train():
    register_rotated_flowers()

    cfg = get_cfg()
    # Start with a standard Faster R-CNN config (Detectron2 logic)
    # Cascade R-CNN is fine, but Mask R-CNN head might conflict if not turned off.
    cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # ---------------------------------------------------------
    # CRITICAL ROTATED CONFIGS
    # ---------------------------------------------------------
    # 1. Enable RRPN
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
    
    # 2. Change Anchor Generator to Rotated
    cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
    # Define angles (in degrees). Standard: -90, 0, 90
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, 0, 90]]
    
    # 3. Set ROI Head to handle Rotated Boxes
    # StandardROIHeads supports RotatedBoxes IF the box_head is configured correctly
    cfg.MODEL.ROI_BOX_HEAD.BBOX_TYPE = "RotatedBoxes"
    
    # 4. Turn off Mask (unless you have rotated masks)
    cfg.MODEL.MASK_ON = True

    cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignRotated"
    # ---------------------------------------------------------

    cfg.DATASETS.TRAIN = ("flowers_rotated_train",)
    cfg.DATASETS.TEST = ("flowers_rotated_val",)
    
    # Dataloader settings
    cfg.DATALOADER.NUM_WORKERS = 2
    # Windows fix: set to 0 if 2 causes "BrokenPipe" or "Pickle" errors
    if os.name == 'nt':
        cfg.DATALOADER.NUM_WORKERS = 0
        
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001 # 2.5e-4 might be too low for start, adjusted slightly
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = []
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # Stem, flower

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = RotatedTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    # Windows Multi-processing safety
    torch.multiprocessing.freeze_support()
    train()