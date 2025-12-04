# train_flowers_rotated.py
import os
import random
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.structures import RotatedBoxes, Instances
from detectron2.structures.boxes import BoxMode
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper

import json
import cv2

# ---------- 1) dataset converter / registrant ----------
def get_rotated_flower_dicts(img_dir, ann_json):
    """
    Return dataset dicts in Detectron2 format, with rotated bbox as bbox and BoxMode.XYWHA_ABS.
    ann_json: path to COCO style json (images, annotations, categories), but annotations have rotated bbox [cx,cy,w,h,theta]
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
        file_name = os.path.join(img_dir, img_info["file_name"]) if not os.path.isabs(img_info["file_name"]) else img_info["file_name"]
        record["file_name"] = file_name
        record["height"] = img_info.get("height", None) or cv2.imread(file_name).shape[0]
        record["width"] = img_info.get("width", None) or cv2.imread(file_name).shape[1]
        record["image_id"] = img_id

        anns = annos_by_image.get(img_id, [])
        objs = []
        for ann in anns:
            bbox = ann.get("bbox", None)
            if bbox is None or len(bbox) != 5:
                # skip invalid entries
                print(f"Warning: ann id {ann.get('id')} invalid bbox {bbox}, skipping")
                continue
            obj = {
                "bbox": bbox,  # [cx, cy, w, h, theta]
                "bbox_mode": BoxMode.XYWHA_ABS,
                "category_id": ann.get("category_id", 0),
                # keep segmentation if exists (optional)
                "segmentation": ann.get("segmentation", []),
                "iscrowd": ann.get("iscrowd", 0)
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register_rotated_flowers():
    train_img_dir = "flowers/train"
    val_img_dir = "flowers/valid"
    train_ann = os.path.join(train_img_dir, "_annotations_rotated.coco.json")
    val_ann = os.path.join(val_img_dir, "_annotations_rotated.coco.json")

    DatasetCatalog.register("flowers_rotated_train", lambda: get_rotated_flower_dicts(train_img_dir, train_ann))
    DatasetCatalog.register("flowers_rotated_val", lambda: get_rotated_flower_dicts(val_img_dir, val_ann))
    MetadataCatalog.get("flowers_rotated_train").set(thing_classes=["Stem", "flower"], evaluator_type="coco")
    MetadataCatalog.get("flowers_rotated_val").set(thing_classes=["Stem", "flower"], evaluator_type="coco")


# ---------- 2) custom mapper for rotated boxes ----------
class SimpleRotatedMapper:
    """
    Minimal mapper: does basic image read and constructs Instances with RotatedBoxes and classes.
    Does NOT apply heavy augmentation. Suitable for Windows single-process training.
    """
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, dataset_dict):
        # dataset_dict is one dict from get_rotated_flower_dicts
        d = dataset_dict.copy()
        image = cv2.imread(d["file_name"])
        if image is None:
            raise FileNotFoundError(f"Image not found: {d['file_name']}")
        # convert BGR->RGB if your pipeline expects; DefaultPredictor uses BGR, keep consistent
        # no augmentation: simply create Instances
        annos = d.get("annotations", [])
        boxes = []
        classes = []
        masks = []  # optional if segmentation provided
        for ann in annos:
            boxes.append(ann["bbox"])  # [cx, cy, w, h, theta]
            classes.append(ann["category_id"])
            masks.append(ann.get("segmentation", []))

        # build Instances
        from detectron2.structures import RotatedBoxes
        from detectron2.structures import Instances
        h, w = image.shape[:2]
        inst = Instances((h, w))
        if len(boxes) > 0:
            inst.gt_boxes = RotatedBoxes(torch.tensor(boxes, dtype=torch.float32))
            inst.gt_classes = torch.tensor(classes, dtype=torch.int64)
        else:
            # empty Instances
            inst.gt_boxes = RotatedBoxes(torch.zeros((0,5), dtype=torch.float32))
            inst.gt_classes = torch.zeros((0,), dtype=torch.int64)

        # put things back
        d["image"] = image  # detectron2's loader expects "image" or actual image array? Default build expects image tensor later; but trainer uses mapper that returns dict with "image" as np array.
        d["instances"] = inst
        return d

# ---------- 3) Trainer override so we can use custom mapper ----------
class RotatedTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        def mapper_with_remap(d):
            for anno in d["annotations"]:
                if anno["category_id"] == 1:
                    anno["category_id"] = 0
                elif anno["category_id"] == 2:
                    anno["category_id"] = 1
            mapper = SimpleRotatedMapper(is_train=True)
            return mapper(d)
        # use our simple mapper; set num_workers=0 for Windows
        # return build_detection_train_loader(cfg, mapper=SimpleRotatedMapper(is_train=True))
        return build_detection_train_loader(cfg, mapper=mapper_with_remap)
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # use default evaluator if needed (skipped here)
        from detectron2.evaluation import COCOEvaluator
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

# ---------- 4) training entry ----------
def train():
    register_rotated_flowers()

    cfg = get_cfg()
    # use cascade mask rcNN misc config or mask rcNN; choose a config that supports ROI_BOX rotate settings
    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_TYPE = "RotatedBoxes"

    cfg.DATASETS.TRAIN = ("flowers_rotated_train",)
    cfg.DATASETS.TEST = ("flowers_rotated_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.PREFETCH_FACTOR = 0
    cfg.DATALOADER.PERSISTENT_WORKERS = False

    # model weights (you can use model_zoo checkpoint or your own pretrain)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 2.5e-4
    cfg.SOLVER.MAX_ITER = 1344
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = RotatedTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    import os
    import torch
    # Windows safe settings
    torch.multiprocessing.freeze_support()
    train()
