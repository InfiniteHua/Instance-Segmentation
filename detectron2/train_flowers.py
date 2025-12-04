import os
import cv2
import random
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

import torch
import detectron2
import subprocess

from detectron2.data import build_detection_train_loader
from detectron2.data import DatasetMapper


# print detectron2 version
print("detectron2:", detectron2.__version__)

def register_dataset():
    register_coco_instances(
        "flowers_train", {},
        "flowers/train/_annotations.coco.json",
        "flowers/train"
    )

    register_coco_instances(
        "flowers_val", {},
        "flowers/valid/_annotations.coco.json",
        "flowers/valid"
    )

def train():
    register_dataset()

    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file(
    #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    # ))
    cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # cfg.merge_from_file("./configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
    # cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
    # cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0, 1.0)
    # cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    # cfg.MODEL.ROI_BOX_HEAD.BBOX_TYPE = "RotatedBoxes"
    dataset_dicts = DatasetCatalog.get("flowers_train")
    metadata = MetadataCatalog.get("flowers_train")

    cfg.DATASETS.TRAIN = ("flowers_train",)
    cfg.DATASETS.TEST = ("flowers_val",)

    cfg.DATALOADER.NUM_WORKERS = 2   # 必须为 0（Windows）

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
    # )
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # flower dataset 有 1 类

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    # trainer.data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=False))
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    # print nvcc version
    try:
        output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)
        print(output.decode())
    except Exception as e:
        print("nvcc not found:", e)

    # print torch and cuda version
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch:", TORCH_VERSION, "; cuda:", CUDA_VERSION)

    train()
