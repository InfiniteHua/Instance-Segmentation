import torch
import detectron2
import subprocess

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

# print detectron2 version
print("detectron2:", detectron2.__version__)

########################## This is for simple prediction ################################

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import requests
import urllib.request
import zipfile

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# url = "http://images.cocodataset.org/val2017/000000439715.jpg"
# img_data = requests.get(url).content
# with open("input.jpg", "wb") as f:
#     f.write(img_data)

# 读取和显示
im = cv2.imread("input.jpg")
# cv2.imshow("image", im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)

# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow("result", out.get_image()[:, :, ::-1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

############################### Load the dataset and train a new model #######################################

# url = "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip"
# zip_path = "balloon_dataset.zip"

# if not os.path.exists(zip_path):
#     print("Downloading balloon_dataset.zip ...")
#     urllib.request.urlretrieve(url, zip_path)
# else:
#     print("balloon_dataset.zip already exists, skip downloading.")

# # 解压缩
# extract_dir = "balloon"
# if not os.path.exists(extract_dir):
#     print("Extracting balloon_dataset.zip ...")
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(".")
# else:
#     print("Dataset already extracted.")

from detectron2.data.datasets import register_coco_instances

register_coco_instances("flowers_train", {}, "flowers/train/_annotations_rotated.coco.json", "flowers/train")
register_coco_instances("flowers_val", {}, "flowers/valid/_annotations_rotated.coco.json", "flowers/valid")

dataset_dicts = DatasetCatalog.get("flowers_train")
metadata = MetadataCatalog.get("flowers_train")

# visualize randomly 3 pics from dataset
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("result", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

# # training
# from detectron2.engine import DefaultTrainer

# cfg.DATASETS.TRAIN = ("flowers_train",)
# cfg.DATASETS.TEST = ("flowers_val",)
# cfg.DATALOADER.NUM_WORKERS = 2

# # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
# #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# # )

# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025
# cfg.SOLVER.MAX_ITER = 300
# cfg.SOLVER.STEPS = []

# # ✔ COCO metadata自动包含类别数量
# metadata = MetadataCatalog.get("flowers_train")
# num_classes = len(metadata.thing_classes)

# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
# # 必须是“类别数量”，不是“类别+1”

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()