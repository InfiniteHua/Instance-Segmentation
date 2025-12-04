import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# =========================
# 配置模型（和训练一致）
# =========================
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # 训练好的权重
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 类别数

predictor = DefaultPredictor(cfg)

# =========================
# 注册数据集的metadata，用于可视化
# =========================
from detectron2.data.datasets import register_coco_instances
register_coco_instances(
    "flowers_val", {}, 
    "flowers/valid/_annotations.coco.json", 
    "flowers/valid"
)
flowers_metadata = MetadataCatalog.get("flowers_val")

# =========================
# 指定要测试的图片
# =========================
img_path = "flowers/temp/11_jpg.rf.e0f41ea72c62db6f5b30d5ad1c07bfc6.jpg"  # 改成你的图片路径
im = cv2.imread(img_path)
outputs = predictor(im)

# =========================
# 可视化
# =========================
v = Visualizer(
    im[:, :, ::-1], 
    metadata=flowers_metadata, 
    scale=0.5, 
    instance_mode=ColorMode.IMAGE_BW
)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
