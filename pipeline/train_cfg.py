import torch, torchvision
import mmdet
import mmcv
import mmengine

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

from pycocotools.coco import COCO

from mmengine import Config

from mmengine.runner import set_random_seed

from mmengine.config import Config
from mmdet.registry import DATASETS   # 注意是 mmdet.registry

print("torch version:",torch.__version__, "cuda:",torch.cuda.is_available())
print("mmdetection:",mmdet.__version__)
print("mmcv:",mmcv.__version__)
print("mmengine:",mmengine.__version__)


###################################################################
annotation = mmengine.load('./Stem_Segmentation/flowers/train/_annotations_coco_1.json')

##################################################################
# Path to load the COCO annotation file
annotation_file = './Stem_Segmentation/flowers/train/_annotations_coco_1.json'
# Initialise the COCO object
coco = COCO(annotation_file)
# Get all category tags and corresponding category IDs
categories = coco.loadCats(coco.getCatIds())
category_id_to_name = {cat['id']: cat['name'] for cat in categories}
# Print all category IDs and corresponding category names
for category_id, category_name in category_id_to_name.items():
    print(f"Category ID: {category_id}, Category Name: {category_name}")

################################################################
# cfg = Config.fromfile('../configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py')
cfg = Config.fromfile('../configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py')

cfg.metainfo = {
    'classes': ('Stem','flower',),
    'palette': [(0, 170, 255), (255, 165, 0)],   
}

# Modify dataset type and path
cfg.data_root = './Stem_Segmentation/flowers'

cfg.train_dataloader.dataset.ann_file = 'train/_annotations_coco_1.json'
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = 'train/'
cfg.train_dataloader.dataset.metainfo = cfg.metainfo

cfg.val_dataloader.dataset.ann_file = 'valid/_annotations_coco_1.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = 'valid/'
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

cfg.val_dataloader.dataset.pipeline[0].type = 'LoadImageFromFile'

cfg.test_dataloader = cfg.val_dataloader

# Modify metric config
cfg.val_evaluator.ann_file = cfg.data_root+'/'+'valid/_annotations_coco_1.json'
cfg.test_evaluator = cfg.val_evaluator

# Modify num classes of the model in box head and mask head
# cfg.model.roi_head.bbox_head.num_classes = 2
cfg.model.panoptic_head.num_things_classes = 2
cfg.model.panoptic_head.num_stuff_classes = 0

# We can still the pre-trained Mask RCNN model to obtain a higher performance
# cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './m2f'


# We can set the evaluation interval to reduce the evaluation times
#cfg.train_cfg.val_interval = 3
### This is for small dataset ###
# cfg.train_cfg.val_interval = 1
# We can set the checkpoint saving interval to reduce the storage cost
cfg.default_hooks.checkpoint.interval = 1
cfg.train_cfg.val_interval = 1

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optim_wrapper.optimizer.lr = 0.02 / 8
cfg.default_hooks.logger.interval = 10


# Set seed thus the results are more reproducible
# cfg.seed = 0
set_random_seed(0, deterministic=False)

# We can also use tensorboard to log the training process
#cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend') 
    ])

#------------------------------------------------------
config=f'../stem_configs/mask2former_r50_8xb2-lsj-50e_coco_test_1.py'
with open(config, 'w') as f:
    f.write(cfg.pretty_text)

##################################################################
# config=f'../stem_configs/mask-rcnn_r50-caffe_fpn_ms-poly-3x_Stem_tinytest.py'
