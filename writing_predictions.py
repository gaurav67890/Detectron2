import detectron2
import yaml
from tqdm import tqdm
from detectron2.utils.logger import setup_logger
setup_logger()
try:
    import Image
except ImportError:
    from PIL import Image
import PIL
import sys
Image.MAX_IMAGE_PIXELS = 933120000
import urllib
# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from tqdm import tqdm
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import coco
import torch
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances


damage_name='dent_ding'
MODE='LOCAL'

dataset_dir=param_data['DATASET'][MODE]['DIR_PATH']
test_json=dataset_dir+damage_name+param_data['DATASET'][MODE]['VAL_PATH']
train_json=dataset_dir+damage_name+param_data['DATASET'][MODE]['TRAIN_PATH']

img_dir=dataset_dir+damage_name+param_data['DATASET'][MODE]['IMAGES_PATH']

register_coco_instances(damage_name+"_test", {}, test_json, img_dir)
register_coco_instances(damage_name+"_train", {}, train_json, img_dir)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TEST = (damage_name+"_test",)
cfg.DATASETS.TRAIN = (damage_name+"_train",)

cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from mode$
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
#cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN=18000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=18000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN=2000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST=2000
cfg.MODEL.RPN.NMS_THRESH=0.7
cfg.SOLVER.MOMENTUM= 0.675
cfg.SOLVER.BASE_LR = 0.00213
cfg.MODEL.ANCHOR_GENERATOR.SIZES=[[16, 32, 64, 128,256]]

cfg.MODEL.WEIGHTS = "/share/dent_model/model_0007499.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set a custom testing threshold for this model

from detectron2.data import build_detection_test_loader
predictor = DefaultPredictor(cfg)




my_dataset_test_metadata = MetadataCatalog.get(damage_name+"_test")
dataset_dicts = DatasetCatalog.get(damage_name+"_test")

for d in tqdm(range(len(dataset_dicts))):
    img = cv2.imread(dataset_dicts[d]["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_test_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(dataset_dicts[d])
    original=vis.get_image()[:, :, ::-1]
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                metadata=my_dataset_test_metadata, 
                scale=0.5
                 )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    predicted=out.get_image()[:, :, ::-1]
    horizontalAppendedImg = np.hstack((original,predicted))
    fn=dataset_dicts[d]["file_name"]
    fn=fn[fn.rfind('/')+1:]
    cv2.imwrite('/share/predictions/'+fn,horizontalAppendedImg)
