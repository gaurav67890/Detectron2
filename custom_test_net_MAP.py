import detectron2
from detectron2.utils.logger import setup_logger
import json
setup_logger()
try:
    import Image
except ImportError:
    from PIL import Image
import PIL
Image.MAX_IMAGE_PIXELS = 933120000
import urllib
# import some common libraries
import numpy as np
import os, json, cv2, random
import yaml
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
from detectron2.evaluation import inference_on_dataset

with open('params.yaml', 'r') as stream:
    param_data=yaml.safe_load(stream)

damage_name='crack'
mode='LOCAL'

dataset_dir=param_data['DATASET'][MODE]['DIR_PATH']
test_json=dataset_dir+damage_name+param_data['DATASET'][MODE]['TEST_PATH']
img_dir=dataset_dir+damage_name+param_data['DATASET'][MODE]['IMAGES_PATH']
register_coco_instances(damage_name+"_test", {}, test_json, img_dir)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(param_data['MODEL']['CONFIG']))
cfg.DATASETS.TEST = (damage_name+"_test",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(param_data['MODEL']['CONFIG'])
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN=12000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=12000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN=2200
cfg.MODEL.RPN.POST_NMS_TOPK_TEST=2200
cfg.MODEL.RPN.NMS_THRESH=0.6
cfg.SOLVER.MOMENTUM= 0.95
cfg.SOLVER.BASE_LR = 0.001
cfg.MODEL.ANCHOR_GENERATOR.SIZES=[[8,16, 32, 64, 128]]

cfg.MODEL.WEIGHTS = "/share/crack_model/model_0002399.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set a custom testing threshold for this model


from detectron2.data import build_detection_test_loader
predictor = DefaultPredictor(cfg)

map_dict={}
predictor = DefaultPredictor(cfg)
val_loader = build_detection_test_loader(cfg, damage_name+"_test")
evaluator = COCOEvaluator(damage_name+"_test", cfg, False,output_dir=None)
results=inference_on_dataset(predictor.model, val_loader, evaluator)
map_val=results['segm']['AP50']
print('MAP value: ',map_val)
map_dict[i]=map_val

