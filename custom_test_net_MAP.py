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
from detectron2.data import build_detection_test_loader


with open('params.yaml', 'r') as stream:
    param_data=yaml.safe_load(stream)

damage_name='dent'
MODE='LOCAL'

test_json='/share/test_total.json'
train_json='/share/train_total.json'
img_dir='/share/dent_testing/'

register_coco_instances(damage_name+"_test", {}, test_json, img_dir)
register_coco_instances(damage_name+"_train", {}, train_json, img_dir)
#register_coco_instances(damage_name+"_test", {}, test_json, img_dir)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(param_data['MODEL']['CONFIG']))
cfg.DATASETS.TEST = (damage_name+"_test",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(param_data['MODEL']['CONFIG'])
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN=12000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=6000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN=1800
cfg.MODEL.RPN.POST_NMS_TOPK_TEST=2000
cfg.MODEL.RPN.NMS_THRESH=0.73921
cfg.SOLVER.MOMENTUM= 0.58664
cfg.SOLVER.BASE_LR = 0.00273
cfg.MODEL.ANCHOR_GENERATOR.SIZES=[[ 32, 64, 128,256,512]]

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
predictor = DefaultPredictor(cfg)
cfg.MODEL.WEIGHTS = "/share/dent_model/model_0007499.pth"
predictor = DefaultPredictor(cfg)
val_loader = build_detection_test_loader(cfg, damage_name+"_test")
evaluator = COCOEvaluator(damage_name+"_test", cfg, False,output_dir=None)
results=inference_on_dataset(predictor.model, val_loader, evaluator)
map_val=results['bbox']['AP50']
print('Results: ')
print(results)
print('MAP:50 value: ',map_val)


