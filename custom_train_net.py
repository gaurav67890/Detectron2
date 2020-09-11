import argparse
import detectron2
from detectron2.utils.logger import setup_logger
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
import matplotlib.pyplot as plt
from tqdm import tqdm
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

ap = argparse.ArgumentParser()
ap.add_argument("-dn", "--damage_name", required=True, help="name of the damage")
args = vars(ap.parse_args())

damage_name=args['damage_name']

with open('params.yaml', 'r') as stream:
    param_data=yaml.safe_load(stream)
MODE='LOCAL'
dataset_dir=param_data['DATASET'][MODE]['DIR_PATH']
train_json=dataset_dir+damage_name+param_data['DATASET'][MODE]['TRAIN_PATH']
val_json=dataset_dir+damage_name+param_data['DATASET'][MODE]['VAL_PATH']
test_json=dataset_dir+damage_name+param_data['DATASET'][MODE]['TEST_PATH']

img_dir=dataset_dir+damage_name+param_data['DATASET'][MODE]['IMAGES_PATH']
register_coco_instances("damage_train", {}, train_json, img_dir)
register_coco_instances("damage_val", {}, val_json, img_dir)
register_coco_instances("damage_test", {}, test_json, img_dir)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(param_data['MODEL']['CONFIG']))
cfg.DATASETS.TRAIN = ("damage_train",)
cfg.DATASETS.TEST = ("damage_val",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(param_data['MODEL']['CONFIG'])  # Let training initialize from mode$
cfg.SOLVER.CHECKPOINT_PERIOD=1000
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 30000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
