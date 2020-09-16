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

with open('params.yaml', 'r') as stream:
    param_data=yaml.safe_load(stream)

damage_name='scratch_2'
MODE='LOCAL'

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
#cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN=12000
#cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=12000
#cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN=2200
#cfg.MODEL.RPN.POST_NMS_TOPK_TEST=2200
#cfg.MODEL.RPN.NMS_THRESH=0.6
#cfg.SOLVER.MOMENTUM= 0.95
cfg.SOLVER.BASE_LR = 0.0025
#cfg.MODEL.ANCHOR_GENERATOR.SIZES=[[8,16, 32, 64, 128]]

cfg.MODEL.WEIGHTS = "./output/model_0033999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3


from detectron2.data import build_detection_test_loader
predictor = DefaultPredictor(cfg)

with open(test_json) as f:
        data = json.load(f)
dice=[]
l=0
for i in tqdm(range(len(data['images']))):
    try:
        h=data['images'][i]['height']
        w=data['images'][i]['width']
        mask=np.zeros((h,w),dtype='uint8')
        for j in range(len(data['annotations'])):
            if data['annotations'][j]['image_id']==data['images'][i]['id']:
                p1=data['annotations'][j]['segmentation'][0]
                p1=[int(i) for i in p1]
                p2=[]
                for p in range(int(len(p1)/2)):
                    p2.append([p1[2*p],p1[2*p+1]])
                fill_pts = np.array([p2], np.int32)
                cv2.fillPoly(mask, fill_pts, 1)
        if np.unique(mask,return_counts=True)[1][1]/(w*h)>0.000:
            img=cv2.imread(img_dir+data['images'][i]['file_name'])
            out = predictor(img)
            pred = torch.sum(out['instances'].pred_masks,dim=0) > 0
            pred = pred.cpu().detach().numpy()
            pred=pred.astype(int)
            intersection = np.logical_and(mask, pred)
            if len(np.unique(pred,return_counts=True)[1])>1:
                ground=np.unique(mask,return_counts=True)[1][1]
                pred_val=np.unique(pred,return_counts=True)[1][1]
                dice_score = 2*np.sum(intersection) / (ground+pred_val)
            else:
                dice_score=0
            dice.append(dice_score)
    except Exception as e:
        print(e)
final_dice=sum(dice)/len(dice)
print('Dice Coeff: '+str(final_dice))

