import yaml
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
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
from collections import namedtuple


def rect_area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    area1=(a.xmax-a.xmin)*(a.ymax-a.ymin)
    area2=(b.xmax-b.xmin)*(b.ymax-b.ymin)
    if (dx>=0) and (dy>=0):
        area_intersection=dx*dy
        area_union=area1+area2
        area_ratio=2*area_intersection/area_union
        return area_ratio
    else:
        return 0
    
with open('params.yaml', 'r') as stream:
    param_data=yaml.safe_load(stream)

damage_name='dent_ding'
MODE='LOCAL'

dataset_dir=param_data['DATASET'][MODE]['DIR_PATH']
test_json=dataset_dir+damage_name+param_data['DATASET'][MODE]['VAL_PATH']
train_json=dataset_dir+damage_name+param_data['DATASET'][MODE]['TRAIN_PATH']
img_dir=dataset_dir+damage_name+param_data['DATASET'][MODE]['IMAGES_PATH']
register_coco_instances(damage_name+"_test", {}, test_json, img_dir)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(param_data['MODEL']['CONFIG']))
cfg.DATASETS.TEST = (damage_name+"_test",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(param_data['MODEL']['CONFIG'])
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
# cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN=12000
# cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=12000
# cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN=2200
# cfg.MODEL.RPN.POST_NMS_TOPK_TEST=2200
# cfg.MODEL.RPN.NMS_THRESH=0.6
# cfg.SOLVER.MOMENTUM= 0.95
cfg.SOLVER.BASE_LR = 0.0025
#cfg.MODEL.ANCHOR_GENERATOR.SIZES=[[8,16, 32, 64, 128]]

cfg.MODEL.WEIGHTS = "output/model_0040999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set a custom testing threshold for this model


from detectron2.data import build_detection_test_loader
predictor = DefaultPredictor(cfg)

my_dataset_train_metadata = MetadataCatalog.get(damage_name+"_train")

with open(test_json) as f:
        data = json.load(f)
        
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

tp=0
fp=0

cf_dict_per_image={}
for i in tqdm(range(len(data['images']))):
    try:
        org_bbox_id=0
        h=data['images'][i]['height']
        w=data['images'][i]['width']
        
        org_bbox_dict={}
        
        for j in range(len(data['annotations'])):
            if data['annotations'][j]['image_id']==data['images'][i]['id']:
                bbox_org=data['annotations'][j]['bbox']
                r_org=Rectangle(bbox_org[0], bbox_org[1],bbox_org[2]+bbox_org[0] , bbox_org[3]+bbox_org[1])
                org_bbox_dict[org_bbox_id]=r_org
                org_bbox_id=org_bbox_id+1
                
        img=cv2.imread(img_dir+data['images'][i]['file_name'])
        outputs = predictor(img)
        bbox_pred=outputs["instances"].pred_boxes
        bbox_pred=getattr(bbox_pred, 'tensor').tolist()

        tp_temp=0
        fp_temp=0
        area_list=[]
        for k in bbox_pred:
            #print('start')
            area_overlaped=0
            r_pred=Rectangle(k[0],k[1],k[2],k[3])
            for org_keys in org_bbox_dict.keys():
                r_org=org_bbox_dict[org_keys]
                print(r_org,r_pred)
                area=rect_area(r_org,r_pred)
                print(area)
                area_final=max(area,area_overlaped)
            area_list.append(area_final)
            if area_final>0.5:
                tp=tp+1
                tp_temp=tp_temp+1
            else:
                fp=fp+1
                fp_temp=fp_temp+1
        cf_dict_per_image[data['images'][i]['file_name']]={'tp':tp_temp,'fp':fp_temp,'area':area_list}
        
    except Exception as e:
        print(e)
        
confusion_matrix={'true_positve':tp,'false_positive':fp}

with open('cf_dict_per_image.json', 'w') as outfile:
        json.dump(cf_dict_per_image,outfile,indent=4,ensure_ascii = False)
with open('confusion_matrix.json', 'w') as outfile:
        json.dump(confusion_matrix,outfile,indent=4,ensure_ascii = False)