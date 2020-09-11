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



with open('params.yaml', 'r') as stream:
    param_data=yaml.safe_load(stream)

damage_name='merged_scratch'
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
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
# cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN=12000
# cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=12000
# cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN=2200
# cfg.MODEL.RPN.POST_NMS_TOPK_TEST=2200
# cfg.MODEL.RPN.NMS_THRESH=0.6
# cfg.SOLVER.MOMENTUM= 0.95
cfg.SOLVER.BASE_LR = 0.0025
#cfg.MODEL.ANCHOR_GENERATOR.SIZES=[[8,16, 32, 64, 128]]

cfg.MODEL.WEIGHTS = "output/model_0010999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set a custom testing threshold for this model



def get_indexes(test_list):
    unique_values=list(set(test_list))
    res_dict = {}
    for i in range(0, len(test_list)) : 
        for j in unique_values:
            if j not in res_dict.keys():
                res_dict[j]=[]
            if test_list[i] == j: 
                res_dict[j].append(i)
                
    return res_dict

from detectron2.data import build_detection_test_loader
predictor = DefaultPredictor(cfg)

with open(test_json) as f:
        data = json.load(f)
dice={}        
org_classes_ids=[]
for i in range(len(data['categories'])):
    class_id=data['categories'][i]['id']
    org_classes_ids.append(class_id)
    dice[class_id]=[]

for i in tqdm(range(len(data['images']))):
    try:
        h=data['images'][i]['height']
        w=data['images'][i]['width']
        org_mask={}
        for org_class_id in org_classes_ids:
            mask=np.zeros((h,w),dtype='uint8')
            for j in range(len(data['annotations'])):
                if data['annotations'][j]['image_id']==data['images'][i]['id']:
                    if data['annotations'][j]['category_id']==org_class_id:
                        p1=data['annotations'][j]['segmentation'][0]
                        p1=[int(i) for i in p1]
                        p2=[]
                        for p in range(int(len(p1)/2)):
                            p2.append([p1[2*p],p1[2*p+1]])
                        fill_pts = np.array([p2], np.int32)
                        cv2.fillPoly(mask, fill_pts, 1)
            org_mask[org_class_id]=mask

        img=cv2.imread(img_dir+data['images'][i]['file_name'])
        out = predictor(img)
        pred_classes=out['instances'].pred_classes.tolist()
        pred_mask=out['instances'].pred_masks.cpu().numpy()
        classes_index=get_indexes(pred_classes)
            
        for cls_idx in classes_index.keys():
            
            pred=np.take(pred_mask, classes_index[cls_idx], 0)
            pred=np.sum(pred,axis=0)
            pred[pred>1]=1
            groundmask=org_mask[cls_idx]
            intersection = np.logical_and(groundmask, pred)
            if np.sum(pred)>0:
                if np.sum(groundmask)>0:
                    ground=np.unique(groundmask,return_counts=True)[1][1]
                    pred_val=np.unique(pred,return_counts=True)[1][1]
                    dice_score = 2*np.sum(intersection) / (ground+pred_val)
                else:
                    dice_score=0
                
            else:
                if np.sum(groundmask)>0:
                    dice_score=0
                else:
                    continue
            dice[cls_idx].append(dice_score)
            print(dice)
                
    except Exception as e:
        print(e)
        
final_dice=0
for i in org_classes_ids:
    cls_dice=sum(dice[i])/len(dice[i])
    final_dice=final_dice+cls_dice
    print('Dice Coeff for id '+str(i)+':'+str(cls_dice))
print('Dice Coeff Mean: '+str(final_dice/len(org_classes_ids)))

