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


damage_name='crack'

test_json="/share/datasets/coco/"+damage_name+"/annotations/instances_test.json"
img_dir="/share/datasets/coco/images/"
register_coco_instances(damage_name+"_test", {}, test_json, img_dir)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TEST = (damage_name+"_test",)

cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from mode$
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
#cfg.TEST.EVAL_PERIOD = 500
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
            #l=l+1
            #cv2.imwrite(data['images'][i]['file_name'],mask)
            img=cv2.imread(img_dir+data['images'][i]['file_name'])
            #cv2.imwrite('im/original'+str(i)+'.png',img)
            #cv2.imwrite('im/mask'+str(i)+'.png',mask*255)
            out = predictor(img)
            pred = torch.sum(out['instances'].pred_masks,dim=0) > 0
            pred = pred.cpu().detach().numpy()
            pred=pred.astype(int)
            #cv2.imwrite('im/pred'+str(i)+'.png',pred*255)
            intersection = np.logical_and(mask, pred)
            if len(np.unique(pred,return_counts=True)[1])>1:
                ground=np.unique(mask,return_counts=True)[1][1]
                pred_val=np.unique(pred,return_counts=True)[1][1]
                dice_score = 2*np.sum(intersection) / (ground+pred_val)
            #print(dice_score)
            else:
                dice_score=0
            print(dice_score)
            #print(dice_score)
            #print(dice)
            dice.append(dice_score)
    except Exception as e:
        print(e)
final_dice=sum(dice)/len(dice)
print('Dice Coeff: '+str(final_dice))

