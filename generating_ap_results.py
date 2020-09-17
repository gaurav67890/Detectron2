import yaml
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import csv
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

damage_name='dent'
#MODE='LOCAL'

#dataset_dir=param_data['DATASET'][MODE]['DIR_PATH']
test_json='/share/test_total.json'
train_json='/share/train_total.json'
img_dir='/share/dent_testing/'
register_coco_instances(damage_name+"_test", {}, test_json, img_dir)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(param_data['MODEL']['CONFIG']))
cfg.DATASETS.TEST = (damage_name+"_test",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(param_data['MODEL']['CONFIG'])
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN=12000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=6000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN=1800
cfg.MODEL.RPN.POST_NMS_TOPK_TEST=2000
cfg.MODEL.RPN.NMS_THRESH=0.73921
cfg.SOLVER.MOMENTUM= 0.58664
cfg.SOLVER.BASE_LR = 0.00273
cfg.MODEL.ANCHOR_GENERATOR.SIZES=[[ 32, 64, 128,256,512]]

cfg.MODEL.WEIGHTS = "/share/dent_model/model_0007499.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set a custom testing threshold for this model


from detectron2.data import build_detection_test_loader
predictor = DefaultPredictor(cfg)

my_dataset_train_metadata = MetadataCatalog.get(damage_name+"_train")

with open(test_json) as f:
        data = json.load(f)
        
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

category_dict={}
for i in data['categories']:
    category_dict[i['id']]=i['name']

tp=0
fp=0
fn=0
confusion_matrix={}
cf_dict_per_image={}
data_out=[]
area_final=None
for cat in category_dict.keys():
    cf_dict_per_image[category_dict[cat]]={}
    for i in tqdm(range(len(data['images']))):
        try:
            #org_bbox_id=0
            h=data['images'][i]['height']
            w=data['images'][i]['width']

            org_bbox_dict={}

            for j in range(len(data['annotations'])):
                if data['annotations'][j]['image_id']==data['images'][i]['id']:
                    if data['annotations'][j]['category_id']!=cat:
                        continue
                    bbox_org=data['annotations'][j]['bbox']
                    r_org=Rectangle(bbox_org[0], bbox_org[1],bbox_org[2]+bbox_org[0] , bbox_org[3]+bbox_org[1])
                    org_bbox_id=data['annotations'][j]['id']
                    org_bbox_dict[org_bbox_id]=r_org
                    #org_bbox_id=org_bbox_id+1
            
            #if len(org_bbox_dict)==0:
            #      continue
            fn=data['images'][i]['file_name']
            img=cv2.imread(img_dir+fn)
            outputs = predictor(img)
            classes_pred=outputs["instances"].pred_classes.tolist()
            bbox_pred=outputs["instances"].pred_boxes
            bbox_pred=getattr(bbox_pred, 'tensor').tolist()

            print(fn)
            tp_temp=0
            fp_temp=0
            fn_temp=0
            area_list=[]
            bbox_detected=[]
            if len(org_bbox_dict.keys())==0:
                tp_temp=0
                fp_temp=len(classes_pred[classes_pred==cat])
                fn_temp=0
            else:
                for org_keys in org_bbox_dict.keys():
                    annt_dict={}
                    r_org=org_bbox_dict[org_keys]
                    area_overlaped=0
                    for cl,k in zip(classes_pred,bbox_pred):
                        #print(cl,cat)
                        if cl != cat:
                            area_final=None
                            continue
                  
                        r_pred=Rectangle(k[0],k[1],k[2],k[3])
                        area=rect_area(r_org,r_pred)
                        area_final=max(area,area_overlaped)
                    if area_final is None:
                        print('none')
                        continue
                    if area_final>0.5:
                        tp_temp=tp_temp+1
                        bbox_detected.append(org_keys)
                        annt_dict['annotation']=org_keys
                        annt_dict['image_name']='/share/results_entire_dent/'+fn
                        annt_dict['detected']=1
                    else:
                        fp_temp=fp_temp+1
                        annt_dict['annotation']=org_keys
                        annt_dict['image_name']='/share/results_entire_dent/'+fn
                        annt_dict['detected']=0
                    data_out.append(annt_dict)
            fp=fp+fp_temp
            tp=tp+tp_temp


        except Exception as e:
            print(e)
        
    confusion_matrix[category_dict[cat]]={'true_positve':tp,'false_positive':fp}


with open(damage_name+'_confusion_matrix.json', 'w') as outfile:
        json.dump(confusion_matrix,outfile,indent=4,ensure_ascii = False)
csv_file = "_data_out.csv"
csv_columns = ['annotation','image_name','detected']
try:
    with open(damage_name+csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in data_out:
            writer.writerow(data)
except IOError:
    print("I/O error")
