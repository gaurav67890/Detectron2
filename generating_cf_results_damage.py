import yaml
import csv
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
from detectron2.data import build_detection_test_loader

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
    
def draw_poly(pts,image):
    pts = np.array(pts, 
               np.int32) 
    pts = pts.reshape((-1, 1, 2)) 
    isClosed = True
    # Blue color in BGR 
    color = (255, 0, 0) 
  
    # Line thickness of 2 px 
    thickness = 1
    image = cv2.polylines(image, [pts],  
                      isClosed, color, thickness) 
    return image


with open('params.yaml', 'r') as stream:
    param_data=yaml.safe_load(stream)

damage_name='dent'
MODE='LOCAL'

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
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   


predictor = DefaultPredictor(cfg)

my_dataset_train_metadata = MetadataCatalog.get(damage_name+"_train")

with open(test_json) as f:
        data = json.load(f)
        
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

category_dict={}
for i in data['categories']:
    category_dict[i['id']]=i['name']

thresh=0.25
thresh_string='_'+str(int(100*thresh))+'_'

confusion_matrix={}
data_out=[]
IOU_25=0
IOU_50=0

for cat in category_dict.keys():
    for i in tqdm(range(len(data['images']))):
        if 1>0:
            annt_dict={}
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
                    org_bbox_id=org_bbox_id+1

            file_name=data['images'][i]['file_name']
            img=cv2.imread(img_dir+file_name)
            outputs = predictor(img)
            print(outputs)
            scores_pred=outputs["instances"].scores.tolist()
            classes_pred=outputs["instances"].pred_classes.tolist()
            bbox_pred=outputs["instances"].pred_boxes
            bbox_pred=getattr(bbox_pred, 'tensor').tolist()
            segm_pred=outputs["instances"].pred_masks
            segm_pred=segm_pred.cpu().numpy()
            print(segm_pred)


            area_list=[]
            bbox_detected=[]
            if len(org_bbox_dict.keys())==0:
                fp_temp=len(classes_pred[classes_pred==cat])
                for cl,k,cf,sgm in zip(classes_pred,bbox_pred,scores_pred,segm_pred):
                    print('mask_pred')
                    print(mask_pred)
                    annt_dict['annotation_id']=-1
                    annt_dict['image_name']='/share/results_entire_dent/'+file_name
                    annt_dict['status']='FP'
                    annt_dict['IOU25']=0
                    annt_dict['score']=cf
                    
            else:
                
                for cl,k,cf,sgm in zip(classes_pred,bbox_pred,scores_pred,segm_pred):
                    if cl != cat:
                        print('c')
                        continue
                    mask_pred=np.transpose(np.nonzero(sgm == True))
                    print('mask_pred')
                    print(mask_pred)
                    image_new=draw_poly(mask_pred,img)
                    cv2.imwrite('poly/'+file_name,image_new)
                    area_dict={}
                    r_pred=Rectangle(k[0],k[1],k[2],k[3])
                    for org_keys in org_bbox_dict.keys():
                        r_org=org_bbox_dict[org_keys]
                        print(r_org,r_pred)
                        area=rect_area(r_org,r_pred)
                        print(area)
                        area_dict[org_keys]=area
                    if max(area_dict.values())>0.25:
                        status='TP'
                        IOU_25=1
                        ann_detected=max(area_dict, key=area_dict.get)
                        bbox_detected.append(ann_detected)
                        if max(area_dict.values())>0.5:
                            IOU_50=1
                        else:
                            IOU_50=0
                        
                    else:
                        status='FP'
                        IOU_25=0
                        IOU_50=0
                        ann_detected=-1
                    annt_dict['annotation_id']=ann_detected
                    annt_dict['image_name']='/share/results_entire_dent/'+file_name
                    annt_dict['status']=status
                    annt_dict['IOU25']=IOU_25
                    annt_dict['IOU50']=IOU_50
                    annt_dict['score']=cf
                    print('a')
                    print(annt_dict)
            for ann in bbox_detected:
                if ann in org_bbox_dict.keys():
                    continue
                annt_dict['annotation_id']=ann
                annt_dict['image_name']='/share/results_entire_dent/'+file_name
                annt_dict['status']='FN'
                annt_dict['IOU25']=IOU_25
                annt_dict['IOU50']=IOU_50
                annt_dict['score']=-1
            
            data_out.append(annt_dict)
        #except Exception as e:
        #    print(e)
        



csv_file = thresh_string+"_data_out.csv"
csv_columns = ['annotation_id','image_name','status','IOU25','IOU50','score']
try:
    with open(damage_name+csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in data_out:
            writer.writerow(data)
except IOError:
    print("I/O error")