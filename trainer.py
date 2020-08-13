# Some basic setup:
# Setup detectron2 logger
import argparse
import hypertune
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

parser = argparse.ArgumentParser(description='Input parameters need to be Specified for hypertuning')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate parameter')
args = parser.parse_args()
lr = args.lr

train_json="/home/appuser/detectron2_repo/datasets/coco/scratch/annotations/instances_train.json"
val_json="/home/appuser/detectron2_repo/datasets/coco/scratch/annotations/instances_validation.json"
test_json="/home/appuser/detectron2_repo/datasets/coco/scratch/annotations/instances_test.json"

img_dir="/home/appuser/detectron2_repo/datasets/coco/images/"
register_coco_instances("scratch_train", {}, train_json, img_dir)
register_coco_instances("scratch_val", {}, val_json, img_dir)
register_coco_instances("scratch_test", {}, val_json, img_dir)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("scratch_train",)
cfg.DATASETS.TEST = ("scratch_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from mode$
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = lr  # pick a good LR
cfg.SOLVER.MAX_ITER = 4000 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (scratch)
cfg.TEST.EVAL_PERIOD = 500


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = CocoTrainer(cfg) 
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model


from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("scratch_test", cfg, False,output_dir="./output/")
predictor = DefaultPredictor(cfg)
val_loader = build_detection_test_loader(cfg, "scratch_test")
#print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way is to use trainer.test


with open(test_json) as f:
        data = json.load(f)
dice=0
l=0
for i in tqdm(range(len(data['images']))):
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
        l=l+1
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
            #print(dice_score)
        dice=dice+dice_score
final_dice=dice/l
print('Dice Coeff: '+str(final_dice))



hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='dice', metric_value=final_dice, global_step=epochs)
