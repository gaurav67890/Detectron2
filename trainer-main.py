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
import json
Image.MAX_IMAGE_PIXELS = 933120000
import urllib
# import some common libraries
import numpy as np
import os, json, cv2, random
import glob,shutil
print(os.system('ls'))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/etc/credentials.json"
os.system('gsutil cp gs://hptuning2/split_damages.zip .')
os.system('unzip split_damages.zip')
from detectron2.data import build_detection_test_loader
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
from google.cloud import storage

parser = argparse.ArgumentParser(description='Input parameters need to be Specified for hypertuning')
parser.add_argument(
    '--damage_name',  # Handled automatically by AI Platform
    help='scratch,dent,crack etc',
    required=True
    )
parser.add_argument('--max_iter',  # Specified in the config file
    type=int,
    default=6000,
    help='maximum iteration')
parser.add_argument('--check_period',  # Specified in the config file
    type=int,
    default=500,
    help='checkpoint period')
parser.add_argument('--thresh_test',  # Specified in the config file
    type=float,
    default=0.4,
    help='testing threshold')
parser.add_argument(
    '--job-dir',  # Handled automatically by AI Platform
    help='GCS location to write checkpoints and export models',
    required=True
    )
parser.add_argument(
    '--lr', default=0.00025, 
    type=float, 
    help='Learning rate parameter')
parser.add_argument('--MOMENTUM',  # Specified in the config file
    type=float,
    default=0.5,
    help='SGD momentum (default: 0.5)')
parser.add_argument('--ANCHOR_SIZES',  # Specified in the config file
    type=int,
    default=3,
    help='ANCHOR_SIZES (default: 3)')
parser.add_argument('--PRE_NMS_TOPK_TRAIN',  # Specified in the config file
    type=int,
    default=12000,
    help='PRE_NMS_TOPK_TRAIN (default: 12000)')
parser.add_argument('--PRE_NMS_TOPK_TEST',  # Specified in the config file
    type=int,
    default=12000,
    help='PRE_NMS_TOPK_TEST (default: 6000)')
parser.add_argument('--POST_NMS_TOPK_TRAIN',  # Specified in the config file
    type=int,
    default=2000,
    help='POST_NMS_TOPK_TRAIN (default: 2000)')
parser.add_argument('--POST_NMS_TOPK_TEST',  # Specified in the config file
    type=int,
    default=1000,
    help='POST_NMS_TOPK_TEST (default: 1000)')

parser.add_argument('--NMS_THRESH',  # Specified in the config file
    type=float,
    default=0.7,
    help='NMS_THRESH (default: 0.7)')

if os.path.exists('output') and os.path.isdir('output'):
    shutil.rmtree('output')

args = parser.parse_args()

damage_name=args.damage_name

train_json="/detectron2_repo/split_damages/datasets/coco/"+damage_name+"/annotations/instances_train.json"
val_json="/detectron2_repo/split_damages/datasets/coco/"+damage_name+"/annotations/instances_validation.json"
test_json="/detectron2_repo/split_damages/datasets/coco/"+damage_name+"/annotations/instances_test.json"

img_dir="/detectron2_repo/split_damages/datasets/coco/images/"
register_coco_instances(damage_name+"_train", {}, train_json, img_dir)
register_coco_instances(damage_name+"_val", {}, val_json, img_dir)
register_coco_instances(damage_name+"_test", {}, test_json, img_dir)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (damage_name+"_train",)
cfg.DATASETS.TEST = (damage_name+"_val",)
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from mode$
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.MAX_ITER = args.max_iter 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (dent)
cfg.SOLVER.CHECKPOINT_PERIOD = args.check_period
cfg.SOLVER.MOMENTUM=args.MOMENTUM
cfg.SOLVER.BASE_LR = args.lr  # pick a good LR
cfg.MODEL.RPN.NMS_THRESH=args.NMS_THRESH
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN=args.PRE_NMS_TOPK_TRAIN
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST=args.PRE_NMS_TOPK_TEST
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN=args.POST_NMS_TOPK_TRAIN
cfg.MODEL.RPN.POST_NMS_TOPK_TEST=args.POST_NMS_TOPK_TEST
if (args.ANCHOR_SIZES==1):
    ANCHOR_SIZES=[[8,16,32, 64, 128]]
elif(args.ANCHOR_SIZES==2) :
    ANCHOR_SIZES =[[16, 32, 64, 128, 256]]
else:
    ANCHOR_SIZES =[[32, 64, 128, 256,512]]
cfg.MODEL.ANCHOR_GENERATOR.SIZES=ANCHOR_SIZES

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh_test   # set a custom testing threshold for this model
cfg.DATASETS.TEST = (damage_name+"_test",)

try:
    os.remove('output/last_checkpoint')
except OSError:
    pass

dice_dict={}
dice=[]
model_list=glob.glob('output/*.pth')
for md in model_list:
    if 'model' in md:
        if 'final' in md:
            continue
        cfg.MODEL.WEIGHTS = md
        predictor = DefaultPredictor(cfg)
        with open(test_json) as f:
            data = json.load(f)
        dice=[]
        #l=0
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
                print(str(e))
        final_dice=sum(dice)/len(dice)

        print('Model name: '+md)
        print('Dice value: ',str(final_dice))
        dice_dict[md]=final_dice
final_model = max(dice_dict, key=dice_dict.get) 
final_dice_val=dice_dict[final_model]
dice_dict_name='dice_dict.json'
with open(dice_dict_name, 'w') as outfile:
    json.dump(dice_dict,outfile,indent=4,ensure_ascii = False)

hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='dice', metric_value=final_dice_val, global_step=1)


def save_model(job_dir, model_name,dice_dict):
    """Saves the model to Google Cloud Storage"""
    job_dir = job_dir.replace('gs://', '')
    bucket_id = job_dir.split('/')[0]
    bucket_path = job_dir.lstrip('{}/'.format(bucket_id))
    # Upload the model to GCS
    bucket = storage.Client().bucket(bucket_id)
    blob = bucket.blob('{}/{}'.format(
        bucket_path,
        model_name[model_name.rfind('/')+1:]))
    blob.upload_from_filename(model_name)
    blob= bucket.blob('{}/{}'.format(
        bucket_path,
        dice_dict))
    blob.upload_from_filename(dice_dict)

save_model(args.job_dir,final_model,dice_dict_name)
