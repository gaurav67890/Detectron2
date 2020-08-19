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
import glob,shutil
print(os.system('ls'))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/etc/credentials.json"
os.system('gsutil cp gs://hptuning/split_damages.zip .')
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
    shutil.rmtree(dirpath)

args = parser.parse_args()

train_json="/detectron2_repo/split_damages/datasets/coco/scratch/annotations/instances_train.json"
val_json="/detectron2_repo/split_damages/datasets/coco/scratch/annotations/instances_validation.json"
test_json="/detectron2_repo/split_damages/datasets/coco/scratch/annotations/instances_test.json"

img_dir="/detectron2_repo/split_damages/datasets/coco/images/"
register_coco_instances("scratch_train", {}, train_json, img_dir)
register_coco_instances("scratch_val", {}, val_json, img_dir)
register_coco_instances("scratch_test", {}, val_json, img_dir)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("scratch_train",)
cfg.DATASETS.TEST = ("scratch_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from mode$
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.MAX_ITER = 1000 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (scratch)
cfg.SOLVER.CHECKPOINT_PERIOD = 200
#cfg.TEST.EVAL_PERIOD = 5000
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
#trainer = CocoTrainer(cfg) 
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set a custom testing threshold for this model
#evaluator = COCOEvaluator("scratch_test", cfg, False,output_dir="./output/")

map_dict={}
model_list=glob.glob('output/*.pth')
for i in model_list:
    if 'model' in i:
        cfg.MODEL.WEIGHTS = i
        predictor = DefaultPredictor(cfg)
        val_loader = build_detection_test_loader(cfg, "scratch_test")
        evaluator = COCOEvaluator("scratch_test", cfg, False,output_dir="./output/")
        results=inference_on_dataset(trainer.model, val_loader, evaluator)
        map_val=results['segm']['AP50']
        map_dict[i]=map_val
final_model = max(map_dict, key=map_dict.get) 
final_map=map_dict[final_model]


hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='MAP', metric_value=final_map, global_step=0)


def save_model(job_dir, model_name):
    """Saves the model to Google Cloud Storage"""
    # Example: job_dir = 'gs://BUCKET_ID/hptuning_sonar/1'
    job_dir = job_dir.replace('gs://', '')  # Remove the 'gs://'
    # Get the Bucket Id
    bucket_id = job_dir.split('/')[0]
    # Get the path. Example: 'hptuning_sonar/1'
    bucket_path = job_dir.lstrip('{}/'.format(bucket_id))

    # Upload the model to GCS
    bucket = storage.Client().bucket(bucket_id)
    blob = bucket.blob('{}/{}'.format(
        bucket_path,
        model_name[model_name.rfind('/')+1:]))
    blob.upload_from_filename(model_name)

save_model(args.job_dir,final_model)
