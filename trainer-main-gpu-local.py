import logging
import json
import numpy as np
import glob,shutil
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import sys
import cv2
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
import argparse
import hypertune
import detectron2
from google.cloud import storage
from pycocotools import coco
from detectron2 import model_zoo
import os
import yaml
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from collections import OrderedDict
import torch
from detectron2.data.datasets import register_coco_instances
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_setup, hooks, launch
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

with open('params.yaml', 'r') as stream:
    param_data=yaml.safe_load(stream)

def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.
    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:
Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth
Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '--damage_name',  # Handled automatically by AI Platform
        help='scratch,dent,crack etc',
        required=True
    )
    parser.add_argument('--max_iter',  # Specified in the config file
        type=int,
        default=6000,
        help='maximum iteration')
    parser.add_argument('--batch_size',  # Specified in the config file
        type=int,
        default=16,
        help='batch_size')
    parser.add_argument('--check_period',  # Specified in the config file
        type=int,
        default=500,
        help='checkpoint period')
    parser.add_argument('--thresh_test',  # Specified in the config file
        type=float,
        default=0.4,
        help='testing threshold')
    parser.add_argument(
        '--lr', default=0.0025, 
        type=float, 
        help='Learning rate parameter')
    parser.add_argument('--MOMENTUM',  # Specified in the config file
        type=float,
        default=0.9,
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
        default=6000,
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

    return parser

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    file_cfg='configs/'+param_data['MODEL']['CONFIG']
    cfg.merge_from_file(file_cfg)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    return cfg


def dice_calc(damage_name,cfg):
    test_json=dataset_dir+damage_name+param_data['DATASET'][MODE]['TEST_PATH']
    img_dir=dataset_dir+damage_name+param_data['DATASET'][MODE]['IMAGES_PATH']
    dice_dict={}
    dice=[]
    model_list=glob.glob(param_data['MODEL']['OUT_PATH']+'/*.pth')
    for md in model_list:
        if 'model' in md:
            if 'final' in md:
                continue
            cfg.MODEL.WEIGHTS = md
            predictor = DefaultPredictor(cfg)
            with open(test_json) as f:
                data = json.load(f)
            dice=[]
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
    return final_model,final_dice_val,dice_dict

def convert_cfg(args):
    cfg = setup(args)
    damage_name=args.damage_name
    print("dataset_dir: "+dataset_dir)
    #sys.exit()
    train_json=dataset_dir+damage_name+param_data['DATASET'][MODE]['TRAIN_PATH']
    val_json=dataset_dir+damage_name+param_data['DATASET'][MODE]['VAL_PATH']
    test_json=dataset_dir+damage_name+param_data['DATASET'][MODE]['TEST_PATH']
    print("dataset_dir: "+train_json)
    img_dir=dataset_dir+damage_name+param_data['DATASET'][MODE]['IMAGES_PATH']
    register_coco_instances(damage_name+"_train", {}, train_json, img_dir)
    register_coco_instances(damage_name+"_val", {}, val_json, img_dir)
    register_coco_instances(damage_name+"_test", {}, test_json, img_dir)

    cfg.DATASETS.TRAIN = (damage_name+"_train",)
    cfg.DATASETS.TEST = (damage_name+"_val",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(param_data['MODEL']['CONFIG'])
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = args.check_period
    cfg.SOLVER.MOMENTUM=args.MOMENTUM
    cfg.SOLVER.BASE_LR = args.lr 
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

    return cfg

def main(args):
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    MODE='LOCAL'

    print ('Available devices ', torch.cuda.device_count())
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    dataset_dir=param_data['DATASET'][MODE]['DIR_PATH']
    cfg=convert_cfg(args)
    print(cfg)
    print(cfg.DATASETS.TRAIN)
    #sys.exit()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    dirpath='plot'
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath) 

    json_file=param_data['JSONS']['TRAIN_LOSS']
    with open(json_file) as f:
        loss_data = json.load(f)

    iter_list=list(range(1,len(loss_data['loss_cls'])+1))
    iter_list=[x * 20 for x in iter_list]

    for i in list(loss_data.keys()):
        plt.plot(iter_list, loss_data[i])
        plt.title(i+' Vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel(i)
        plt.savefig('plot/'+i+'.png')
        plt.close()


