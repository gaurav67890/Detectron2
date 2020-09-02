import pickle
import logging
import json
import numpy as np
import glob,shutil
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
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
print(os.system('ls'))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/etc/credentials.json"
#os.system('gsutil cp gs://hptuning2/split_damages.zip .')
#os.system('unzip split_damages.zip')
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
        '--job-dir',  # Handled automatically by AI Platform
        help='GCS location to write checkpoints and export models'
        #required=True
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
        # In the end of training, run an evaluation with TTA
# Only support some R-CNN models.
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
    file_cfg='configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
    cfg.merge_from_file(file_cfg)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    default_setup(cfg, args)
    return cfg



def save_model(job_dir, model_name,dice_dict_name,plot_path):
    """Saves the model to Google Cloud Storage"""
    # Example: job_dir = 'gs://BUCKET_ID/hptuning_sonar/1'
    job_dir = job_dir.replace('gs://', '')  # Remove the 'gs://'
    # Get the Bucket Id
    bucket_id = job_dir.split('/')[0]
    # Get the path. Example: 'hptuning_sonar/1'
    bucket_path = job_dir.lstrip('{}/'.format(bucket_id))

    # Upload the data to GCS
    plot_names=glob.glob('plot/*.png')
    all_files=[plot_names,model_name,dice_dict_name]

    for f in all_files:
        bucket = storage.Client().bucket(bucket_id)
        if f==model_name:
            blob = bucket.blob('{}/{}'.format(
                bucket_path,
                f[f.rfind('/')+1:]))
        else:
            blob= bucket.blob('{}/{}'.format(
                bucket_path,
                f))
        blob.upload_from_filename(f)


def dice_calc(damage_name,cfg):
    test_json="/detectron2_repo/split_damages/datasets/coco/"+damage_name+"/annotations/instances_test.json"
    img_dir="/detectron2_repo/split_damages/datasets/coco/images/"
    dice_dict={}
    dice=[]
    model_list=glob.glob('output/*.pth')
    for md in model_list:
        if 'model' in md:
        #print('Model name: '+i)
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
                        dice.append(dice_score)
                        #l=l+1
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

    train_json="/detectron2_repo/split_damages/datasets/coco/"+damage_name+"/annotations/instances_train.json"
    val_json="/detectron2_repo/split_damages/datasets/coco/"+damage_name+"/annotations/instances_validation.json"
    test_json="/detectron2_repo/split_damages/datasets/coco/"+damage_name+"/annotations/instances_test.json"

    img_dir="/detectron2_repo/split_damages/datasets/coco/images/"
    register_coco_instances(damage_name+"_train", {}, train_json, img_dir)
    register_coco_instances(damage_name+"_val", {}, val_json, img_dir)
    register_coco_instances(damage_name+"_test", {}, test_json, img_dir)

    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))    
    cfg.DATASETS.TRAIN = (damage_name+"_train",)
    cfg.DATASETS.TEST = (damage_name+"_val",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from mode$
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.MAX_ITER = args.max_iter 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (dent)
    cfg.SOLVER.CHECKPOINT_PERIOD = args.check_period
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

    return cfg

def main(args):
    cfg = convert_cfg(args)

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
    os.system('gsutil cp gs://hptuning2/split_damages.zip .')
    os.system('unzip split_damages.zip')

    os.makedirs('output', exist_ok=True)
    print ('Available devices ', torch.cuda.device_count())
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    cfg=convert_cfg(args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    plotpath='plot'
    if os.path.exists(plotpath) and os.path.isdir(plotpath):
        shutil.rmtree(plotpath)
    os.mkdir(plotpath) 

    json_file='trainloss.pkl'
    #with open(json_file) as f:
    #    loss_data = json.load(f)
    a_filem = open(json_file, "rb")
    loss_data = pickle.load(a_filem)
    a_filem.close()
    iter_list=list(range(1,len(loss_data['loss_cls'])+1))
    iter_list=[x * 20 for x in iter_list]

    for i in list(loss_data.keys()):
        plt.plot(iter_list, loss_data[i])
        plt.title(i+' Vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel(i)
        plt.savefig('plot/'+i+'.png')
        plt.close()

    #cfg=convert_cfg(args)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh_test   # set a custom testing threshold for this model
    cfg.DATASETS.TEST = (args.damage_name+"_test",)

    try:
        os.remove('output/last_checkpoint')
    except OSError:
        pass
    final_model,final_dice_val,dice_dict=dice_calc(args.damage_name,cfg)
    dice_dict_name='dice_dict.json'
    with open(dice_dict_name, 'w') as outfile:
        json.dump(dice_dict,outfile,indent=4,ensure_ascii = False)

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='dice', metric_value=final_dice_val, global_step=1)

    save_model(args.job_dir,final_model,dice_dict_name,plot_path)
