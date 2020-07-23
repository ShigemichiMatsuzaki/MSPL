import argparse
import sys
from packaging import version
import time
import datetime
# import util
import os
import os.path as osp
import timeit
from collections import OrderedDict
import scipy.io

import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import transforms as ext_transforms
from operator import itemgetter
import copy

import scipy
from scipy import ndimage
import math
from PIL import Image
import numpy as np
import shutil
import random

from utilities.utils import save_checkpoint, model_parameters, compute_flops, in_training_visualization_img, set_logger, calc_cls_class_weight
from utilities.utils import AverageMeter
from utilities.metrics.segmentation_miou import MIOU
from utilities.train_eval_seg import train_seg as train
from utilities.train_eval_seg import val_seg as val
from loss_fns.segmentation_loss import NIDLoss, UncertaintyWeightedSegmentationLoss, PixelwiseKLD, SegmentationLoss

###
# Matsuzaki
###
#from enet.model import ENet 
# For visualization using TensorBoardX
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
#from metric.iou import IoU
from tqdm import tqdm
from data_loader.segmentation.greenhouse import id_camvid_to_greenhouse
from data_loader.segmentation.greenhouse import id_cityscapes_to_greenhouse

### shared ###
IMG_MEAN = np.array((0.406, 0.456, 0.485), dtype=np.float32) # BGR
IMG_STD = np.array((0.225, 0.224, 0.229), dtype=np.float32) # BGR
# data
### source
## gta
DATA_SRC_DIRECTORY = '/tmp/dataset/greenhouse'
DATA_SRC_LIST_PATH = './vision_datasets/greenhouse/train_greenhouse.txt'
DATA_SRC = 'greenhouse'
RESTORE_FROM = './src_model/gta5/src_model.pth'
NUM_CLASSES_SEG = 5
INIT_SRC_PORT = 0.03 # GTA: 0.03
### target
DATA_TGT_DIRECTORY = '/tmp/dataset/label_traversed'
DATA_TGT_TRAIN_LIST_PATH = './dataset/list/greenhouse/train_greenhouse_more.lst'
DATA_TGT_TEST_LIST_PATH = './dataset/list/greenhouse/val_greenhouse.lst'
IGNORE_LABEL = 4
# train scales for src and tgt
TRAIN_SCALE_SRC = '1.2,1.5'
TRAIN_SCALE_TGT = '1.2,1.5'
# model
MODEL = 'espdnet'
# gpu
GPU = 0
PIN_MEMORY = False
# log files
LOG_FILE = 'self_training_log'

### train ###
BATCH_SIZE = 2
INPUT_SIZE = '256,480'# 512,1024 for GTA;
RANDSEED = 3
# params for optimizor
LEARNING_RATE = 0.00005
POWER = 0.0
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_ROUNDS = 4
EPR = 2
SRC_SAMPLING_POLICY = 'r'
KC_POLICY = 'cb'
KC_VALUE = 'conf'
INIT_TGT_PORT = 0.2
MAX_TGT_PORT = 0.5
TGT_PORT_STEP = 0.05
# varies but dataset
MAX_SRC_PORT = 0.06 #0.06;
SRC_PORT_STEP = 0.0025 #0.0025:
MRKLD = 0.0
LRENT = 0.0
MRSRC = 0.0
MINE_PORT = 1e-3
RARE_CLS_NUM = 3
MINE_CHANCE = 0.0
### val ###
SAVE_PATH = 'debug'
TEST_IMAGE_SIZE = '256,480'
EVAL_SCALE = 0.9
TEST_SCALE = '0.9,1.0,1.2'
DS_RATE = 4
OPTIMIZER = 'Adam'
RUNS_ROOT = './runs'

TRAV_ROOT = None

DATASET = 'greenhouse'

def seed_torch(seed=0):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
   #torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.enabled = False
   #torch.backends.cudnn.deterministic = True

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    ### shared by train & val
    # data
    parser.add_argument('--savedir', type=str, default='./results_segmentation', help='Location to save the results')
    parser.add_argument('--data-path', type=str, default='', help='dataset path')
    parser.add_argument("--data-src", type=str, default=DATA_SRC,
                        help="Name of source dataset.")
    parser.add_argument("--data-src-dir", type=str, default=DATA_SRC_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-src-list", type=str, default=DATA_SRC_LIST_PATH,
                        help="Path to the file listing the images&labels in the source dataset.")
    parser.add_argument("--data-tgt-dir", type=str, default=DATA_TGT_DIRECTORY,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-tgt-train-list", type=str, default=DATA_TGT_TRAIN_LIST_PATH,
                        help="Path to the file listing the images*GT labels in the target train dataset.")
    parser.add_argument("--data-tgt-test-list", type=str, default=DATA_TGT_TEST_LIST_PATH,
                        help="Path to the file listing the images*GT labels in the target test dataset.")
    parser.add_argument("--num-classes-seg", type=int, default=NUM_CLASSES_SEG,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    # model
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    # gpu
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--pin-memory", type=bool, default=PIN_MEMORY,
                        help="Whether to pin memory in train & eval.")
    # log files
    parser.add_argument("--log-file", type=str, default=LOG_FILE,
                        help="The name of log file.")
    parser.add_argument('--debug',help='True means logging debug info.',
                        default=False, action='store_true')
    ### train ###
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")

    parser.add_argument('--crop-size', type=int, nargs='+', default=[480, 256],
                        help='list of image crop sizes, with each item storing the crop size (should be a tuple).')

    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--eval-training", action="store_true",
                        help="Use the saved means and variances, or running means and variances during the evaluation.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--train-scale-src", type=str, default=TRAIN_SCALE_SRC,
                        help="The scale for multi-scale training in source domain.")
    parser.add_argument("--train-scale-tgt", type=str, default=TRAIN_SCALE_TGT,
                    help="The scale for multi-scale training in target domain.")

    # params for optimizor
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER,
                        help="Optimizer used in the training")
    ### val
    parser.add_argument('--test-flipping', dest='test_flipping',
                        help='If average predictions of original and flipped images.',
                        default=False, action='store_true')
    parser.add_argument("--test-image-size", type=str, default=TEST_IMAGE_SIZE,
                        help="The test image size.")
    parser.add_argument("--eval-scale", type=float, default=EVAL_SCALE,
                        help="The test image scale.")
    parser.add_argument("--test-scale", type=str, default=TEST_SCALE,
                        help="The test image scale.")
    ### self-training params
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result for self-training.")
    parser.add_argument("--num-rounds", type=int, default=NUM_ROUNDS,
                        help="Number of rounds for self-training.")
    parser.add_argument("--epr", type=int, default=EPR,
                        help="Number of epochs per round for self-training.")
    parser.add_argument('--kc-policy', default=KC_POLICY, type=str, dest='kc_policy',
                        help='The policy to determine kc. "cb" for weighted class-balanced threshold')
    parser.add_argument('--kc-value', default=KC_VALUE, type=str,
                        help='The way to determine kc values, either "conf", or "prob".')
    parser.add_argument('--ds-rate', default=DS_RATE, type=int,
                        help='The downsampling rate in kc calculation.')
    parser.add_argument('--init-tgt-port', default=INIT_TGT_PORT, type=float, dest='init_tgt_port',
                        help='The initial portion of target to determine kc')
    parser.add_argument('--max-tgt-port', default=MAX_TGT_PORT, type=float, dest='max_tgt_port',
                        help='The max portion of target to determine kc')
    parser.add_argument('--tgt-port-step', default=TGT_PORT_STEP, type=float, dest='tgt_port_step',
                        help='The portion step in target domain in every round of self-paced self-trained neural network')
    parser.add_argument('--init-src-port', default=INIT_SRC_PORT, type=float, dest='init_src_port',
                        help='The initial portion of source portion for self-trained neural network')
    parser.add_argument('--max-src-port', default=MAX_SRC_PORT, type=float, dest='max_src_port',
                        help='The max portion of source portion for self-trained neural network')
    parser.add_argument('--src-port-step', default=SRC_PORT_STEP, type=float, dest='src_port_step',
                        help='The portion step in source domain in every round of self-paced self-trained neural network')
    parser.add_argument('--randseed', default=RANDSEED, type=int,
                        help='The random seed to sample the source dataset.')
    parser.add_argument("--src-sampling-policy", type=str, default=SRC_SAMPLING_POLICY,
                        help="The sampling policy on source dataset: 'c' for 'cumulative' and 'r' for replace ")
    parser.add_argument('--mine-port', default=MINE_PORT, type=float,
                        help='If a class has a predication portion lower than the mine_port, then mine the patches including the class in self-training.')
    parser.add_argument('--rare-cls-num', default=RARE_CLS_NUM, type=int,
                        help='The number of classes to be mined.')
    parser.add_argument('--mine-chance', default=MINE_CHANCE, type=float,
                        help='The chance of patch mining.')
    parser.add_argument('--rm-prob',
                        help='If remove the probability maps generated in every round.',
                        default=False, action='store_true')
    parser.add_argument('--mr-weight-kld', default=MRKLD, type=float, dest='mr_weight_kld',
                        help='weight of kld model regularization')
    parser.add_argument('--lr-weight-ent', default=LRENT, type=float, dest='lr_weight_ent',
                        help='weight of negative entropy label regularization')
    parser.add_argument('--mr-weight-src', default=MRSRC, type=float, dest='mr_weight_src',
                        help='weight of regularization in source domain')

    parser.add_argument('--dataset', default=DATASET, type=str, dest='dataset',
                        help='the name of dataset to train')
    parser.add_argument("--runs-root", type=str, default=RUNS_ROOT, dest='runs_root',
                        help="Path to save tensorboard log")
    parser.add_argument("--trav-root", type=str, default=TRAV_ROOT, dest='trav_root',
                        help="The root directory of the dataset including traversability_masks")
    # model related params
    parser.add_argument('--s', type=float, default=2.0, help='Factor by which channels will be scaled')
    parser.add_argument('--model', default='espnet',
                        help='Which model? basic= basic CNN model, res=resnet style)')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')
    parser.add_argument('--finetune', default='', type=str, help='Finetune the segmentation model')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--use-depth', default=False, type=bool, help='Use depth')
    parser.add_argument('--trainable-fusion', default=False, type=bool, help='Use depth')
    parser.add_argument('--dense-fuse', default=False, type=bool, help='Use depth')
    parser.add_argument("--outsource", type=str, default=None, dest='outsource',
                        help="A dataset name that is used as a external dataset to provide initial pseudo-labels")
    parser.add_argument("--os-model", type=str, default="espdnet", dest='os_model',
                        help="Model for generating pseudo-labels")
    parser.add_argument("--os-weights", type=str, default='./results_segmentation/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb_/20200420-095339/espdnet_2.0_480_best.pth', dest='os_weights',
                        help="A dataset name that is used as a external dataset to provide initial pseudo-labels")
    parser.add_argument("--use-traversable", type=str, default=False, dest='use_traversable',
                        help="Whether to use a class 'traversable plant'")
    parser.add_argument("--early-stop", type=str, default=False, dest='early_stop',
                        help="Whether to stop the training if the mean IoU is substancially degraded")
    parser.add_argument('--use-nid', default=False, type=bool, help='Use NID loss')
    parser.add_argument('--nid-bin', default=32, type=int, help='Bin size of an image intensity histogram in calculating  NID loss')
    parser.add_argument('--use-uncertainty', default=False, type=bool, help='Use uncertainty weighting')
    parser.add_argument("--tidyup", type=bool, default=True,
                        help="Whether to remove label images etc. after the training")

    return parser.parse_args()

args = get_arguments()

# palette
if args.data_src == 'greenhouse':
    palette = [0, 255, 0,
               0, 255, 255,
               255, 0, 0,
               255, 255, 0,
               0, 0, 0]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def main():
    randseed = args.randseed
    seed_torch(randseed)
    device = torch.device("cuda:" + str(args.gpu))

    args.weights = args.restore_from
    args.scale = (0.5, 2.0)
    args.crop_size = tuple(args.crop_size)
#    timestr = time.strftime("%Y%m%d-%H%M%S")
    now = datetime.datetime.now()
    now += datetime.timedelta(hours=9)
    timestr = now.strftime("%Y%m%d-%H%M%S")
    use_depth_str = "_rgbd" if args.use_depth else "_rgb"
    if args.use_depth:
        trainable_fusion_str = "_gated" if args.trainable_fusion else "_naive"
    else:
        trainable_fusion_str = ""

    outsource_str = "_os_" + args.outsource if args.outsource else ""
    ue_str   = "_ue" if args.use_uncertainty else ""
    loss_str = "_nid_{}".format(args.nid_bin) if args.use_nid else ""

    args.savedir = '{}/model_{}_{}/s_{}_res_{}_uest{}{}{}{}{}/{}'.format(
        args.save, 
        args.model, args.dataset, 
        args.s, args.crop_size[0], use_depth_str, trainable_fusion_str, outsource_str, ue_str, loss_str,
        timestr)

    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    save_path = args.savedir
    # Initialize save path
    save_pseudo_label_path = osp.join(save_path, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    save_stats_path = osp.join(save_path, 'stats') # in 'save_path'
    save_lst_path = osp.join(save_path, 'list')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_pseudo_label_path):
        os.makedirs(save_pseudo_label_path)
    if not os.path.exists(save_stats_path):
        os.makedirs(save_stats_path)
    if not os.path.exists(save_lst_path):
        os.makedirs(save_lst_path)

    logger = set_logger(args.save, args.log_file, args.debug)
    logger.info('start with arguments %s', args)

    # Import Dataset
    sys.path.insert(0, 'dataset/helpers')
    if args.dataset == 'greenhouse':
        from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation, GREENHOUSE_CLASS_LIST

        class_weights = np.load('class_weights.npy')# [:4]
        class_weights = torch.from_numpy(class_weights).float().to(device)

        seg_classes = len(GREENHOUSE_CLASS_LIST)
        class_encoding = OrderedDict([
            ('end_of_plant', (0, 255, 0)),
            ('other_part_of_plant', (0, 255, 255)),
            ('artificial_objects', (255, 0, 0)),
            ('ground', (255, 255, 0)),
            ('background', (0, 0, 0))
        ])

        # Maybe error
        label_2_id = 255 * np.ones((256,))
        for l in range(args.num_classes):
            label_2_id[l] = l
        id_2_label = label_2_id

    # Import pretrained model
    if args.model == 'espdnet':
        from model.segmentation.espdnet import espdnet_seg_with_pre_rgbd
        args.classes = seg_classes
        model = espdnet_seg_with_pre_rgbd(args, load_entire_weights=True)

    elif args.model == 'espdnetue':
        from model.segmentation.espdnet_ue import espdnetue_seg2
        args.classes = seg_classes
        model = espdnetue_seg2(args, load_entire_weights=True, fix_pyr_plane_proj=True)
            
    elif args.model == 'deeplabv3':
        from torchvision.models.segmentation.segmentation import deeplabv3_resnet101
        model = deeplabv3_resnet101(num_classes=seg_classes, aux_loss=True)

    # Outsource
    # Import pretrained model trained for giving initial pseudo-labels
    if args.outsource == 'camvid':
        os_seg_classes = 13
        # Import model
        if args.os_model == 'espdnet':
            from model.segmentation.espdnet import espdnet_seg_with_pre_rgbd
            tmp_args = copy.deepcopy(args)
            tmp_args.trainable_fusion = False
            tmp_args.dense_fuse = False
            tmp_args.use_depth  = False
            tmp_args.classes = os_seg_classes
            tmp_args.dataset = 'camvid'
            tmp_args.weights = args.os_weights
            model_outsource = espdnet_seg_with_pre_rgbd(tmp_args, load_entire_weights=True)
        elif args.os_model == 'espdnetue':
            from model.segmentation.espdnet_ue import espdnetue_seg2
            tmp_args = copy.deepcopy(args)
            tmp_args.trainable_fusion = False
            tmp_args.dense_fuse = False
            tmp_args.use_depth  = False
            tmp_args.classes = os_seg_classes
            tmp_args.dataset = 'camvid'
            tmp_args.weights = args.os_weights
           
            model_outsource = espdnetue_seg2(tmp_args, load_entire_weights=True, fix_pyr_plane_proj=True)
        elif args.os_model == 'deeplabv3':
            from torchvision.models.segmentation.segmentation import deeplabv3_resnet101

            model_outsource = deeplabv3_resnet101(num_classes=os_seg_classes, aux_loss=True)
            # Import pre-trained weights
            #/tmp/runs/model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200710-185848/
            load_weights(model_outsource, args.os_weights)

        # Calculate class weights from the outsource dataset
        from data_loader.segmentation.camvid import CamVidSegmentation
        ds = CamVidSegmentation(
            root='', list_name=args.data_src_list, train=True, label_conversion=True)
        tmp_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, pin_memory=args.pin_memory)
        
        print("Calculate class weights!")
        class_weights = calc_cls_class_weight(tmp_loader, args.num_classes_seg)
        print(class_weights)
        class_weights = torch.from_numpy(class_weights).float().to(device)
    elif args.outsource == 'cityscapes':
        os_seg_classes = 20
        if args.os_model == 'espdnet':
            from model.segmentation.espdnet import espdnet_seg_with_pre_rgbd
            tmp_args = copy.deepcopy(args)
            tmp_args.trainable_fusion = False
            tmp_args.dense_fuse = False
            tmp_args.use_depth  = False
            tmp_args.classes = os_seg_classes
            tmp_args.dataset = 'city'
            tmp_args.weights = args.os_weights

            model_outsource = espdnet_seg_with_pre_rgbd(tmp_args, load_entire_weights=True)
        elif args.os_model == 'espdnetue':
            from model.segmentation.espdnet_ue import espdnetue_seg2
            tmp_args = copy.deepcopy(args)
            tmp_args.trainable_fusion = False
            tmp_args.dense_fuse = False
            tmp_args.use_depth  = False
            tmp_args.classes = os_seg_classes
            tmp_args.dataset = 'city'
            tmp_args.weights = args.os_weights
           
            model_outsource = espdnetue_seg2(tmp_args, load_entire_weights=True, fix_pyr_plane_proj=True)

        elif args.os_model == 'deeplabv3':
            from torchvision.models.segmentation.segmentation import deeplabv3_resnet101
            model_outsource = deeplabv3_resnet101(num_classes=os_seg_classes, aux_loss=True)
            # Import pre-trained weights
            #/tmp/runs/model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200710-185848/
            load_weights(model_outsource, args.os_weights)

        class_weights = torch.ones(seg_classes).float().to(device)
        print(class_weights)


    print("Model {} is load successfully!".format(args.restore_from))

    # List of training images
    if args.use_depth:
        image_src_list, _, label_src_list, depth_src_list, src_num = parse_split_list(args.data_src_list)
    else:
        image_src_list, _, label_src_list, depth_src_list, src_num = parse_split_list(args.data_src_list)
    image_tgt_list, image_name_tgt_list, _, depth_tgt_list, tgt_num = parse_split_list(args.data_tgt_train_list)
    # print("tgt_num", tgt_num)
    _, _, _, _, test_num = parse_split_list(args.data_tgt_test_list)

    # valid_labels : A list of valid labels?
    #   ravel(): Flatten a multi-dimensional array to a 1D list
    #   set()  : A constructor to generate a set object
    valid_labels = sorted(set(id_2_label.ravel()))

    # portions
    tgt_portion = args.init_tgt_port
    src_portion = args.init_src_port

    # training crop size
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    lscale_src, hscale_src = map(float, args.train_scale_src.split(','))
    train_scale_src = (lscale_src, hscale_src)
    lscale_tgt, hscale_tgt = map(float, args.train_scale_tgt.split(','))
    train_scale_tgt = (lscale_tgt, hscale_tgt)

#    metric = IoU(args.num_classes, ignore_index=args.ignore_label)
    metric = None
    print("Let's start! Hey hey!")

    # Loss functions
    if args.use_uncertainty:
        criterion = UncertaintyWeightedSegmentationLoss(seg_classes, class_weights, args.ignore_label)
    else:
        criterion = SegmentationLoss(n_classes=seg_classes,
                                     device=device, ignore_idx=args.ignore_label,
                                     class_wts=class_weights.to(device))

    criterion_test = SegmentationLoss(n_classes=seg_classes,
                                     device=device, ignore_idx=args.ignore_label,
                                     class_wts=class_weights.to(device))
    nid_loss = NIDLoss(image_bin=args.nid_bin, label_bin=seg_classes) if args.use_nid else None

    # 
    # Main loop
    #s
    writer_idx = 0
    #class_weights = None
    writer = SummaryWriter(save_path)
    old_miou = -0.0

    #
    # Pseudo-label generation
    #
    ### Preparation
    save_pseudo_label_color_path = osp.join(save_path, 'pseudo_label_color')  # in every 'save_round_eval_path'
    if not os.path.exists(save_pseudo_label_color_path):
        os.makedirs(save_pseudo_label_color_path)
    ## output folder
    save_pred_vis_path = osp.join(save_path, 'pred_vis')
    save_prob_path = osp.join(save_path, 'prob')
    save_pred_path = osp.join(save_path, 'pred')
    save_conf_path = osp.join(save_path, 'conf')
    if not os.path.exists(save_pred_vis_path):
        os.makedirs(save_pred_vis_path)
    if not os.path.exists(save_prob_path):
        os.makedirs(save_prob_path)
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)
    if not os.path.exists(save_conf_path):
        os.makedirs(save_conf_path)

    if args.outsource is not None:
        tgt_train_lst = val(model_outsource, device, save_path, 
                            0, tgt_num, label_2_id, valid_labels,
                            args, logger, class_encoding, writer, args.outsource)
    else:
        tgt_train_lst = val(model, device, save_path, 
                            0, tgt_num, label_2_id, valid_labels,
                            args, logger, class_encoding, writer)

    best_miou = 0.0
    for round_idx in range(args.num_rounds):

        ########## pseudo-label generation
        if round_idx != args.num_rounds - 1: # If it's not the last round
            # Create the list of training data
#            src_train_lst, tgt_train_lst = 

            ########### model retraining
            # dataset
            epoch_per_round = args.epr # The number of epochs
            # reg weights
            if args.mr_weight_kld == 0:
                reg_weight_tgt = 0.0
            else:  # currently only one kind of model regularizer is supported
                reg_weight_tgt = args.mr_weight_kld

            reg_weight_src = args.mr_weight_src

            # Initial test
            tgt_set = 'test'
            if round_idx:
                test(model, criterion_test, device, round_idx, tgt_set, test_num, args.data_tgt_test_list, label_2_id,
                     valid_labels, args, logger, class_encoding, metric, writer, class_weights, reg_weight_tgt)
            else:
                # Get the initial IoU in the first round for early stop
                old_miou = test(model, criterion_test, device, round_idx, tgt_set, test_num, args.data_tgt_test_list, label_2_id,
                                valid_labels, args, logger, class_encoding, metric, writer, class_weights, reg_weight_tgt)

            # dataloader
            #  New dataset (labels) is created every round
            if args.lr_weight_ent == 0.0:
                if args.dataset == 'greenhouse':
                    from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation
                    from data_loader.segmentation.greenhouse import GreenhouseRGBDStMineDataSet
                    from data_loader.segmentation.greenhouse import GREENHOUSE_CLASS_LIST
                    from data_loader.segmentation.camvid import CamVidSegmentation

                    # Initialize the class weights 
                    if class_weights is None:
                        class_weights = np.ones(args.num_classes_seg)
                        class_weights = torch.from_numpy(class_weights).float().to(device)

                    tgttrainset = GreenhouseRGBDSegmentation(
                        list_name=tgt_train_lst, train=True, 
                        size=args.crop_size, scale=args.scale, use_depth=args.use_depth)

                else:
                    pass
            elif args.lr_weight_ent > 0.0:
                pass

            # Create a dataset concatinating the source dataset and the target dataset
#            mixtrainset = torch.utils.data.ConcatDataset([srctrainset, tgttrainset])
            mixtrainset = tgttrainset
            mix_trainloader = torch.utils.data.DataLoader(
                mixtrainset, batch_size=args.batch_size, shuffle=True,
                num_workers=0, pin_memory=args.pin_memory)
            # optimizer
            if args.use_depth:
                train_params = [{'params': model.get_basenet_params(), 'lr': args.learning_rate},
                                {'params': model.get_segment_params(), 'lr': args.learning_rate * 10},
                                {'params': model.get_depth_encoder_params(), 'lr': args.learning_rate * 10}]
            else:
                train_params = [{'params': model.get_basenet_params(), 'lr': args.learning_rate},
                                {'params': model.get_segment_params(), 'lr': args.learning_rate * 10}]

            tot_iter = np.ceil(float(tgt_num) / args.batch_size)
            if args.optimizer == 'SGD':
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=args.learning_rate,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)            
            elif args.optimizer == 'Adam':
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay)            

            interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

            logger.info('###### Start model retraining dataset in round {}! ######'.format(round_idx))

            # model
            if args.is_training:
                model.train()
            else:
                model.eval()

            start = timeit.default_timer()

            # cudnn
            cudnn.enabled = True  # enable cudnn
            cudnn.benchmark = True  # enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

            #
            # start retraining
            #
            for epoch in range(epoch_per_round):
                #  train(mix_trainloader, model, device, interp, optimizer, tot_iter, round_idx, epoch, args, logger)
                writer_idx = train(
                    mix_trainloader, model, criterion, device, interp, optimizer, tot_iter, round_idx, 
                    epoch, args, logger, metric, class_encoding, writer_idx, class_weights, writer, nid_loss)

            end = timeit.default_timer()
            logger.info('###### Finish model retraining dataset in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx, end - start))

            # test self-trained model in target domain test set
            tgt_set = 'test'
            new_miou = test(model, criterion_test, device, round_idx+1, tgt_set, test_num, args.data_tgt_test_list, label_2_id,
                            valid_labels, args, logger, class_encoding, metric, writer, class_weights, reg_weight_tgt)

            if args.early_stop and old_miou - new_miou > 10.0:
                logger.info(
                    '###### Accuracy degraded too much. ({} -> {}) Sorry, no hope. ######'.format(old_miou, new_miou))

                return

            # remember best miou and save checkpoint
            is_best = new_miou > best_miou
            best_miou = max(new_miou, best_miou)
            weights_dict = model.module.state_dict() if device == 'cuda' else model.state_dict()
            extra_info_ckpt = '{}'.format(args.model)
            if is_best:
                save_checkpoint({
                    'epoch': round_idx + 1,
                    'arch': args.model,
                    'state_dict': weights_dict,
                    'best_miou': best_miou,
                    'optimizer': optimizer.state_dict(),
                }, is_best, args.savedir, extra_info_ckpt)

        elif round_idx == args.num_rounds - 1:
            shutil.rmtree(save_pseudo_label_path)
            tgt_set = 'train'
            test(model, criterion_test, device, round_idx+1, tgt_set, tgt_num, args.data_tgt_train_list, label_2_id,
                 valid_labels, args, logger, class_encoding, metric, writer, class_weights, reg_weight_tgt)
            tgt_set = 'test'
            test(model, criterion_test, device, round_idx+2, tgt_set, test_num, args.data_tgt_test_list, label_2_id,
                 valid_labels, args, logger, class_encoding, metric, writer, class_weights, reg_weight_tgt)

    # Remove label images and weight files
    if args.tidyup:
        shutil.rmtree(osp.join(save_path, 'pred_vis'))
        shutil.rmtree(osp.join(save_path, 'prob'))
        shutil.rmtree(osp.join(save_path, 'pred'))
        shutil.rmtree(osp.join(save_path, 'conf'))


"""Create the model and start the evaluation process."""
def val(model, device, save_path, round_idx, 
        tgt_num, label_2_id, valid_labels, args, logger, class_encoding, writer, outsource=None):
    ## scorer
    scorer = ScoreUpdater(valid_labels, args.num_classes_seg, tgt_num, logger)
    scorer.reset()
    h, w = map(int, args.test_image_size.split(','))
    test_image_size = (h, w)
    test_size = ( int(h*args.eval_scale), int(w*args.eval_scale) )

    ## test data loader
    if args.dataset == 'greenhouse':
        from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation

        ds = GreenhouseRGBDSegmentation(
            list_name=args.data_tgt_train_list, train=False, use_traversable=args.use_traversable, use_depth=args.use_depth)
#        testloader = data.DataLoader(ds, batch_size=1, shuffle=False, pin_memory=args.pin_memory)

    testloader = data.DataLoader(ds, batch_size=1, shuffle=False, pin_memory=args.pin_memory)

    ## model for evaluation
    if args.eval_training:
        model.train()
    else:
        model.eval()
    #
    model.to(device)

    ## upsampling layer
    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=test_image_size, mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=test_image_size, mode='bilinear')

    ## output of deeplab is logits, not probability
    softmax2d = nn.Softmax2d()

    save_pred_vis_path = osp.join(save_path, 'pred_vis')
    save_prob_path = osp.join(save_path, 'prob')
    save_pred_path = osp.join(save_path, 'pred')
    save_conf_path = osp.join(save_path, 'conf')
    tgt_train_lst = osp.join(save_path, 'tgt_train.lst')

    # saving output data
    conf_dict = {k: [] for k in range(args.num_classes_seg)}
    pred_cls_num = np.zeros(args.num_classes_seg)

    ## evaluation process
    logger.info('###### Start evaluating target domain train set in round {}! ######'.format(round_idx))
    start_eval = time.time()
    image_path_list = []
    label_path_list = []
    depth_path_list = []
    with torch.no_grad():
        ious = 0
        with tqdm(total=len(testloader)) as pbar:
            for index, batch in enumerate(tqdm(testloader)):
                if args.use_depth:
                    image, label, depth, name, _ = batch
                else:
                    image, label, name, _ = batch

                # if args.model == 'ENet':
                if not args.use_depth: #or outsource == 'camvid':
                    output2 = model(image.to(device))
                else:
                    output2 = model(image.to(device), depth.to(device))

                if isinstance(output2, OrderedDict):
                    if len(output2) == 2:
                        output2 = output2['out'] + 0.5 * output2['aux']
                    else:
                        output2 = output2['out']
                elif args.os_model == 'espdnetue':
                    output2 = output2[0] + 0.5 * output2[1]

                output = softmax2d(interp(output2)).cpu().data[0].numpy()
    
                if args.test_flipping: # and args.model == 'ENet':
                    if not args.use_depth:# or outsource == 'camvid':
                        output2 = model(
                            torch.from_numpy(image.numpy()[:,:,:,::-1].copy()).to(device))
                    else:
                        output2 = model(
                            torch.from_numpy(image.numpy()[:,:,:,::-1].copy()).to(device),
                            torch.from_numpy(depth.numpy()[:,:,:,::-1].copy()).to(device))

                    if isinstance(output2, OrderedDict):
                        if len(output2) == 2:
                            output2 = output2['out'] + 0.5 * output2['aux']
                        else:
                            output2 = output2['out']
                    elif args.os_model == 'espdnetue':
                        output2 = output2[0] + 0.5 * output2[1]


                    output = 0.5 * ( output + softmax2d(interp(output2)).cpu().data[0].numpy()[:,:,::-1] )

                # If the label is transfered from an external dataset,
                #   convert the output tensor (numpy)
#                if outsource == 'camvid':
#                    output = transfer_output_to_greenhouse(id_camvid_to_greenhouse, output)
#                elif outsource == 'cityscapes':
#                    output = transfer_output_to_greenhouse(id_cityscapes_to_greenhouse, output)

                output = output.transpose(1,2,0)
                amax_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                conf = np.amax(output,axis=2) # amax : get a maximum value
                # label = label_2_id[np.asarray(label.numpy(), dtype=np.uint8)]
    
                # save visualized seg maps & predication prob map
                if outsource == 'camvid':
                    amax_output = id_camvid_to_greenhouse[amax_output]
                elif outsource == 'cityscapes':
                    amax_output = id_cityscapes_to_greenhouse[amax_output]

                # score
                pred_label = amax_output.copy()
                amax_output_col = colorize_mask(amax_output)

    #            label = label_2_id[np.asarray(label.numpy(), dtype=np.uint8)]
                path_name = name[0]
                name = name[0].split('/')[-1]
                image_name = name.rsplit('.', 1)[0]
                # prob
                np.save('%s/%s.npy' % (save_prob_path, image_name), output)
                np.save('%s/%s.npy' % (save_conf_path, image_name), conf)
                # trainIDs/vis seg maps
                amax_output = Image.fromarray(amax_output.astype(np.uint8))
                # Save the predicted images (+ colorred images for visualization)
                amax_output.save('%s/%s.png' % (save_pred_path, image_name))
                amax_output_col.save('%s/%s_color.png' % (save_pred_vis_path, image_name))

                image_path_list.append(path_name)
                label_path_list.append('%s/%s.png' % (save_pred_path, image_name)) 
                if args.use_depth:
                    depth_path_list.append(path_name.replace('color', 'depth'))
        
        pbar.close()

    with open(tgt_train_lst, 'w') as f:
        for idx in range(len(image_path_list)):
            if args.use_depth:
                f.write("%s,%s,%s\n" % (image_path_list[idx], label_path_list[idx], depth_path_list[idx]))
            else:
                f.write("%s,%s\n" % (image_path_list[idx], label_path_list[idx]))

    batch = iter(testloader).next()
    image = batch[0].to(device)
    label = batch[1].long()
    if args.use_depth:
        depth = batch[2].to(device)
        in_training_visualization_img(
            model, images=image, depths=depth, labels=label,
            class_encoding=class_encoding, writer=writer, epoch=round_idx, data='cbst_enet/val', device=device)

#    if args.dataset != 'greenhouse':
#        label = label_2_id[label.cpu().numpy()]
#        writer.add_scalar('cbst_enet/val/mean_IoU', np.mean(ious)/len(testloader), round_idx)
#        util.in_training_visualization_img(model, image, torch.from_numpy(label).long(), class_encoding, writer, round_idx, 'cbst_enet/val', device)

    logger.info('###### Finish evaluating target domain train set in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx, time.time()-start_eval))

    return tgt_train_lst  # return the dictionary containing all the class-wise confidence vectors

def train(trainloader, model, criterion, device, interp, optimizer, tot_iter, round_idx, epoch_idx,
          args, logger, metric, class_encoding, writer_idx, class_weights=None, writer=None, add_loss=None):
    """Create the model and start the training."""
    epoch_loss = 0
    
    # For logging the training status
    losses = AverageMeter()
    nid_losses = AverageMeter()
    kld_losses = AverageMeter()
    batch_time = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()

    miou_class = MIOU(num_classes=4)

    kld_layer = PixelwiseKLD()
    with tqdm(total=len(trainloader)) as pbar:
        for i_iter, batch in enumerate(tqdm(trainloader)):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            if args.use_depth:
                depths = batch[2].to(device)
    
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter, tot_iter)
    
            # Upsample the output of the model to the input size
            # TODO: Change the input according to the model type
            interp = None
            if interp is not None:
                if args.use_depth:
                    if args.model == 'espdnet':
                        pred = interp(model(images, depths))
                    elif args.model == 'espdnetue':
                        (pred, pred_aux) = interp(model(images, depths))
                else:
                    if args.model == 'espdnet':
                        pred = interp(model(images))
                    elif args.model == 'espdnetue':
                        (pred, pred_aux) = interp(model(images))
                    elif args.model == 'deeplabv3':
                        output = model(images)
                        pred = interp(output['out'])
                        pred_aux = interp(output['aux'])
            else:
                if args.use_depth:
                    if args.model == 'espdnet':
                        pred = model(images, depths)
                    elif args.model == 'espdnetue':
                        (pred, pred_aux) = model(images, depths)
                else:
                    if args.model == 'espdnet':
                        pred = model(images)
                    elif args.model == 'espdnetue':
                        (pred, pred_aux) = model(images)
                    elif args.model == 'deeplabv3':
                        output = model(images)
                        pred = output['out']
                        pred_aux = output['aux']
    
    #        print(pred.size())
            # Model regularizer
            kld = kld_layer(pred, pred_aux)
            kld_losses.update(kld.mean().item(), 1)
            if args.use_uncertainty:
                loss = criterion(pred + 0.5*pred_aux, labels, kld) * 20 + kld.mean()
            else:
                loss = criterion(pred + 0.5*pred_aux, labels)# + kld.mean()

            if add_loss is not None:
                loss2 = add_loss(images, pred.to(device))
                nid_losses.update(loss2.item(), 1)
                loss += loss2

            inter, union = miou_class.get_iou(pred, labels)
    
            inter_meter.update(inter)
            union_meter.update(union)
    
            losses.update(loss.item(), images.size(0))
    
            # Optimise
            loss.backward()
            optimizer.step()
    
            # Keep track of the evaluation metric
            # TODO: Change the metric
#            metric.add(pred.detach(), labels.detach())
    
            # Calculate IoU
#            (iou, miou) = metric.value()
    
    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou[[1, 2, 3]].mean() * 100

    # Write summary
    writer.add_scalar('cbst_enet/train/loss', losses.avg, writer_idx)
    writer.add_scalar('cbst_enet/train/nid_loss', nid_losses.avg, writer_idx)
    writer.add_scalar('cbst_enet/train/mean_IoU', miou, writer_idx)
    writer.add_scalar('cbst_enet/train/traversable_plant_IoU', iou[0], writer_idx)
    writer.add_scalar('cbst_enet/train/other_plant_mean_IoU', iou[1], writer_idx)
    writer.add_scalar('cbst_enet/train/learning_rate', optimizer.param_groups[0]['lr'], writer_idx)
    writer.add_scalar('cbst_enet/train/kld', kld_losses.avg, writer_idx)
  
    #
    # Investigation of labels
    #
    
    # Before label conversion
    if args.use_depth: 
        # model, images, depths=None, labels=None, predictions=None, class_encoding=None, writer=None, epoch=None, data=None, device=None
        in_training_visualization_img(model, images=images, depths=depths, labels=labels.long(), class_encoding=class_encoding, 
            writer=writer, epoch=writer_idx, data='cbst_enet/train', device=device)
    else:
        in_training_visualization_img(model, images=images, labels=labels.long(), class_encoding=class_encoding, 
            writer=writer, epoch=writer_idx, data='cbst_enet/train', device=device)

    
    writer_idx += 1
    
    #        logger.info('iter = {} of {} completed, loss = {:.4f}'.format(i_iter+1, tot_iter, loss.data.cpu().numpy()))

    print('taking snapshot ...')
#    torch.save(model.state_dict(), osp.join(args.save, args.data_src + '2city_round' + str(round_idx) + '_epoch' + str(epoch_idx+1)  + '.pth'))
#    torch.save(model.state_dict(), osp.join(save_path, args.data_src + '2city_round' + str(round_idx) + '_epoch' + str(epoch_idx+1)  + '.pth'))

    return writer_idx


def test(model, criterion, device, round_idx, tgt_set, test_num, test_list,
         label_2_id, valid_labels, args, logger, class_encoding, metric, writer, class_weights, reg_weight=0.0):
    """Create the model and start the evaluation process."""
    ## scorer
    scorer = ScoreUpdater(valid_labels, args.num_classes_seg, test_num, logger)
    scorer.reset()
    h, w = map(int, args.test_image_size.split(','))
    test_image_size = (h, w)
    test_size = ( h, w )
    test_scales = [float(_) for _ in str(args.test_scale).split(',')]
    num_scales = len(test_scales)

    # For logging the training status
    losses = AverageMeter()
    batch_time = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()

    miou_class = MIOU(num_classes=4)

    kld_layer = PixelwiseKLD()
    # TODO: Remove the composes
    if args.dataset == 'greenhouse':
        # TODO: Change dataset
        from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation

        ds = GreenhouseRGBDSegmentation(
            list_name=args.data_tgt_test_list, train=False, use_traversable=args.use_traversable, use_depth=args.use_depth)

#        testloader = data.DataLoader(ds, batch_size=1, shuffle=False, pin_memory=args.pin_memory)

    testloader = data.DataLoader(ds, batch_size=32, shuffle=False, pin_memory=args.pin_memory)

    ## model for evaluation
    if args.eval_training:
        model.train()
    else:
        model.eval()
    #
    model.to(device)

    ## upsampling layer
    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=test_image_size, mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=test_image_size, mode='bilinear')

    ## output of deeplab is logits, not probability
    softmax2d = nn.Softmax2d()

    ## evaluation process
    logger.info('###### Start evaluating in target domain {} set in round {}! ######'.format(tgt_set, round_idx))
    start_eval = time.time()
    total_loss = 0
    ious = 0

    # TODO: Change this (implement the same function in 'utility/utils.py', or uncomment the code below with slight modification)
#    total_loss, (iou, miou) = util.run_validation(model, testloader, criterion, metric, device, writer, interp)
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            #image, label, depth, name, reg_weights = batch

            images = batch[0].to(device)
            labels = batch[1].to(device)
            if args.use_depth:
                depths = batch[2].to(device)
                reg_weights = batch[4]
            else:
                reg_weights = batch[3]

            # Upsample the output of the model to the input size
            # TODO: Change the input according to the model type
            interp = None
            if interp is not None:
                if args.use_depth:
                    if args.model == 'espdnet':
                        pred = interp(model(images, depths))
                    elif args.model == 'espdnetue':
                        (pred, pred_aux) = interp(model(images, depths))
                else:
                    if args.model == 'espdnet':
                        pred = interp(model(images))
                    elif args.model == 'espdnetue':
                        (pred, pred_aux) = interp(model(images))
                    elif args.model == 'deeplabv3':
                        output = model(images)
                        pred = interp(output['out'])
                        pred_aux = interp(output['aux'])
            else:
                if args.use_depth:
                    if args.model == 'espdnet':
                        pred = model(images, depths)
                    elif args.model == 'espdnetue':
                        (pred, pred_aux) = model(images, depths)
                else:
                    if args.model == 'espdnet':
                        pred = model(images)
                    elif args.model == 'espdnetue':
                        (pred, pred_aux) = model(images)
                    elif args.model == 'deeplabv3':
                        output = model(images)
                        pred = output['out']
                        pred_aux = output['aux']

            loss = criterion(pred, labels) # torch.max returns a tuple of (maxvalues, indices)

            inter, union = miou_class.get_iou(pred, labels)
    
            inter_meter.update(inter)
            union_meter.update(union)
    
            losses.update(loss.item(), images.size(0))

#            for scale_idx in range(num_scales):
#                if version.parse(torch.__version__) > version.parse('0.4.0'):
#                    # Resize the images to do multi-scale testing
#                    #  * Only this block should be considered because my environment is >0.4.0
#                    image = F.interpolate(img, scale_factor=test_scales[scale_idx], mode='bilinear', align_corners=True)
#                else:
#                    test_size = (int(h * test_scales[scale_idx]), int(w * test_scales[scale_idx]))
#                    interp_tmp = nn.Upsample(size=test_size, mode='bilinear', align_corners=True)
#                    image = interp_tmp(img)
#
#                print("crst_seg")
#                print(image.size(), depth.size())
#                output2 = interp(model(image.to(device), depth.to(device)))
#                coutput = interp(output2).cpu().data[0].numpy()
     
                # Check "What if the image is flipped?"
#                if args.test_flipping:
                    # Output for the flipped image
                    #   image.numpy()[:,:,:,::-1] -> Flipped image
#                    output2 = model(
#                        torch.from_numpy(image.numpy()[:,:,:,::-1].copy()).to(device),
#                        torch.from_numpy(depth.numpy()[:,:,:,::-1].copy()).to(device))
#                    # Flip the output of flipped image (output2) and take avarages with the original output
#                    coutput = 0.5 * ( coutput + interp(output2).cpu().data[0].numpy()[:,:,::-1] )
#     
#                if scale_idx == 0:
#                    output = coutput.copy()
#                else:
#                    output = output + coutput
# 
#            output = output/num_scales

#            output = output.transpose(1,2,0)
#            amax_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            # score
#            pred_label = amax_output.copy()
#            label = label_2_id[np.asarray(label.numpy(), dtype=np.uint8)]
            # scorer.update(pred_label.flatten(), label.flatten(), index)
            # save visualized seg maps & predication prob map
#            amax_output_col = colorize_mask(amax_output)
#            name = name[0].split('/')[-1]
#            image_name = name.split('.')[0]
            # vis seg maps
#            amax_output_col.save('%s/%s_color.png' % (save_test_vis_path, image_name))



            # greenhouse_more has no ground truth, so the metrics are meaningless
#            if args.dataset == 'greenhouse':
#                label = np.asarray(label.numpy(), dtype=np.uint8)
#            else:
#                label = label_2_id[np.asarray(label.numpy(), dtype=np.uint8)]

#            pred = torch.from_numpy(output_c)

            #print(pred.size(), label.size())

            # Keep track of the evaluation metric
#            metric.add(output.detach(), label.detach())
#    
#            # Calculate IoU
#            (iou, miou) = metric.value()
#
#            ious += miou
#            total_loss += reg_loss_calc(output, label.to(device), reg_weights.to(device), args).item()
#
#            # save visualized seg maps & predication prob map
#            amax_output_col = colorize_mask(amax_output)
#            name = name[0].split('/')[-1]
#            image_name = name.rsplit('.', 1)[0]
            # vis seg maps
    #        amax_output_col.save('%s/%s_color.png' % (save_test_vis_path, image_name))

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou[[1, 2, 3]].mean() * 100
 
    writer.add_scalar('cbst_enet/test/mean_IoU', miou, round_idx)
    writer.add_scalar('cbst_enet/test/loss', losses.avg, round_idx)
    writer.add_scalar('cbst_enet/test/traversable_plant_IoU', iou[0], round_idx)
    writer.add_scalar('cbst_enet/test/other_plant_mean_IoU', iou[1], round_idx)
    logger.info('###### Finish evaluating in target domain {} set in round {}! Time cost: {:.2f} seconds. ######'.format(
        tgt_set, round_idx, time.time()-start_eval))

#    if args.dataset == 'greenhouse':
#        # TODO: Check
    if args.use_depth:
        # model, images, depths=None, labels=None, predictions=None, class_encoding=None, writer=None, epoch=None, data=None, device=None
        in_training_visualization_img(model, images=images, depths=depths, labels=labels, 
            class_encoding=class_encoding, writer=writer, epoch=round_idx, data='cbst_enet/test', device=device)
    else:
        in_training_visualization_img(model, images=images, labels=labels, class_encoding=class_encoding,
            writer=writer, epoch=round_idx, data='cbst_enet/test', device=device)

    return miou

def kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path, args, logger):
    logger.info('###### Start kc generation in round {} ! ######'.format(round_idx))
    start_kc = time.time()
    # threshold for each class
    conf_tot = 0.0
    #
    # If all labels are used, cls_sel_size etc. overflow with float32.
    # Therefore set it to float64
    #
    cls_thresh = np.ones(args.num_classes_seg, dtype = np.float64)
    cls_sel_size = np.zeros(args.num_classes_seg, dtype=np.float64)
    cls_size = np.zeros(args.num_classes_seg, dtype=np.float64)
    if args.kc_policy == 'cb' and args.kc_value == 'conf':
        for idx_cls in np.arange(0, args.num_classes_seg):
        # Set N_c (the number of pixels predicted as class c)
            print("kc : {}".format(idx_cls))
            cls_size[idx_cls] = pred_cls_num[idx_cls]
            #
            # conf_dict[idx_cls] : All the confidence values of idx_cls 
            #
            if conf_dict[idx_cls] is not None and idx_cls != args.ignore_label:
                conf_dict[idx_cls].sort(reverse=True) # sort in descending order
                len_cls = len(conf_dict[idx_cls])
                #
                # round(N_c*p) -> The number of pixels to be selected from all idx_cls pixels
                #
#                cls_sel_size[idx_cls] = int(math.floor(len_cls * tgt_portion))
                cls_sel_size[idx_cls] = math.floor(len_cls * tgt_portion)
                print(math.floor(len_cls * tgt_portion), int(math.floor(len_cls * tgt_portion)))
                len_cls_thresh = int(cls_sel_size[idx_cls])
                print(cls_sel_size[idx_cls], int(cls_sel_size[idx_cls]), len_cls_thresh)
                print(len(conf_dict[idx_cls]))

                if len_cls_thresh != 0:
                    print(math.floor(len_cls * tgt_portion), int(math.floor(len_cls * tgt_portion)), cls_sel_size[idx_cls], len_cls_thresh)
                    cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
   
                conf_dict[idx_cls] = None

    # threshold for mine_id with priority
    num_mine_id = len(np.nonzero(cls_size / np.sum(cls_size) < args.mine_port)[0])
    # chose the smallest mine_id
    id_all = np.argsort(cls_size / np.sum(cls_size))
    rare_id = id_all[:args.rare_cls_num]
    mine_id = id_all[:num_mine_id] # sort mine_id in ascending order w.r.t predication portions
    # save mine ids
    np.save(save_stats_path + '/rare_id_round' + str(round_idx) + '.npy', rare_id)
    np.save(save_stats_path + '/mine_id_round' + str(round_idx) + '.npy', mine_id)
    logger.info('Mining ids : {}! {} rarest ids: {}!'.format(mine_id,args.rare_cls_num,rare_id))
    # save thresholds
    np.save(save_stats_path + '/cls_thresh_round' + str(round_idx) + '.npy', cls_thresh)
    np.save(save_stats_path + '/cls_sel_size_round' + str(round_idx) + '.npy', cls_sel_size)
    logger.info('###### Finish kc generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx,time.time() - start_kc))

    return cls_thresh

def label_selection(
        cls_thresh, tgt_num, image_name_tgt_list, id_2_label, round_idx, save_prob_path,
        save_pred_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args, logger):
    logger.info('###### Start pseudo-label generation in round {} ! ######'.format(round_idx))
    start_pl = time.time()

    # For all the data in target set
    with tqdm(total=tgt_num) as pbar:
        for idx in tqdm(range(tgt_num)):
            # Image name without an extension
            sample_name = image_name_tgt_list[idx].rsplit('.', 1)[0]
            probmap_path = osp.join(save_prob_path, '{}.npy'.format(sample_name))
            pred_path = osp.join(save_pred_path, '{}.png'.format(sample_name))
    
            # Prediction probabilities saved in val()
            pred_prob = np.load(probmap_path)
    
            # Prediction result. It returns train IDs
            pred_label_trainIDs = np.asarray(Image.open(pred_path))
            # Map the predicted train IDs to the corresponding label IDs
            pred_label_labelIDs = id_2_label[pred_label_trainIDs]
    
            # ???
            pred_label_trainIDs = pred_label_trainIDs.copy()
            # Normal CBST?
            if args.kc_policy == 'cb' and args.lr_weight_ent == 0.0:
                save_wpred_vis_path = osp.join(save_round_eval_path, 'weighted_pred_vis')
                if not os.path.exists(save_wpred_vis_path):
                    os.makedirs(save_wpred_vis_path)
    
                # Weight the probability of the threshold
                weighted_prob = pred_prob/cls_thresh
                weighted_pred_trainIDs = np.asarray(np.argmax(weighted_prob, axis=2), dtype=np.uint8)
                # save weighted predication
                wpred_label_col = weighted_pred_trainIDs.copy()
                wpred_label_col = colorize_mask(wpred_label_col)
                wpred_label_col.save('%s/%s_color.png' % (save_wpred_vis_path, sample_name))
                # ID of maximum output
                weighted_conf = np.amax(weighted_prob, axis=2)
                pred_label_trainIDs = weighted_pred_trainIDs.copy()
                pred_label_labelIDs = id_2_label[pred_label_trainIDs]
                pred_label_labelIDs[weighted_conf < 1] = 4 #0  # '0' in cityscapes indicates 'unlabaled' for labelIDs
                pred_label_trainIDs[weighted_conf < 1] = 4 # 255 # '255' in cityscapes indicates 'unlabaled' for trainIDs
            elif args.kc_policy == 'cb' and args.lr_weight_ent > 0.0: # check if cb can be combined with kc_value == conf or prob; also check if \alpha can be larger than 1
                save_wpred_vis_path = osp.join(save_round_eval_path, 'weighted_pred_vis')
                if not os.path.exists(save_wpred_vis_path):
                    os.makedirs(save_wpred_vis_path)
                # soft pseudo-label
                soft_pseudo_label = np.power(pred_prob/cls_thresh,1.0/args.lr_weight_ent) # weighted softmax with temperature
                soft_pseudo_label_sum = soft_pseudo_label.sum(2)
                soft_pseudo_label = soft_pseudo_label.transpose(2,0,1)/soft_pseudo_label_sum
                soft_pseudo_label = soft_pseudo_label.transpose(1,2,0).astype(np.float32)
                np.save('%s/%s.npy' % (save_pseudo_label_path, sample_name), soft_pseudo_label)
                # hard pseudo-label
                weighted_pred_trainIDs = np.asarray(np.argmax(soft_pseudo_label, axis=2), dtype=np.uint8)
                reg_score = np.sum(
                        -soft_pseudo_label*np.log(pred_prob+1e-32) + args.lr_weight_ent*soft_pseudo_label*np.log(soft_pseudo_label+1e-32),
                        axis=2)
                sel_score = np.sum( -soft_pseudo_label*np.log(cls_thresh+1e-32), axis=2)
                # save weighted predication
                wpred_label_col = weighted_pred_trainIDs.copy()
                wpred_label_col = colorize_mask(wpred_label_col)
                wpred_label_col.save('%s/%s_color.png' % (save_wpred_vis_path, sample_name))
                pred_label_trainIDs = weighted_pred_trainIDs.copy()
                pred_label_labelIDs = id_2_label[pred_label_trainIDs]
                pred_label_labelIDs[reg_score >= sel_score] = 4 #0  # '0' in cityscapes indicates 'unlabaled' for labelIDs
                pred_label_trainIDs[reg_score >= sel_score] = 4 #255 # '255' in cityscapes indicates 'unlabaled' for trainIDs
    
            # pseudo-labels with labelID
            pseudo_label_labelIDs = pred_label_labelIDs.copy()
            pseudo_label_trainIDs = pred_label_trainIDs.copy()
            # save colored pseudo-label map
            pseudo_label_col = colorize_mask(pseudo_label_trainIDs)
            pseudo_label_col.save('%s/%s_color.png' % (save_pseudo_label_color_path, sample_name))
            # save pseudo-label map with label IDs
            pseudo_label_save = Image.fromarray(pseudo_label_labelIDs.astype(np.uint8))
            pseudo_label_save.save('%s/%s.png' % (save_pseudo_label_path, sample_name))

    pbar.close()

    # remove probability maps
    if args.rm_prob:
        shutil.rmtree(save_prob_path)

    logger.info('###### Finish pseudo-label generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx,time.time() - start_pl))

def parse_split_list(list_name):
    image_list = []
    image_name_list = []
    label_list = []
    depth_list = [] if args.use_depth else None
    file_num = 0
    with open(list_name) as f:
        for item in f.readlines():
            # strip() : Remove spaces at the begining and the end of the line
            #  fields[0] : Image name
            #  fields[1] : Label name
            fields = item.strip().split(',')
            # Image name, without file path
            # Negative index of arrays -> from the end of the array
            image_name = fields[0].split('/')[-1]
            image_list.append(fields[0])
            image_name_list.append(image_name)
            label_list.append(fields[1])
            if depth_list is not None and len(fields) == 3:
                depth_list.append(fields[2])

            file_num += 1

    return image_list, image_name_list, label_list, depth_list, file_num


def savelst_SrcTgt(
        src_portion, image_tgt_list, depth_tgt_list, image_name_tgt_list,
        image_src_list, label_src_list, depth_src_list,
        save_lst_path, save_pseudo_label_path, src_num, tgt_num, randseed, args):
    src_num_sel = int(np.floor(src_num*src_portion))
    np.random.seed(randseed)
    sel_idx = list( np.random.choice(src_num, src_num_sel, replace=False) )
    sel_src_img_list = list( itemgetter(*sel_idx)(image_src_list) )
    sel_src_label_list = list(itemgetter(*sel_idx)(label_src_list))
    if depth_src_list is not None:
        sel_src_depth_list = list(itemgetter(*sel_idx)(depth_src_list))
    src_train_lst = osp.join(save_lst_path,'src_train.lst')
    tgt_train_lst = osp.join(save_lst_path, 'tgt_train.lst')

    # generate src train list
    with open(src_train_lst, 'w') as f:
        for idx in range(src_num_sel):
            if depth_src_list is not None:
                f.write("%s,%s,%s\n" % (sel_src_img_list[idx], sel_src_label_list[idx], sel_src_depth_list[idx]))
            else:
                f.write("%s,%s\n" % (sel_src_img_list[idx], sel_src_label_list[idx]))

    # generate tgt train list
    if args.lr_weight_ent > 0:
        with open(tgt_train_lst, 'w') as f:
            for idx in range(tgt_num):
                softlabel_name = image_name_tgt_list[idx].rsplit('.', 1)[0] + '.npy'
                soft_label_tgt_path = osp.join(save_pseudo_label_path, softlabel_name)
                image_tgt_path = osp.join(save_pseudo_label_path,image_name_tgt_list[idx])
                f.write("%s\t%s\t%s\n" % (image_tgt_list[idx], image_tgt_path, soft_label_tgt_path))
    if args.lr_weight_ent == 0:
        with open(tgt_train_lst, 'w') as f:
            for idx in range(tgt_num):
                image_tgt_path = osp.join(save_pseudo_label_path, image_name_tgt_list[idx])
                if depth_tgt_list is not None:
                    f.write("%s,%s,%s\n" % (image_tgt_list[idx], image_tgt_path, depth_tgt_list[idx]))
                else:
                    f.write("%s,%s\n" % (image_tgt_list[idx], image_tgt_path))

    return src_train_lst, tgt_train_lst, src_num_sel

def load_weights(model, weights):
    if os.path.isfile(weights):
        num_gpus = torch.cuda.device_count()
        device = 'cuda' if num_gpus >= 1 else 'cpu'
        pretrained_dict = torch.load(weights, map_location=torch.device(device))
    else:
        print('Weight file does not exist at {}. Please check. Exiting!!'.format(weights))
        exit()

    model_dict = model.state_dict()
    overlap_dict = {k: v for k, v in pretrained_dict.items() 
                    if k in model_dict}

    model_dict.update(overlap_dict)
    model.load_state_dict(model_dict)

class ScoreUpdater(object):
    # only IoU are computed. accu, cls_accu, etc are ignored.
    def __init__(self, valid_labels, c_num, x_num, logger=None, label=None, info=None):
        self._valid_labels = valid_labels

        self._confs = np.zeros((c_num, c_num))
        self._per_cls_iou = np.zeros(c_num)
        self._logger = logger
        self._label = label
        self._info = info
        self._num_class = c_num
        self._num_sample = x_num

    @property
    def info(self):
        return self._info

    def reset(self):
        self._start = time.time()
        self._computed = np.zeros(self._num_sample) # one-dimension
        self._confs[:] = 0

    def fast_hist(self,label, pred_label, n):
        k = (label >= 0) & (label < n)
        print("fast_hist")
        print("n : {} pred_label : {} ".format(n, pred_label))
        return np.bincount(n * label[k].astype(int) + pred_label[k], minlength=n ** 2).reshape(n, n)

    def per_class_iu(self,hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def do_updates(self, conf, i, computed=True):
        if computed:
            self._computed[i] = 1
        self._per_cls_iou = self.per_class_iu(conf)

    def update(self, pred_label, label, i, computed=True):
        conf = self.fast_hist(label, pred_label, self._num_class)
        self._confs += conf
        self.do_updates(self._confs, i, computed)
        self.scores(i)

    def scores(self, i=None, logger=None):
        x_num = self._num_sample
        ious = np.nan_to_num( self._per_cls_iou )

        logger = self._logger if logger is None else logger
        if logger is not None:
            if i is not None:
                speed = 1. * self._computed.sum() / (time.time() - self._start)
                logger.info('Done {}/{} with speed: {:.2f}/s'.format(i + 1, x_num, speed))
            name = '' if self._label is None else '{}, '.format(self._label)
            logger.info('{}mean iou: {:.2f}%'. \
                        format(name, np.mean(ious) * 100))
            with util.np_print_options(formatter={'float': '{:5.2f}'.format}):
                logger.info('\n{}'.format(ious * 100))

        return ious

# Adjust learning rate based on the number iteration
def lr_poly(base_lr, iter_n, max_iter, power):
    lr = base_lr * ((1 - float(iter_n) / max_iter) ** (power))

#    return base_lr * ((1 - float(iter_n) / max_iter) ** (power))
    return lr

def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def adjust_learning_rate(optimizer, i_iter, tot_iter):
    lr = lr_poly(args.learning_rate, i_iter, tot_iter, args.power)
    optimizer.param_groups[0]['lr'] = lr
    # optimizer.param_groups[1]['lr'] = lr * 10

def transfer_id_to_greenhouse(id_to_greenhouse, output_amax_np):

    return id_to_greenhouse[output_amax_np]

def transfer_output_to_greenhouse(id_to_greenhouse, output_np):
   # output_greenhouse_np = np.array([]) #args.num_classes_seg
    output_shape = (1, output_np.shape[1], output_np.shape[2])

    # ID 0:traversable plant is not transfered from CamVid
    output_greenhouse_np = np.zeros(output_shape)
    for gh_class_id in range(1, args.num_classes_seg):
        bool_index = id_to_greenhouse == gh_class_id
#        print(bool_index.shape)
        if bool_index.sum():
            output_gh_class = output_np[bool_index].max(axis=0).reshape(output_shape)
        else:
            output_gh_class = np.zeros(output_shape)

        output_greenhouse_np = np.append(output_greenhouse_np, output_gh_class, axis=0)

    return output_greenhouse_np

if __name__ == '__main__':
    main()
