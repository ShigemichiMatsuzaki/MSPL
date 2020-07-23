# ============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
# ============================================

import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from utilities.utils import save_checkpoint, model_parameters, compute_flops, in_training_visualization_img, calc_cls_class_weight
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter

from loss_fns.segmentation_loss import SegmentationLoss, NIDLoss
import random
import math
import time
import numpy as np
from utilities.print_utils import *
from collections import OrderedDict

from utilities.utils import AverageMeter
from utilities.metrics.segmentation_miou import MIOU
from data_loader.segmentation.greenhouse import id_camvid_to_greenhouse
from data_loader.segmentation.greenhouse import id_cityscapes_to_greenhouse

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

def main(args):
    crop_size = args.crop_size
    assert isinstance(crop_size, tuple)
    print_info_message('Running Model at image resolution {}x{} with batch size {}'.format(crop_size[0], crop_size[1],
                                                                                           args.batch_size))
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    writer = SummaryWriter(log_dir=args.savedir)

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'

    from data_loader.segmentation.greenhouse import color_encoding as color_encoding_greenhouse
    from data_loader.segmentation.camvid import color_encoding as color_encoding_camvid
    if args.dataset == 'pascal':
        from data_loader.segmentation.voc import VOCSegmentation, VOC_CLASS_LIST
        train_dataset = VOCSegmentation(root=args.data_path, train=True, crop_size=crop_size, scale=args.scale,
                                        coco_root_dir=args.coco_path)
        val_dataset = VOCSegmentation(root=args.data_path, train=False, crop_size=crop_size, scale=args.scale)
    elif args.dataset == 'city':
        from data_loader.segmentation.cityscapes import CityscapesSegmentation, CITYSCAPE_CLASS_LIST
        train_dataset = CityscapesSegmentation(root=args.data_path, train=True, coarse=False)
        val_dataset = CityscapesSegmentation(root=args.data_path, train=False, coarse=False)

        seg_classes = len(CITYSCAPE_CLASS_LIST)
    elif args.dataset == 'greenhouse':
        print(args.use_depth)
        from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation, GREENHOUSE_CLASS_LIST
        train_dataset = GreenhouseRGBDSegmentation(root=args.data_path, list_name='train_greenhouse_gt.txt', train=True, size=crop_size, scale=args.scale, use_depth=args.use_depth)
        val_dataset = GreenhouseRGBDSegmentation(root=args.data_path, list_name='val_greenhouse.txt', train=False, size=crop_size, scale=args.scale, use_depth=args.use_depth)
        class_weights = np.load('class_weights.npy')# [:4]
        print(class_weights)
        class_wts = torch.from_numpy(class_weights).float().to(device)

        print(GREENHOUSE_CLASS_LIST)
        seg_classes = len(GREENHOUSE_CLASS_LIST)
    elif args.dataset == 'ishihara':
        print(args.use_depth)
        from data_loader.segmentation.ishihara_rgbd import IshiharaRGBDSegmentation, ISHIHARA_RGBD_CLASS_LIST
        train_dataset = IshiharaRGBDSegmentation(root=args.data_path, list_name='ishihara_rgbd_train.txt', train=True, size=crop_size, scale=args.scale, use_depth=args.use_depth)
        val_dataset = IshiharaRGBDSegmentation(root=args.data_path, list_name='ishihara_rgbd_val.txt', train=False, size=crop_size, scale=args.scale, use_depth=args.use_depth)

        seg_classes = len(ISHIHARA_RGBD_CLASS_LIST)
    elif args.dataset == 'sun':
        print(args.use_depth)
        from data_loader.segmentation.sun_rgbd import SUNRGBDSegmentation, SUN_RGBD_CLASS_LIST
        train_dataset = SUNRGBDSegmentation(root=args.data_path, list_name='sun_rgbd_train.txt', train=True, size=crop_size, ignore_idx=args.ignore_idx, scale=args.scale, use_depth=args.use_depth)
        val_dataset = SUNRGBDSegmentation(root=args.data_path, list_name='sun_rgbd_val.txt', train=False, size=crop_size, ignore_idx=args.ignore_idx, scale=args.scale, use_depth=args.use_depth)

        seg_classes = len(SUN_RGBD_CLASS_LIST)
    elif args.dataset == 'camvid':
        print(args.use_depth)
        from data_loader.segmentation.camvid import CamVidSegmentation, CAMVID_CLASS_LIST
        train_dataset = CamVidSegmentation(
            root=args.data_path, list_name='train_camvid.txt', 
            train=True, size=crop_size, scale=args.scale, label_conversion=args.label_conversion, normalize=args.normalize)
        val_dataset = CamVidSegmentation(
            root=args.data_path, list_name='val_camvid.txt',
            train=False, size=crop_size, scale=args.scale, label_conversion=args.label_conversion, normalize=args.normalize)
        
        seg_classes = len(CAMVID_CLASS_LIST)

        args.use_depth = False
    else:
        print_error_message('Dataset: {} not yet supported'.format(args.dataset))
        exit(-1)

    print_info_message('Training samples: {}'.format(len(train_dataset)))
    print_info_message('Validation samples: {}'.format(len(val_dataset)))

    if args.model == 'espnetv2':
        from model.segmentation.espnetv2 import espnetv2_seg
        args.classes = seg_classes
        model = espnetv2_seg(args)
    elif args.model == 'espdnet':
        from model.segmentation.espdnet import espdnet_seg_with_pre_rgbd
        args.classes = seg_classes
        print("Trainable fusion : {}".format(args.trainable_fusion))
        print("Segmentation classes : {}".format(seg_classes))
        model = espdnet_seg_with_pre_rgbd(args, load_entire_weights=True)
    elif args.model == 'espdnetue':
        from model.segmentation.espdnet_ue import espdnetue_seg2
        args.classes = seg_classes
        print("Trainable fusion : {}".format(args.trainable_fusion))
        print("Segmentation classes : {}".format(seg_classes))
        model = espdnetue_seg2(args, load_entire_weights=True, fix_pyr_plane_proj=True)
    elif args.model == 'deeplabv3':
        # from model.segmentation.deeplabv3 import DeepLabV3
        from torchvision.models.segmentation.segmentation import deeplabv3_resnet101

        args.classes = seg_classes
        # model = DeepLabV3(seg_classes)
        model = deeplabv3_resnet101(num_classes=seg_classes, aux_loss=True)
        torch.backends.cudnn.enabled = False
        load_weights(model, args.finetune)

    elif args.model == 'dicenet':
        from model.segmentation.dicenet import dicenet_seg
        model = dicenet_seg(args, classes=seg_classes)
    else:
        print_error_message('Arch: {} not yet supported'.format(args.model))
        exit(-1)

    model.to(device=device)
    from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation, GREENHOUSE_CLASS_LIST
    val_dataset = GreenhouseRGBDSegmentation(root='./vision_datasets/greenhouse/', list_name='val_greenhouse_more.lst', use_traversable=False, 
                                             train=False, size=crop_size, scale=args.scale, use_depth=args.use_depth,
                                             normalize=args.normalize)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)

    start_epoch = 0
    best_miou = 0.0
    losses = AverageMeter()
    ce_losses = AverageMeter()
    nid_losses = AverageMeter()
    batch_time = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    miou_class = MIOU(num_classes=seg_classes)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch[0].to(device=device)
            target = batch[1].to(device=device)
            
            outputs = model(inputs)

            if isinstance(outputs, OrderedDict):
                outputs = outputs['out'] + 0.5 * outputs['aux']
            elif isinstance(outputs, (list, tuple)):
                outputs = outputs[0] + 0.5 * outputs[1]

            _, outputs_argmax = torch.max(outputs, 1)
            outputs_argmax = outputs_argmax.cpu().numpy()

            if args.dataset == 'camvid':
                outputs_argmax = id_camvid_to_greenhouse[outputs_argmax]
#                target = id_camvid_to_greenhouse[target.cpu().numpy()]
#                target = torch.from_numpy(target)
            elif args.dataset == 'city':
                outputs_argmax = id_cityscapes_to_greenhouse[outputs_argmax]
#                target = id_cityscapes_to_greenhouse[target.cpu().numpy()]
#                target = torch.from_numpy(target)

            outputs_argmax = torch.from_numpy(outputs_argmax)
            
            inter, union = miou_class.get_iou(outputs_argmax, target)
            inter_meter.update(inter)
            union_meter.update(union)

            # measure elapsed time
            #batch_time.update(time.time() - end)
            #end = time.time()


            in_training_visualization_img(
                model, 
                images=batch[0].to(device=device),
                labels=target,
                predictions=outputs_argmax,
                class_encoding=color_encoding_greenhouse,
                writer=writer,
                data='label_eval',
                device=device)

            print("Batch {}/{} finished".format(i+1, len(val_loader)))
    
    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou[[1, 2, 3]].mean() * 100
    writer.add_scalar('label_eval/IoU', miou, 0)

    writer.close()


if __name__ == "__main__":
    from commons.general_details import segmentation_models, segmentation_schedulers, segmentation_loss_fns, \
        segmentation_datasets

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--ignore-idx', type=int, default=255, help='Index or label to be ignored during training')

    # model details
    parser.add_argument('--freeze-bn', action='store_true', default=False, help='Freeze BN params or not')

    # dataset and result directories
    parser.add_argument('--dataset', type=str, default='pascal', choices=segmentation_datasets, help='Datasets')
    parser.add_argument('--data-path', type=str, default='', help='dataset path')
    parser.add_argument('--coco-path', type=str, default='', help='MS COCO dataset path')
    parser.add_argument('--savedir', type=str, default='./results_segmentation', help='Location to save the results')
    ## only for cityscapes
    parser.add_argument('--coarse', action='store_true', default=False, help='Want to use coarse annotations or not')

    # scheduler details
    parser.add_argument('--scheduler', default='hybrid', choices=segmentation_schedulers,
                        help='Learning rate scheduler (fixed, clr, poly)')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--step-size', default=51, type=int, help='steps at which lr should be decreased')
    parser.add_argument('--lr', default=9e-3, type=float, help='initial learning rate')
    parser.add_argument('--lr-mult', default=10.0, type=float, help='initial learning rate')
    parser.add_argument('--lr-decay', default=0.5, type=float, help='factor by which lr should be decreased')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=4e-5, type=float, help='weight decay (default: 4e-5)')
    # for Polynomial LR
    parser.add_argument('--power', default=0.9, type=float, help='power factor for Polynomial LR')

    # for hybrid LR
    parser.add_argument('--clr-max', default=61, type=int, help='Max number of epochs for cylic LR before '
                                                                'changing last cycle to linear')
    parser.add_argument('--cycle-len', default=5, type=int, help='Duration of cycle')

    # input details
    parser.add_argument('--batch-size', type=int, default=40, help='list of batch sizes')
    parser.add_argument('--crop-size', type=int, nargs='+', default=[480, 256],
                        help='list of image crop sizes, with each item storing the crop size (should be a tuple).')
    parser.add_argument('--loss-type', default='ce', choices=segmentation_loss_fns, help='Loss function (ce or miou)')

    # model related params
    parser.add_argument('--s', type=float, default=2.0, help='Factor by which channels will be scaled')
    parser.add_argument('--model', default='espnet', choices=segmentation_models,
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
    parser.add_argument('--label-conversion', default=False, type=bool, help='Use label conversion in CamVid')
    parser.add_argument('--use-nid', default=False, type=bool, help='Use NID loss')
    parser.add_argument('--use-aux', default=False, type=bool, help='Use auxiliary loss')
    parser.add_argument('--normalize', default=False, type=bool, help='Use auxiliary loss')
#    parser.add_argument('--suffix', default='', type=str, help='Suffix of the save directory')

    args = parser.parse_args()

    random.seed(1882)
    torch.manual_seed(1882)

    if args.dataset == 'pascal':
        args.scale = (0.5, 2.0)
    elif args.dataset == 'city':
        if args.crop_size[0] == 512:
            args.scale = (0.25, 0.5)
        elif args.crop_size[0] == 1024:
            args.scale = (0.35, 1.0)  # 0.75 # 0.5 -- 59+
        elif args.crop_size[0] == 2048:
            args.scale = (1.0, 2.0)
        else:
            print_error_message('Select image size from 512x256, 1024x512, 2048x1024')
        print_log_message('Using scale = ({}, {})'.format(args.scale[0], args.scale[1]))
    elif args.dataset == 'greenhouse':
        args.scale = (0.5, 2.0)
    elif args.dataset == 'ishihara':
        args.scale = (0.5, 2.0)
    elif args.dataset == 'sun':
        args.scale = (0.5, 2.0)
    elif args.dataset == 'camvid':
        args.scale = (0.5, 2.0)
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    if not args.finetune:
        from model.weight_locations.classification import model_weight_map

        if args.model == 'espdnet' or args.model == 'espdnetue':
            weight_file_key = '{}_{}'.format('espnetv2', args.s)
            assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
            args.weights = model_weight_map[weight_file_key]
        elif args.model == 'deeplabv3':
            args.weights  = ''

    #        if args.use_depth:
    #            args.weights
        else:
            weight_file_key = '{}_{}'.format(args.model, args.s)
            assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
            args.weights = model_weight_map[weight_file_key]
    else:
        assert os.path.isfile(args.finetune), '{} weight file does not exist'.format(args.finetune)
        args.weights = args.finetune

    assert len(args.crop_size) == 2, 'crop-size argument must contain 2 values'
    assert args.data_path != '', 'Dataset path is an empty string. Please check.'

    args.crop_size = tuple(args.crop_size)
    #timestr = time.strftime("%Y%m%d-%H%M%S")
    now = datetime.datetime.now()
    now += datetime.timedelta(hours=9)
    timestr = now.strftime("%Y%m%d-%H%M%S")

    suffix = args.finetune.rsplit('/', 2)[1]
    args.savedir = '{}/{}_{}_{}_{}'.format(
        args.savedir, args.model, args.dataset, suffix, timestr)

    main(args)

