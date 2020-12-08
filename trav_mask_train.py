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

import math
from PIL import Image
import numpy as np
import shutil
import random

from utilities.utils import save_checkpoint, model_parameters, compute_flops, in_training_visualization_img, set_logger, calc_cls_class_weight, import_os_model
from utilities.utils import AverageMeter
from utilities.metrics.segmentation_miou import MIOU
#from utilities.train_eval_seg import train_seg as train
#from utilities.train_eval_seg import val_seg as val
from loss_fns.segmentation_loss import NIDLoss, UncertaintyWeightedSegmentationLoss, PixelwiseKLD, SegmentationLoss

###
# Matsuzaki
###
from torch.utils.tensorboard import SummaryWriter
#from metric.iou import IoU
from tqdm import tqdm

# Default arguments
RESTORE_FROM = './src_model/gta5/src_model.pth'
GPU = 0
PIN_MEMORY = False
BATCH_SIZE = 32
INPUT_SIZE = '256,480'# 512,1024 for GTA;
LEARNING_RATE = 0.00005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
OPTIMIZER = 'Adam'
EPOCH=200
SAVE_PATH = '/tmp/runs'

TRAVERSED = 1
TRAVERSABLE_PLANT = 0
PLANT = 1

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Training using traversability masks")
    ### shared by train & val
    # data
    parser.add_argument('--savedir', type=str, default='./results_segmentation', help='Location to save the results')
    parser.add_argument('--data-trav-list', type=str, default='./vision_datasets/traversability_mask/greenhouse_b_train.lst',
                        help='Location to save the results')
    parser.add_argument('--data-test-list', type=str, default='./vision_datasets/greenhouse/val_greenhouse2.lst',
                        help='Location to save the results')
    # model
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    # gpu
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--pin-memory", type=bool, default=PIN_MEMORY,
                        help="Whether to pin memory in train & eval.")
    ### train ###
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")

    parser.add_argument('--crop-size', type=int, nargs='+', default=[480, 256],
                        help='list of image crop sizes, with each item storing the crop size (should be a tuple).')


    # params for optimizor
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER,
                        help="Optimizer used in the training")
    ### self-training params
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result for self-training.")
    parser.add_argument("--epoch", type=int, default=EPOCH,
                        help="Number of epochs")

    # model related params
    parser.add_argument('--model', default='espdnetue',
                        help='Which model? basic= basic CNN model, res=resnet style)')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')
    parser.add_argument('--finetune', default='', type=str, help='Finetune the segmentation model')
    parser.add_argument('--dataset', default='greenhouse', type=str, help='Type of the dataset')
    parser.add_argument('--model-width', default=480, type=int, help='Model width')
    parser.add_argument('--model-height', default=256, type=int, help='Model height')
    parser.add_argument('--s', type=float, default=2.0, help='Factor by which channels will be scaled')
    parser.add_argument('--use-depth', default=False, type=bool, help='Use depth')
    parser.add_argument('--trainable-fusion', default=False, type=bool, help='Use depth')
    parser.add_argument('--dense-fuse', default=False, type=bool, help='Use depth')

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

def main():
    device = 'cuda'

    now = datetime.datetime.now()
    now += datetime.timedelta(hours=9)
    timestr = now.strftime("%Y%m%d-%H%M%S")
    use_depth_str = "_rgbd" if args.use_depth else "_rgb"
    if args.use_depth:
        trainable_fusion_str = "_gated" if args.trainable_fusion else "_naive"
    else:
        trainable_fusion_str = ""

    save_path = '{}/model_{}_{}/{}'.format(
        args.save, 
        args.model, args.dataset, 
        timestr)

    print(save_path)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    tgt_train_lst = osp.join(save_path, 'tgt_train.lst')
    save_pred_path = osp.join(save_path, 'pred')
    if not os.path.isdir(save_pred_path):
        os.makedirs(save_pred_path)
    writer = SummaryWriter(save_path)

    # Dataset
    from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentationTrav, GREENHOUSE_CLASS_LIST
    args.classes = len(GREENHOUSE_CLASS_LIST)
    travset = GreenhouseRGBDSegmentationTrav(list_name=args.data_trav_list, use_depth=args.use_depth)

    class_encoding = OrderedDict([
        ('end_of_plant', (0, 255, 0)),
        ('other_part_of_plant', (0, 255, 255)),
        ('artificial_objects', (255, 0, 0)),
        ('ground', (255, 255, 0)),
        ('background', (0, 0, 0))
    ])

    # Dataloader for generating the pseudo-labels
    travloader = torch.utils.data.DataLoader(
        travset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=args.pin_memory)

    # Model
    from model.segmentation.espdnet_ue import espdnetue_seg2
    args.weights = args.restore_from
    model = espdnetue_seg2(args, load_entire_weights=True, fix_pyr_plane_proj=True)
    model.to(device)

    generate_label(model, travloader, save_pred_path, tgt_train_lst)
    # Datset for training
    from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation
    trainset = GreenhouseRGBDSegmentation(list_name=tgt_train_lst, use_depth=args.use_depth, use_traversable=True)
    testset  = GreenhouseRGBDSegmentation(list_name=args.data_test_list, use_depth=args.use_depth, use_traversable=True)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=args.pin_memory)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=args.pin_memory)

    # Loss
    if args.use_uncertainty:
        criterion = UncertaintyWeightedSegmentationLoss(args.classes)
    else:
        criterion = SegmentationLoss(n_classes=args.classes, device=device)

    criterion_test = SegmentationLoss(n_classes=args.classes, device=device)

    # Optimizer
    if args.use_depth:
        train_params = [{'params': model.get_basenet_params(), 'lr': args.learning_rate * 0.1},
                        {'params': model.get_segment_params(), 'lr': args.learning_rate},
                        {'params': model.get_depth_encoder_params(), 'lr': args.learning_rate}]
    else:
        train_params = [{'params': model.get_basenet_params(), 'lr': args.learning_rate * 0.1},
                        {'params': model.get_segment_params(), 'lr': args.learning_rate}]

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            train_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)            
    else:
        optimizer = optim.Adam(
            train_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.learning_rate, max_lr=args.learning_rate*10,
                                     step_size_up=10, step_size_down=20, cycle_momentum=True if args.optimizer == 'SGD' else False)
#    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)

    best_miou = 0.0
    for i in range(0, args.epoch):

        # Run a training epoch
        train(trainloader, model, criterion, device, optimizer, class_encoding, i, writer=writer)

        # Update the learning rate
        scheduler.step()
        # set the optimizer with the learning rate
        # This can be done inside the MyLRScheduler
#        optimizer.param_groups[0]['lr'] = lr_base
#        if len(optimizer.param_groups) > 1:
#            optimizer.param_groups[1]['lr'] = lr_seg
#        if args.use_depth:
#            optimizer.param_groups[2]['lr'] = lr_base * 10

        new_miou = test(testloader, model, criterion_test, device, optimizer, class_encoding, i, writer=writer)

        # Save the weights if it produces the best IoU
        is_best = new_miou > best_miou
        best_miou = max(new_miou, best_miou)
        model.to(device)
#        weights_dict = model.module.state_dict() if device == 'cuda' else model.state_dict()
        weights_dict = model.state_dict()
        extra_info_ckpt = '{}'.format(args.model)
        if is_best:
            save_checkpoint({
                'epoch': i + 1,
                'arch': args.model,
                'state_dict': weights_dict,
                'best_miou': best_miou,
                'optimizer': optimizer.state_dict(),
            }, is_best, save_path, extra_info_ckpt)


def train(trainloader, model, criterion, device, optimizer, 
          class_encoding, writer_idx, class_weights=None, writer=None):
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

    model.train()

    kld_layer = PixelwiseKLD()
    with tqdm(total=len(trainloader)) as pbar:
        for i_iter, batch in enumerate(tqdm(trainloader)):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            if args.use_depth:
                depths = batch[2].to(device)
    
    
            optimizer.zero_grad()

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

            inter, union = miou_class.get_iou(pred, labels)
    
            inter_meter.update(inter)
            union_meter.update(union)
    
            losses.update(loss.item(), images.size(0))
    
            # Optimise
            loss.backward()
            optimizer.step()



    
    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100

    # Write summary
    writer.add_scalar('traversability_mask/train/loss', losses.avg, writer_idx)
    writer.add_scalar('traversability_mask/train/nid_loss', nid_losses.avg, writer_idx)
    writer.add_scalar('traversability_mask/train/mean_IoU', miou, writer_idx)
    writer.add_scalar('traversability_mask/train/traversable_plant_IoU', iou[0], writer_idx)
    writer.add_scalar('traversability_mask/train/other_plant_mean_IoU', iou[1], writer_idx)
    writer.add_scalar('traversability_mask/train/artificial_object_mean_IoU', iou[2], writer_idx)
    writer.add_scalar('traversability_mask/train/ground_mean_IoU', iou[3], writer_idx)
    writer.add_scalar('traversability_mask/train/learning_rate', optimizer.param_groups[0]['lr'], writer_idx)
#    writer.add_scalar('uest/train/kld', kld_losses.avg, writer_idx)
  
    #
    # Investigation of labels
    #
    
    # Before label conversion
    if args.use_depth: 
        # model, images, depths=None, labels=None, predictions=None, class_encoding=None, writer=None, epoch=None, data=None, device=None
        in_training_visualization_img(model, images=images, depths=depths, labels=labels.long(), class_encoding=class_encoding, 
            writer=writer, epoch=writer_idx, data='traversability_mask/train', device=device)
    else:
        in_training_visualization_img(model, images=images, labels=labels.long(), class_encoding=class_encoding, 
            writer=writer, epoch=writer_idx, data='traversability_mask/train', device=device)

    
    writer_idx += 1
    
    print('taking snapshot ...')

    return writer_idx

def test(testloader, model, criterion, device, optimizer, 
         class_encoding, writer_idx, class_weights=None, writer=None):
    """Create the model and start the evaluation process."""
    ## scorer
    h, w = map(int, args.input_size.split(','))
    test_image_size = (h, w)

    # For logging the training status
    losses = AverageMeter()
    batch_time = AverageMeter()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()

    miou_class = MIOU(num_classes=4)

    kld_layer = PixelwiseKLD()

    from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation
    ds = GreenhouseRGBDSegmentation(
        list_name=args.data_test_list, train=False, use_traversable=True, use_depth=args.use_depth)

    testloader = data.DataLoader(ds, batch_size=32, shuffle=False, pin_memory=args.pin_memory)

    ## model for evaluation
    model.eval()

    ## upsampling layer
    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=test_image_size, mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=test_image_size, mode='bilinear')

    ## output of deeplab is logits, not probability
    softmax2d = nn.Softmax2d()

    ## evaluation process
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

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100

    writer.add_scalar('traversability_mask/test/mean_IoU', miou, writer_idx)
    writer.add_scalar('traversability_mask/test/loss', losses.avg, writer_idx)
    writer.add_scalar('traversability_mask/test/traversable_plant_IoU', iou[0], writer_idx)
    writer.add_scalar('traversability_mask/test/other_plant_mean_IoU', iou[1], writer_idx)
    writer.add_scalar('traversability_mask/test/artificial_object_mean_IoU', iou[2], writer_idx)
    writer.add_scalar('traversability_mask/test/ground_mean_IoU', iou[3], writer_idx)

#    if args.dataset == 'greenhouse':
#        # TODO: Check
    if args.use_depth:
        # model, images, depths=None, labels=None, predictions=None, class_encoding=None, writer=None, epoch=None, data=None, device=None
        in_training_visualization_img(model, images=images, depths=depths, labels=labels, 
            class_encoding=class_encoding, writer=writer, epoch=writer_idx, data='traversability_mask/test', device=device)
    else:
        in_training_visualization_img(model, images=images, labels=labels, class_encoding=class_encoding,
            writer=writer, epoch=writer_idx, data='traversability_mask/test', device=device)

    return miou

def generate_label(model, dataloader, save_pred_path, tgt_train_lst):
    image_path_list = []
    label_path_list = []
    depth_path_list = []

    model.eval()

    with torch.no_grad():
        ious = 0
        with tqdm(total=len(dataloader)) as pbar:
            for index, batch in enumerate(tqdm(dataloader)):
                if args.use_depth:
                    image, label, depth, name = batch
                else:
                    image, label, name = batch

                # Output: Numpy, KLD: Numpy
                output, _ = get_output(model, image) 
#                output2, _ = get_output(model2, image)

                output = output.transpose(1,2,0)
                amax_output = np.array(np.argmax(output, axis=2), dtype=np.uint8)

                path_name = name[0]
                name = name[0].split('/')[-1]
                image_name = name.rsplit('.', 1)[0]

                # Fuse the output with the corresponding traversablity mask
                label_np = np.array(label, np.uint8)
#                print((label_np[0]==TRAVERSED))
#                print((amax_output==PLANT))

                amax_output[(amax_output==PLANT) & (label_np[0]==TRAVERSED)] = TRAVERSABLE_PLANT
                #
                # prob
                # trainIDs/vis seg maps
                amax_output = Image.fromarray(amax_output.astype(np.uint8))
                # Save the predicted images (+ colorred images for visualization)
                amax_output.save('%s/%s.png' % (save_pred_path, image_name))

                image_path_list.append(path_name)
                label_path_list.append('%s/%s.png' % (save_pred_path, image_name)) 
                if args.use_depth:
                    depth_path_list.append(path_name.replace('color', 'depth'))
        
        pbar.close()

    update_image_list(tgt_train_lst, image_path_list, label_path_list, depth_path_list)

def get_output(model, image, model_name='espdnetue', device='cuda'):
    kld_layer = PixelwiseKLD()
    softmax2d = nn.Softmax2d()
    '''
    Get outputs from the input images
    '''
    # Forward the data
    output2 = model(image.to(device))

    # Calculate the output from the two classification layers
    if isinstance(output2, OrderedDict):
        pred = output2['out']
        pred_aux = output2['aux']
    elif model_name == 'espdnetue':
        pred = output2[0]
        pred_aux = output2[1]

    output2 = pred + 0.5 * pred_aux

    output = softmax2d(output2).cpu().data[0].numpy()

    kld = kld_layer(pred, pred_aux).cpu().data[0].numpy()

    return output, kld

def update_image_list(tgt_train_lst, image_path_list, label_path_list, depth_path_list=None):
    with open(tgt_train_lst, 'w') as f:
        for idx in range(len(image_path_list)):
            if depth_path_list:
                f.write("%s,%s,%s\n" % (image_path_list[idx], label_path_list[idx], depth_path_list[idx]))
            else:
                f.write("%s,%s\n" % (image_path_list[idx], label_path_list[idx]))

    return

if __name__=='__main__':
    main()