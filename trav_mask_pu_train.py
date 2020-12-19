#============================================
__author__ = "Shigemichi Matsuzaki"
__maintainer__ = "Shigemichi Matsuzaki"
#============================================

import argparse
from packaging import version
import time
import datetime
import os
import os.path as osp
from collections import OrderedDict
from torchsummary import summary
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from operator import itemgetter

from PIL import Image
import numpy as np

from utilities.utils import save_checkpoint, in_training_visualization_img
from utilities.utils import AverageMeter
from utilities.metrics.segmentation_miou import MIOU
#from utilities.train_eval_seg import train_seg as train
#from utilities.train_eval_seg import val_seg as val
from loss_fns.segmentation_loss import SelectiveBCE

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
    parser.add_argument('--spatial', default=False, type=bool, help='Use 3x3 kernel or 1x1 kernel in probability estimation')
    parser.add_argument('--feature-construction', default='concat', type=str, help='Use 3x3 kernel or 1x1 kernel in probability estimation')

    return parser.parse_args()

args = get_arguments()



def main():
    device = 'cuda'

    now = datetime.datetime.now()
    now += datetime.timedelta(hours=9)
    timestr = now.strftime("%Y%m%d-%H%M%S")

    save_path = '{}/model_{}_{}/{}'.format(
        args.save, 
        args.model, args.dataset, 
        timestr)

    print(save_path)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_pred_path = osp.join(save_path, 'pred')
    if not os.path.isdir(save_pred_path):
        os.makedirs(save_pred_path)
    writer = SummaryWriter(save_path)

    # Dataset
    from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentationTrav, GREENHOUSE_CLASS_LIST
    args.classes = len(GREENHOUSE_CLASS_LIST)
    travset = GreenhouseRGBDSegmentationTrav(list_name=args.data_trav_list, use_depth=args.use_depth)

    # Dataloader for generating the pseudo-labels
    travloader = torch.utils.data.DataLoader(
        travset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=args.pin_memory)

    # Models
    from model.classification.label_prob_estimator import LabelProbEstimator
    in_channels = 32 if args.feature_construction == 'concat' else 16
    prob_model = LabelProbEstimator(in_channels=in_channels, spatial=args.spatial)
    prob_model.to(device)

    from model.segmentation.espdnet_ue import espdnetue_seg2
    args.weights = args.restore_from
    seg_model = espdnetue_seg2(args, load_entire_weights=True, fix_pyr_plane_proj=True)
    seg_model.to(device)

    criterion = SelectiveBCE()
#    # Datset for training
    from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation
#    trainset = GreenhouseRGBDSegmentation(list_name=tgt_train_lst, use_depth=args.use_depth, use_traversable=True)
    testset  = GreenhouseRGBDSegmentation(list_name=args.data_test_list, use_depth=args.use_depth, use_traversable=True)
#
#    trainloader = torch.utils.data.DataLoader(
#        trainset, batch_size=args.batch_size, shuffle=True,
#        num_workers=0, pin_memory=args.pin_memory)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=args.pin_memory)
#
#    # Loss
#    class_weights = torch.tensor([1.0, 0.2, 1.0, 1.0, 0.0]).to(device)
#
#    criterion = nn.BCEWithLogitsLoss().to(device)
#
#    # Optimizer
#    if args.use_depth:
#        train_params = [{'params': model.get_basenet_params(), 'lr': args.learning_rate * 0.1},
#                        {'params': model.get_segment_params(), 'lr': args.learning_rate},
#                        {'params': model.get_depth_encoder_params(), 'lr': args.learning_rate}]
#    else:
#        train_params = [{'params': model.get_basenet_params(), 'lr': args.learning_rate * 0.1},
#                        {'params': model.get_segment_params(), 'lr': args.learning_rate}]
#
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            prob_model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)            
    else:
        optimizer = optim.Adam(
            prob_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.learning_rate, max_lr=args.learning_rate*10,
                                     step_size_up=10, step_size_down=20, cycle_momentum=True if args.optimizer == 'SGD' else False)
##    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)
#
#    best_miou = 0.0
    for epoch in range(0, args.epoch):
        # Run a training epoch
        train(travloader, prob_model, seg_model, criterion, device, optimizer, epoch, writer)
        scheduler.step()

        test(testloader, prob_model, seg_model, device, epoch, writer)

def train(trainloader, prob_model, seg_model, criterion, device, optimizer, writer_idx, writer=None):
    """Create the model and start the training."""
    # For logging the training status
    losses = AverageMeter()

    prob_model.train()
    seg_model.eval()

    with tqdm(total=len(trainloader)) as pbar:
        for i_iter, batch in enumerate(tqdm(trainloader)):
#            feature = batch['feature'].to(device)
#            label = batch['label'].to(device)
            images = batch[0].to(device)
            labels = batch[1].to(device)

            optimizer.zero_grad()

            # Output probability
            # output = model(feature, label)
            output_dict = get_output(seg_model, images)
            seg_output = output_dict['output']
            feature = output_dict['feature']

            # Generate mask from the segmentation result
            seg_output_amax = torch.argmax(seg_output, dim=1, keepdim=True)

            # Forward the feature and calculate probability
            prob_output = prob_model(feature)

            # Loss calculation
#            loss = criterion(prob_output, labels, seg_output_amax==PLANT)
            loss = criterion(prob_output, labels)

            # Model regularizer
            losses.update(loss.item(), feature.size(0))
    
            # Optimise
            loss.backward()
            optimizer.step()

    # Write summary
    writer.add_scalar('traversability_mask/train/loss', losses.avg, writer_idx)
    writer.add_scalar('traversability_mask/train/learning_rate', optimizer.param_groups[0]['lr'], writer_idx)

def test(testloader, prob_model, seg_model, device, writer_idx, writer):
    """Create the model and start the evaluation process."""
    # For logging the training status
    from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation
    ds = GreenhouseRGBDSegmentation(
        list_name=args.data_test_list, train=False, use_traversable=True, use_depth=args.use_depth)

    testloader = data.DataLoader(ds, batch_size=16, shuffle=False, pin_memory=args.pin_memory)
    sigmoid = nn.Sigmoid()
    prob_sum_meter = AverageMeter()

    ## model for evaluation
    prob_model.eval()

    # TODO: Change this (implement the same function in 'utility/utils.py', or uncomment the code below with slight modification)
    with torch.no_grad():
        # Calculate a constant c

        for i_iter, batch in enumerate(tqdm(testloader)):
            images = batch[0].to(device)
            masks = batch[1].to(device)
            if args.use_depth:
                depths = batch[2].to(device)

            output_dict = get_output(seg_model, images)
            feature = output_dict['feature']
            masks = torch.reshape(masks, (masks.size(0), -1, masks.size(1), masks.size(2)))
            prob_sum_meter.update(sigmoid(prob_model(feature))[masks==1].mean().item(), images.size(0))
        
        # Visualize
        batch = iter(testloader).next()

        images = batch[0].to(device)
        if args.use_depth:
            depths = batch[2].to(device)

        output_dict = get_output(seg_model, images)

        feature = output_dict['feature']

        c = prob_sum_meter.avg
        prob = sigmoid(prob_model(feature)) / c
        prob /= prob.max()

        image_grid = torchvision.utils.make_grid(images.data.cpu()).numpy()
        prob_grid = torchvision.utils.make_grid(prob.data.cpu()).numpy()

        writer.add_image('traversability_mask/train/image', image_grid, writer_idx)
        writer.add_image('traversability_mask/train/prob', prob_grid, writer_idx)

def get_output(model, image, model_name='espdnetue', device='cuda'):
    softmax2d = nn.Softmax2d()
    '''
    Get outputs from the input images
    '''
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.bu_dec_l4.merge_layer[2].register_forward_hook(get_activation('output_main'))
    model.aux_decoder.merge_layer[2].register_forward_hook(get_activation('output_aux'))

    with torch.no_grad():
        output2 = model(image.to(device))
#    output = model(input.to(device))

    # Forward the data
    # Calculate the output from the two classification layers
    if isinstance(output2, OrderedDict):
        pred = output2['out']
        pred_aux = output2['aux']
    elif model_name == 'espdnetue':
        pred = output2[0]
        pred_aux = output2[1]

    output2 = pred + 0.5 * pred_aux

    output = softmax2d(output2)# .cpu().data[0].numpy()

    # Calculate feature from the intermediate layers
    main_feature = F.interpolate(activation['output_main'], size=(image.size(2), image.size(3)), mode='bilinear')
    aux_feature = F.interpolate(activation['output_aux'], size=(image.size(2), image.size(3)), mode='bilinear')
    if args.feature_construction == 'concat':
        feature = torch.cat((main_feature, aux_feature), dim=1)
    else:
        feature = main_feature + 0.5 * aux_feature

    return {'output': output, 'feature': feature}

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