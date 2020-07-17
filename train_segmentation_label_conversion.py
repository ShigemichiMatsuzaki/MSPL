# ============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
# ============================================

import argparse
import os
import datetime
import copy
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

def import_dataset(label_conversion=True, device='cuda'):
    from data_loader.segmentation.camvid import CamVidSegmentation, CAMVID_CLASS_LIST, color_encoding
    train_dataset = CamVidSegmentation(
        root=args.data_path, list_name='train_camvid.txt', 
        train=True, size=args.crop_size, scale=args.scale, label_conversion=label_conversion)
    val_dataset = CamVidSegmentation(
        root=args.data_path, list_name='val_camvid.txt',
        train=False, size=args.crop_size, scale=args.scale, label_conversion=label_conversion)

    if label_conversion:
        from data_loader.segmentation.greenhouse import GREENHOUSE_CLASS_LIST, color_encoding
        seg_classes = len(GREENHOUSE_CLASS_LIST)
        class_wts = np.ones(seg_classes)
    else:
        seg_classes = len(CAMVID_CLASS_LIST)
        tmp_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

        class_wts = calc_cls_class_weight(tmp_loader, seg_classes)

#    if args.ignore_idx == seg_classes-1:
#        class_wts = torch.from_numpy(class_wts).float().to(device)[:args.ignore_idx]
#    else:
    class_wts = torch.from_numpy(class_wts).float().to(device)

    print("class weights : {}".format(class_wts))

    return train_dataset, val_dataset, class_wts, seg_classes, color_encoding

def freeze_bn_layer(model):
    print_info_message('Freezing batch normalization layers')
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

def show_network_stats(model, crop_size):
    num_params = model_parameters(model)
    flops = compute_flops(model, input=torch.Tensor(1, 3, crop_size[0], crop_size[1]))
    print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(crop_size[0], crop_size[1], flops))
    print_info_message('Network Parameters: {:.2f} million'.format(num_params))

    return num_params, flops

def get_lr_scheduler(scheduler='hybrid'):
    if scheduler == 'fixed':
        step_size = args.step_size
        step_sizes = [step_size * i for i in range(1, int(math.ceil(args.epochs / step_size)))]
        from utilities.lr_scheduler import FixedMultiStepLR
        lr_scheduler = FixedMultiStepLR(base_lr=args.lr, steps=step_sizes, gamma=args.lr_decay)
    elif scheduler == 'clr':
        step_size = args.step_size
        step_sizes = [step_size * i for i in range(1, int(math.ceil(args.epochs / step_size)))]
        from utilities.lr_scheduler import CyclicLR
        lr_scheduler = CyclicLR(min_lr=args.lr, cycle_len=5, steps=step_sizes, gamma=args.lr_decay)
    elif scheduler == 'poly':
        from utilities.lr_scheduler import PolyLR
        lr_scheduler = PolyLR(base_lr=args.lr, max_epochs=args.epochs, power=args.power)
    elif scheduler == 'hybrid':
        from utilities.lr_scheduler import HybirdLR
        lr_scheduler = HybirdLR(base_lr=args.lr, max_epochs=args.epochs, clr_max=args.clr_max,
                                cycle_len=args.cycle_len)
    elif scheduler == 'linear':
        from utilities.lr_scheduler import LinearLR
        lr_scheduler = LinearLR(base_lr=args.lr, max_epochs=args.epochs)
    else:
        print_error_message('{} scheduler Not supported'.format(scheduler))
        exit()

    print_info_message(lr_scheduler)

    return lr_scheduler

def write_stats_to_json(num_params, flops):
    with open(args.savedir + os.sep + 'arguments.json', 'w') as outfile:
        import json
        arg_dict = vars(args)
        arg_dict['model_params'] = '{} '.format(num_params)
        arg_dict['flops'] = '{} '.format(flops)
        json.dump(arg_dict, outfile)

def set_parameters_for_finetuning(args):
    pass

def main(args):
    crop_size = args.crop_size
    assert isinstance(crop_size, tuple)
    print_info_message('Running Model at image resolution {}x{} with batch size {}'.format(crop_size[0], crop_size[1],
                                                                                           args.batch_size))
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    print('device : ' + device)

    # Get a summary writer for tensorboard
    writer = SummaryWriter(log_dir=args.savedir, comment='Training and Validation logs')
    
    #
    # Training the model with 13 classes of CamVid dataset
    # TODO: This process should be done only if specified
    #
    if not args.finetune:
        train_dataset, val_dataset, class_wts, seg_classes, color_encoding = import_dataset(label_conversion=False) # 13 classes
        args.use_depth = False # 'use_depth' is always false for camvid
    
        print_info_message('Training samples: {}'.format(len(train_dataset)))
        print_info_message('Validation samples: {}'.format(len(val_dataset)))
    
        # Import model
        if args.model == 'espnetv2':
            from model.segmentation.espnetv2 import espnetv2_seg
            args.classes = seg_classes
            model = espnetv2_seg(args)
        elif args.model == 'espdnet':
            from model.segmentation.espdnet import espdnet_seg
            args.classes = seg_classes
            print("Trainable fusion : {}".format(args.trainable_fusion))
            print("Segmentation classes : {}".format(seg_classes))
            model = espdnet_seg(args)
        elif args.model == 'espdnetue':
            from model.segmentation.espdnet_ue import espdnetue_seg2
            args.classes = seg_classes
            print("Trainable fusion : {}".format(args.trainable_fusion))
            ("Segmentation classes : {}".format(seg_classes))
            print(args.weights)
            model = espdnetue_seg2(args, False, fix_pyr_plane_proj=True)
        else:
            print_error_message('Arch: {} not yet supported'.format(args.model))
            exit(-1)
    
        # Freeze batch normalization layers?
        if args.freeze_bn:
            freeze_bn_layer(model)
    
        # Set learning rates
        train_params = [{'params': model.get_basenet_params(), 'lr': args.lr},
                        {'params': model.get_segment_params(), 'lr': args.lr * args.lr_mult}]
        
        # Define an optimizer
        optimizer = optim.SGD(train_params, lr=args.lr * args.lr_mult, momentum=args.momentum, weight_decay=args.weight_decay)
    
        # Compute the FLOPs and the number of parameters, and display it
        num_params, flops = show_network_stats(model, crop_size)
    
        try:
            writer.add_graph(model, input_to_model=torch.Tensor(1, 3, crop_size[0], crop_size[1]))
        except:
            print_log_message("Not able to generate the graph. Likely because your model is not supported by ONNX")
    
    
        #criterion = nn.CrossEntropyLoss(weight=class_wts, reduction='none', ignore_index=args.ignore_idx)
        criterion = SegmentationLoss(n_classes=seg_classes, loss_type=args.loss_type,
                                     device=device, ignore_idx=args.ignore_idx,
                                     class_wts=class_wts.to(device))
        nid_loss = NIDLoss(image_bin=32, label_bin=seg_classes) if args.use_nid else None
    
        if num_gpus >= 1:
            if num_gpus == 1:
                # for a single GPU, we do not need DataParallel wrapper for Criteria.
                # So, falling back to its internal wrapper
                from torch.nn.parallel import DataParallel
                model = DataParallel(model)
                model = model.cuda()
                criterion = criterion.cuda()
                if args.use_nid:
                    nid_loss.cuda()
            else:
                from utilities.parallel_wrapper import DataParallelModel, DataParallelCriteria
                model = DataParallelModel(model)
                model = model.cuda()
                criterion = DataParallelCriteria(criterion)
                criterion = criterion.cuda()
                if args.use_nid:
                    nid_loss = DataParallelCriteria(nid_loss)
                    nid_loss = nid_loss.cuda()
    
            if torch.backends.cudnn.is_available():
                import torch.backends.cudnn as cudnn
                cudnn.benchmark = True
                cudnn.deterministic = True
    
        # Get data loaders for training and validation data
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=False,
                                                 pin_memory=True, num_workers=args.workers)
    
        # Get a learning rate scheduler
        lr_scheduler = get_lr_scheduler(args.scheduler)
    
        write_stats_to_json(num_params, flops)
    
        extra_info_ckpt = '{}_{}_{}'.format(args.model, args.s, crop_size[0])
        #
        # Main training loop of 13 classes
        # 
        start_epoch = 0
        best_miou = 0.0
        for epoch in range(start_epoch, args.epochs):
            lr_base = lr_scheduler.step(epoch)
            # set the optimizer with the learning rate
            # This can be done inside the MyLRScheduler
            lr_seg = lr_base * args.lr_mult
            optimizer.param_groups[0]['lr'] = lr_base
            optimizer.param_groups[1]['lr'] = lr_seg
    
            print_info_message(
                'Running epoch {} with learning rates: base_net {:.6f}, segment_net {:.6f}'.format(epoch, lr_base, lr_seg))
    
            # Use different training functions for espdnetue
            if args.model == 'espdnetue':
                from utilities.train_eval_seg import train_seg_ue as train
                from utilities.train_eval_seg import val_seg_ue as val
            else:
                from utilities.train_eval_seg import train_seg as train
                from utilities.train_eval_seg import val_seg as val
    
            miou_train, train_loss = train(
                model, train_loader, optimizer, criterion, seg_classes, epoch, device=device, use_depth=args.use_depth, add_criterion=nid_loss)
            miou_val, val_loss = val(model, val_loader, criterion, seg_classes, device=device, use_depth=args.use_depth, add_criterion=nid_loss)
    
            batch_train = iter(train_loader).next()
            batch = iter(val_loader).next()
            in_training_visualization_img(
                model, images=batch_train[0].to(device=device), labels=batch_train[1].to(device=device), 
                class_encoding=color_encoding, writer=writer, epoch=epoch, data='Segmentation/train', device=device)
            in_training_visualization_img(
                model, images=batch[0].to(device=device), labels=batch[1].to(device=device),
                class_encoding=color_encoding, writer=writer, epoch=epoch, data='Segmentation/val', device=device)
    
            # remember best miou and save checkpoint
            is_best = miou_val > best_miou
            best_miou = max(miou_val, best_miou)
    
            weights_dict = model.module.state_dict() if device == 'cuda' else model.state_dict()
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': weights_dict,
                'best_miou': best_miou,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.savedir, extra_info_ckpt)
    
            writer.add_scalar('Segmentation/LR/base', round(lr_base, 6), epoch)
            writer.add_scalar('Segmentation/LR/seg', round(lr_seg, 6), epoch)
            writer.add_scalar('Segmentation/Loss/train', train_loss, epoch)
            writer.add_scalar('Segmentation/Loss/val', val_loss, epoch)
            writer.add_scalar('Segmentation/mIOU/train', miou_train, epoch)
            writer.add_scalar('Segmentation/mIOU/val', miou_val, epoch)
            writer.add_scalar('Segmentation/Complexity/Flops', best_miou, math.ceil(flops))
            writer.add_scalar('Segmentation/Complexity/Params', best_miou, math.ceil(num_params))

        # Save the pretrained weights
        model_dict = copy.deepcopy(model.state_dict())
        del model
        torch.cuda.empty_cache()
    
    #
    # Finetuning with 4 classes
    #
    args.ignore_idx = 4
    train_dataset, val_dataset, class_wts, seg_classes, color_encoding = import_dataset(label_conversion=True) # 5 classes

    print_info_message('Training samples: {}'.format(len(train_dataset)))
    print_info_message('Validation samples: {}'.format(len(val_dataset)))

    #set_parameters_for_finetuning()
    

    # Import model
    if args.model == 'espnetv2':
        from model.segmentation.espnetv2 import espnetv2_seg
        args.classes = seg_classes
        model = espnetv2_seg(args)
    elif args.model == 'espdnet':
        from model.segmentation.espdnet import espdnet_seg
        args.classes = seg_classes
        print("Trainable fusion : {}".format(args.trainable_fusion))
        print("Segmentation classes : {}".format(seg_classes))
        model = espdnet_seg(args)
    elif args.model == 'espdnetue':
        from model.segmentation.espdnet_ue import espdnetue_seg2
        args.classes = seg_classes
        print("Trainable fusion : {}".format(args.trainable_fusion))
        print("Segmentation classes : {}".format(seg_classes))
        print(args.weights)
        model = espdnetue_seg2(args, args.finetune, fix_pyr_plane_proj=True)
    else:
        print_error_message('Arch: {} not yet supported'.format(args.model))
        exit(-1)

    if not args.finetune:
        new_model_dict = model.state_dict() 
#        for k, v in model_dict.items():
#            if k.lstrip('module.') in new_model_dict:
#                print('In:{}'.format(k.lstrip('module.')))
#            else:
#                print('Not In:{}'.format(k.lstrip('module.')))
        overlap_dict = {k.replace('module.', ''): v for k, v in model_dict.items() 
                        if k.replace('module.', '') in new_model_dict and new_model_dict[k.replace('module.', '')].size() == v.size()}
        no_overlap_dict = {k.replace('module.', ''): v for k, v in new_model_dict.items() 
                           if k.replace('module.', '') not in new_model_dict or new_model_dict[k.replace('module.', '')].size() != v.size()}
        print(no_overlap_dict.keys())

        new_model_dict.update(overlap_dict)
        model.load_state_dict(new_model_dict)

    output = model(torch.ones(1, 3, 288, 480))
    print(output[0].size())

    print(seg_classes)
    print(class_wts.size())
    #print(model_dict.keys())
    #print(new_model_dict.keys())
    criterion = SegmentationLoss(n_classes=seg_classes, loss_type=args.loss_type,
                                 device=device, ignore_idx=args.ignore_idx,
                                 class_wts=class_wts.to(device))
    nid_loss = NIDLoss(image_bin=32, label_bin=seg_classes) if args.use_nid else None

    # Set learning rates
    args.lr /= 100
    train_params = [{'params': model.get_basenet_params(), 'lr': args.lr},
                    {'params': model.get_segment_params(), 'lr': args.lr * args.lr_mult}]
    # Define an optimizer
    optimizer = optim.SGD(train_params, lr=args.lr * args.lr_mult, momentum=args.momentum, weight_decay=args.weight_decay)

    if num_gpus >= 1:
        if num_gpus == 1:
            # for a single GPU, we do not need DataParallel wrapper for Criteria.
            # So, falling back to its internal wrapper
            from torch.nn.parallel import DataParallel
            model = DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
            if args.use_nid:
                nid_loss.cuda()
        else:
            from utilities.parallel_wrapper import DataParallelModel, DataParallelCriteria
            model = DataParallelModel(model)
            model = model.cuda()
            criterion = DataParallelCriteria(criterion)
            criterion = criterion.cuda()
            if args.use_nid:
                nid_loss = DataParallelCriteria(nid_loss)
                nid_loss = nid_loss.cuda()

        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    # Get data loaders for training and validation data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)

    # Get a learning rate scheduler
    args.epochs = 50
    lr_scheduler = get_lr_scheduler(args.scheduler)

    # Compute the FLOPs and the number of parameters, and display it
    num_params, flops = show_network_stats(model, crop_size)
    write_stats_to_json(num_params, flops)

    extra_info_ckpt = '{}_{}_{}_{}'.format(args.model, seg_classes, args.s, crop_size[0])
    #
    # Main training loop of 13 classes
    # 
    start_epoch = 0
    best_miou = 0.0
    for epoch in range(start_epoch, args.epochs):
        lr_base = lr_scheduler.step(epoch)
        # set the optimizer with the learning rate
        # This can be done inside the MyLRScheduler
        lr_seg = lr_base * args.lr_mult
        optimizer.param_groups[0]['lr'] = lr_base
        optimizer.param_groups[1]['lr'] = lr_seg

        print_info_message(
            'Running epoch {} with learning rates: base_net {:.6f}, segment_net {:.6f}'.format(epoch, lr_base, lr_seg))

        # Use different training functions for espdnetue
        if args.model == 'espdnetue':
            from utilities.train_eval_seg import train_seg_ue as train
            from utilities.train_eval_seg import val_seg_ue as val
        else:
            from utilities.train_eval_seg import train_seg as train
            from utilities.train_eval_seg import val_seg as val

        miou_train, train_loss = train(
            model, train_loader, optimizer, criterion, seg_classes, epoch, device=device, use_depth=args.use_depth, add_criterion=nid_loss)
        miou_val, val_loss = val(model, val_loader, criterion, seg_classes, device=device, use_depth=args.use_depth, add_criterion=nid_loss)

        batch_train = iter(train_loader).next()
        batch = iter(val_loader).next()
        in_training_visualization_img(
            model, images=batch_train[0].to(device=device), labels=batch_train[1].to(device=device), 
            class_encoding=color_encoding, writer=writer, epoch=epoch, data='SegmentationConv/train', device=device)
        in_training_visualization_img(
            model, images=batch[0].to(device=device), labels=batch[1].to(device=device),
            class_encoding=color_encoding, writer=writer, epoch=epoch, data='SegmentationConv/val', device=device)

        # remember best miou and save checkpoint
        is_best = miou_val > best_miou
        best_miou = max(miou_val, best_miou)

        weights_dict = model.module.state_dict() if device == 'cuda' else model.state_dict()
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model,
            'state_dict': weights_dict,
            'best_miou': best_miou,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.savedir, extra_info_ckpt)

        writer.add_scalar('SegmentationConv/LR/base', round(lr_base, 6), epoch)
        writer.add_scalar('SegmentationConv/LR/seg', round(lr_seg, 6), epoch)
        writer.add_scalar('SegmentationConv/Loss/train', train_loss, epoch)
        writer.add_scalar('SegmentationConv/Loss/val', val_loss, epoch)
        writer.add_scalar('SegmentationConv/mIOU/train', miou_train, epoch)
        writer.add_scalar('SegmentationConv/mIOU/val', miou_val, epoch)
        writer.add_scalar('SegmentationConv/Complexity/Flops', best_miou, math.ceil(flops))
        writer.add_scalar('SegmentationConv/Complexity/Params', best_miou, math.ceil(num_params))

    writer.close()


if __name__ == "__main__":
    from commons.general_details import segmentation_models, segmentation_schedulers, segmentation_loss_fns, \
        segmentation_datasets

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--ignore-idx', type=int, default=255, help='Index or label to be ignored during training')
    parser.add_argument('--ignore-idx-conversion', type=int, default=4, help='Index or label to be ignored during training of 4 classes')

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
    parser.add_argument('--crop-size', type=int, nargs='+', default=[256, 256],
                        help='list of image crop sizes, with each item storing the crop size (should be a tuple).')
    parser.add_argument('--loss-type', default='ce', choices=segmentation_loss_fns, help='Loss function (ce or miou)')

    # model related params
    parser.add_argument('--s', type=float, default=2.0, help='Factor by which channels will be scaled')
    parser.add_argument('--model', default='espnet', choices=segmentation_models,
                        help='Which model? basic= basic CNN model, res=resnet style)')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')
    parser.add_argument('--weights', default='', type=str, help='Finetune the segmentation model')
    parser.add_argument('--finetune', default=False, type=bool, help='Whether to do only finetuning with 4 labels')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--use-depth', default=False, type=bool, help='Use depth')
    parser.add_argument('--trainable-fusion', default=False, type=bool, help='Use depth')
    parser.add_argument('--dense-fuse', default=False, type=bool, help='Use depth')
    parser.add_argument('--label-conversion', default=False, type=bool, help='Use label conversion in CamVid')
    parser.add_argument('--use-nid', default=False, type=bool, help='Use NID loss')

    args = parser.parse_args()

    random.seed(1882)
    torch.manual_seed(1882)

    if args.dataset == 'camvid':
        args.scale = (0.5, 2.0)

    if not args.weights:
        from model.weight_locations.classification import model_weight_map

        if args.model == 'espdnet' or args.model == 'espdnetue':
            weight_file_key = '{}_{}'.format('espnetv2', args.s)
            assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
            args.weights = model_weight_map[weight_file_key]

    #        if args.use_depth:
    #            args.weights
        else:
            weight_file_key = '{}_{}'.format(args.model, args.s)
            assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
            args.weights = model_weight_map[weight_file_key]

    assert len(args.crop_size) == 2, 'crop-size argument must contain 2 values'
    assert args.data_path != '', 'Dataset path is an empty string. Please check.'

    args.crop_size = tuple(args.crop_size)
    #timestr = time.strftime("%Y%m%d-%H%M%S")
    now = datetime.datetime.now()
    now += datetime.timedelta(hours=9)
    timestr = now.strftime("%Y%m%d-%H%M%S")
    use_depth_str = "_rgbd" if args.use_depth else "_rgb"
    if args.use_depth:
        trainable_fusion_str = "_gated" if args.trainable_fusion else "_naive"
    else:
        trainable_fusion_str = ""

    args.savedir = '{}/model_{}_{}/s_{}_sch_{}_loss_{}_res_{}_sc_{}_{}{}{}/{}'.format(
        args.savedir, args.model, args.dataset, args.s,
        args.scheduler, args.loss_type, args.crop_size[0], args.scale[0], args.scale[1], use_depth_str, trainable_fusion_str, timestr)

    main(args)
