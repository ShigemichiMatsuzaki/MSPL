# ============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
# ============================================

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from utilities.utils import save_checkpoint, model_parameters, compute_flops
from utilities.print_utils import *
from utilities.utils import AverageMeter
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import random
import math
import time
import numpy as np

def main(args):
    crop_size = args.crop_size
    assert isinstance(crop_size, tuple)
    print_info_message('Running Model at image resolution {}x{} with batch size {}'.format(crop_size[0], crop_size[1],
                                                                                           args.batch_size))
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'

    if args.dataset == 'pascal':
        from data_loader.segmentation.voc import VOCSegmentation, VOC_CLASS_LIST
        train_dataset = VOCSegmentation(root=args.data_path, train=True, crop_size=crop_size, scale=args.scale,
                                        coco_root_dir=args.coco_path)
        val_dataset = VOCSegmentation(root=args.data_path, train=False, crop_size=crop_size, scale=args.scale)
        seg_classes = len(VOC_CLASS_LIST)
        class_wts = torch.ones(seg_classes)
    elif args.dataset == 'city':
        from data_loader.segmentation.cityscapes import CityscapesSegmentation, CITYSCAPE_CLASS_LIST
        train_dataset = CityscapesSegmentation(root=args.data_path, train=True, size=crop_size, scale=args.scale,
                                               coarse=args.coarse)
        val_dataset = CityscapesSegmentation(root=args.data_path, train=False, size=crop_size, scale=args.scale,
                                             coarse=False)
        seg_classes = len(CITYSCAPE_CLASS_LIST)
        class_wts = torch.ones(seg_classes)
        class_wts[0] = 2.8149201869965
        class_wts[1] = 6.9850029945374
        class_wts[2] = 3.7890393733978
        class_wts[3] = 9.9428062438965
        class_wts[4] = 9.7702074050903
        class_wts[5] = 9.5110931396484
        class_wts[6] = 10.311357498169
        class_wts[7] = 10.026463508606
        class_wts[8] = 4.6323022842407
        class_wts[9] = 9.5608062744141
        class_wts[10] = 7.8698215484619
        class_wts[11] = 9.5168733596802
        class_wts[12] = 10.373730659485
        class_wts[13] = 6.6616044044495
        class_wts[14] = 10.260489463806
        class_wts[15] = 10.287888526917
        class_wts[16] = 10.289801597595
        class_wts[17] = 10.405355453491
        class_wts[18] = 10.138095855713
        class_wts[19] = 0.0

    elif args.dataset == 'greenhouse':
        print(args.use_depth)
        from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation, GreenhouseDepth, GREENHOUSE_CLASS_LIST
        train_dataset = GreenhouseDepth(root=args.data_path, list_name='train_depth_ae.txt', train=True, size=crop_size, scale=args.scale, use_filter=True)
        val_dataset = GreenhouseRGBDSegmentation(root=args.data_path, list_name='val_depth_ae.txt', train=False, size=crop_size, scale=args.scale, use_depth=True)
        class_weights = np.load('class_weights.npy')[:4]
        print(class_weights)
        class_wts = torch.from_numpy(class_weights).float().to(device)

        seg_classes = len(GREENHOUSE_CLASS_LIST)
    else:
        print_error_message('Dataset: {} not yet supported'.format(args.dataset))
        exit(-1)

    print_info_message('Training samples: {}'.format(len(train_dataset)))
    print_info_message('Validation samples: {}'.format(len(val_dataset)))

    from model.autoencoder.depth_autoencoder import espnetv2_autoenc
    args.classes = 3
    model = espnetv2_autoenc(args)

    train_params = [{'params': model.get_basenet_params(), 'lr': args.lr * args.lr_mult}]

    optimizer = optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay)

    num_params = model_parameters(model)
    flops = compute_flops(model, input=torch.Tensor(1, 1, crop_size[0], crop_size[1]))
    print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(crop_size[0], crop_size[1], flops))
    print_info_message('Network Parameters: {:.2f} million'.format(num_params))

    writer = SummaryWriter(log_dir=args.savedir, comment='Training and Validation logs')
    try:
        writer.add_graph(model, input_to_model=torch.Tensor(1, 3, crop_size[0], crop_size[1]))
    except:
        print_log_message("Not able to generate the graph. Likely because your model is not supported by ONNX")

    start_epoch = 0

    print('device : ' + device)

    #criterion = nn.CrossEntropyLoss(weight=class_wts, reduction='none', ignore_index=args.ignore_idx)
    #criterion = SegmentationLoss(n_classes=seg_classes, loss_type=args.loss_type,
    #                             device=device, ignore_idx=args.ignore_idx,
    #                             class_wts=class_wts.to(device))
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    if num_gpus >= 1:
        if num_gpus == 1:
            # for a single GPU, we do not need DataParallel wrapper for Criteria.
            # So, falling back to its internal wrapper
            from torch.nn.parallel import DataParallel
            model = DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
        else:
            from utilities.parallel_wrapper import DataParallelModel, DataParallelCriteria
            model = DataParallelModel(model)
            model = model.cuda()
            criterion = DataParallelCriteria(criterion)
            criterion = criterion.cuda()

        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)

    if args.scheduler == 'fixed':
        step_size = args.step_size
        step_sizes = [step_size * i for i in range(1, int(math.ceil(args.epochs / step_size)))]
        from utilities.lr_scheduler import FixedMultiStepLR
        lr_scheduler = FixedMultiStepLR(base_lr=args.lr, steps=step_sizes, gamma=args.lr_decay)
    elif args.scheduler == 'clr':
        step_size = args.step_size
        step_sizes = [step_size * i for i in range(1, int(math.ceil(args.epochs / step_size)))]
        from utilities.lr_scheduler import CyclicLR
        lr_scheduler = CyclicLR(min_lr=args.lr, cycle_len=5, steps=step_sizes, gamma=args.lr_decay)
    elif args.scheduler == 'poly':
        from utilities.lr_scheduler import PolyLR
        lr_scheduler = PolyLR(base_lr=args.lr, max_epochs=args.epochs, power=args.power)
    elif args.scheduler == 'hybrid':
        from utilities.lr_scheduler import HybirdLR
        lr_scheduler = HybirdLR(base_lr=args.lr, max_epochs=args.epochs, clr_max=args.clr_max,
                                cycle_len=args.cycle_len)
    elif args.scheduler == 'linear':
        from utilities.lr_scheduler import LinearLR
        lr_scheduler = LinearLR(base_lr=args.lr, max_epochs=args.epochs)
    else:
        print_error_message('{} scheduler Not supported'.format(args.scheduler))
        exit()

    print_info_message(lr_scheduler)

    with open(args.savedir + os.sep + 'arguments.json', 'w') as outfile:
        import json
        arg_dict = vars(args)
        arg_dict['model_params'] = '{} '.format(num_params)
        arg_dict['flops'] = '{} '.format(flops)
        json.dump(arg_dict, outfile)

    extra_info_ckpt = '{}_{}_{}'.format(args.model, args.s, crop_size[0])
    best_loss = 0.0
    for epoch in range(start_epoch, args.epochs):
        lr_base = lr_scheduler.step(epoch)
        # set the optimizer with the learning rate
        # This can be done inside the MyLRScheduler
        lr_seg = lr_base * args.lr_mult
        optimizer.param_groups[0]['lr'] = lr_seg
        # optimizer.param_groups[1]['lr'] = lr_seg

        # Train
        model.train()
        losses = AverageMeter()
        for i, batch in enumerate(train_loader):
            inputs = batch[1].to(device=device) # Depth
            target = batch[0].to(device=device) # RGB
    
            outputs = model(inputs)
    
            if device == 'cuda':
                loss = criterion(outputs, target).mean()
                if isinstance(outputs, (list, tuple)):
                    target_dev = outputs[0].device
                    outputs = gather(outputs, target_device=target_dev)
            else:
                loss = criterion(outputs, target)
    
            losses.update(loss.item(), inputs.size(0))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#             if not (i % 10):
#                 print("Step {}, write images".format(i))
#                 image_grid = torchvision.utils.make_grid(outputs.data.cpu()).numpy()
#                 writer.add_image('Autoencoder/results/train', image_grid, len(train_loader) * epoch + i)

            writer.add_scalar('Autoencoder/Loss/train', loss.item(), len(train_loader) * epoch + i)

            print_info_message(
                'Running batch {}/{} of epoch {}'.format(i+1, len(train_loader), epoch+1))


        train_loss = losses.avg
    
        writer.add_scalar('Autoencoder/LR/seg', round(lr_seg, 6), epoch)

        # Val
        if epoch % 5 == 0:
            losses = AverageMeter()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    inputs = batch[2].to(device=device) # Depth
                    target = batch[0].to(device=device) # RGB
            
                    outputs = model(inputs)
            
                    if device == 'cuda':
                        loss = criterion(outputs, target)# .mean()
                        if isinstance(outputs, (list, tuple)):
                            target_dev = outputs[0].device
                            outputs = gather(outputs, target_device=target_dev)
                    else:
                        loss = criterion(outputs, target)
            
                    losses.update(loss.item(), inputs.size(0))

                    image_grid = torchvision.utils.make_grid(outputs.data.cpu()).numpy()
                    writer.add_image('Autoencoder/results/val', image_grid, epoch)
                    image_grid = torchvision.utils.make_grid(inputs.data.cpu()).numpy()
                    writer.add_image('Autoencoder/inputs/val', image_grid, epoch)
                    image_grid = torchvision.utils.make_grid(target.data.cpu()).numpy()
                    writer.add_image('Autoencoder/target/val', image_grid, epoch)

            val_loss = losses.avg
    
            print_info_message(
                'Running epoch {} with learning rates: base_net {:.6f}, segment_net {:.6f}'.format(epoch, lr_base, lr_seg))
    
            # remember best miou and save checkpoint
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
    
            weights_dict = model.module.state_dict() if device == 'cuda' else model.state_dict()
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': weights_dict,
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.savedir, extra_info_ckpt)
    
            writer.add_scalar('Autoencoder/Loss/val', val_loss, epoch)

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
    parser.add_argument('--finetune', default='', type=str, help='Finetune the segmentation model')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--use-depth', default=False, type=bool, help='Use depth')

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
    else:
        print_error_message('{} dataset not yet supported'.format(args.dataset))

    if not args.finetune:
        from model.weight_locations.classification import model_weight_map

        if args.model == 'espdnet':
            weight_file_key = '{}_{}'.format('espnetv2', args.s)
            assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
            args.weights = model_weight_map[weight_file_key]

            args.depth_weights = './results_segmentation/model_espnetv2_greenhouse/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_autoenc/20200323-073331/espnetv2_2.0_480_checkpoint.pth.tar'
        else:
            weight_file_key = '{}_{}'.format(args.model, args.s)
            assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
            args.weights = model_weight_map[weight_file_key]
    else:
        args.weights = ''
        assert os.path.isfile(args.finetune), '{} weight file does not exist'.format(args.finetune)

    assert len(args.crop_size) == 2, 'crop-size argument must contain 2 values'
    assert args.data_path != '', 'Dataset path is an empty string. Please check.'

    args.crop_size = tuple(args.crop_size)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    args.savedir = '{}/model_{}_{}/s_{}_sch_{}_loss_{}_res_{}_sc_{}_{}_autoenc/{}'.format(args.savedir, args.model, args.dataset, args.s,
                                                                         args.scheduler,
                                                                         args.loss_type, args.crop_size[0], args.scale[0], args.scale[1], timestr)
    main(args)