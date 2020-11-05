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
from utilities.utils import save_checkpoint, model_parameters, compute_flops, in_training_visualization_img, calc_cls_class_weight, import_os_model
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
from PIL import Image

from loss_fns.segmentation_loss import SegmentationLoss, NIDLoss, UncertaintyWeightedSegmentationLoss, PixelwiseKLD
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
from data_loader.segmentation.greenhouse import id_forest_to_greenhouse

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

def get_output(model, image, model_name='espdnetue', device='cuda'):
    model.to(device)
    kld_layer = PixelwiseKLD()
    softmax2d = nn.Softmax2d()
    '''
    Get outputs from the input images
    '''
    # Forward the data
    if not args.use_depth: #or outsource == 'camvid':
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

def merge_outputs(amax_outputs, seg_classes, thresh=None):
    # If not specified, the label with votes more than half of the number of the outputs is selected
    # We've found that 'all' policy provides the best accuracy
    num_data = amax_outputs.shape[0]
    if thresh is None or thresh == 'half':
        thresh = num_data // 2 + 1
    elif thresh == 'all':
        thresh = num_data
    elif isinstance(thresh, int) and thresh <= num_data:
        pass 
    else:
        thresh = num_data // 2 + 1
    
    counts_lst = []
    for class_id in range(seg_classes):
        # Count the number of data with class 'class_id' on each pixel
        count = (amax_outputs == class_id).sum(axis=0)
        counts_lst.append(count)

    counts_np = np.array(counts_lst)
    count_amax = counts_np.argmax(axis=0)
    count_max  = counts_np.max(axis=0)
    count_amax[count_max < thresh] = 4

    return count_amax

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
    from data_loader.segmentation.greenhouse import color_palette
    from data_loader.segmentation.camvid import color_encoding as color_encoding_camvid

    # Outsource
    os_model_name_list = [args.os_model1, args.os_model2, args.os_model3]
    os_weights_name_list = [args.os_weights1, args.os_weights2, args.os_weights3]
    os_data_name_list = [args.outsource1, args.outsource2, args.outsource3]
    os_model_name_list = [x for x in os_model_name_list if x is not None]
    os_weights_name_list = [x for x in os_weights_name_list if x is not None] 
    os_data_name_list = [x for x in os_data_name_list if x is not None]
    os_model_list = []
    print(os_model_name_list)
    print(os_weights_name_list)
    print(os_data_name_list)
    for os_m, os_w, os_d in zip(os_model_name_list, os_weights_name_list, os_data_name_list):
        if os_d == 'camvid':
            os_seg_classes = 13
        elif os_d == 'cityscapes':
            os_seg_classes = 20
        elif os_d == 'forest' or os_d == 'greenhouse':
            os_seg_classes = 5

        os_model = import_os_model(args, os_model=os_m, os_weights=os_w, os_seg_classes=os_seg_classes)
        os_model_list.append(os_model)

    from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation, GREENHOUSE_CLASS_LIST
    seg_classes = len(GREENHOUSE_CLASS_LIST)
    val_dataset = GreenhouseRGBDSegmentation(root='./vision_datasets/greenhouse/', list_name=args.val_list, use_traversable=False, 
                                             train=False, size=crop_size, use_depth=args.use_depth,
                                             normalize=args.normalize)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
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
            name   = batch[2]
            
            output_list = []
            for m, os_data in zip(os_model_list, os_data_name_list):
                # Output: Numpy, KLD: Numpy
                output, _ = get_output(m, inputs) 

                output = output.transpose(1,2,0)
                amax_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

                # save visualized seg maps & predication prob map
                if os_data == 'camvid':
                    amax_output = id_camvid_to_greenhouse[amax_output]
                elif os_data == 'cityscapes':
                    amax_output = id_cityscapes_to_greenhouse[amax_output]
                elif os_data == 'forest':
                    amax_output = id_forest_to_greenhouse[amax_output]

                output_list.append(amax_output)

            amax_output = merge_outputs(np.array(output_list), 
                seg_classes=5, thresh='all')
            
            # Output the generated label images
            if args.output_image:
                for path_name in name:
#                    path_name = name[0]
                    image_name = path_name.split('/')[-1]
                    image_name = image_name.rsplit('.', 1)[0]
                    amax_output_img_color = colorize_mask(amax_output, color_palette)
                    amax_output_img_color.save('%s/%s_color.png' % (args.savedir, image_name))

                    for output_i, name_i in zip(output_list, os_data_name_list):
                        amax_output_img_color = colorize_mask(output_i, color_palette)
                        amax_output_img_color.save('%s/%s_color_%s.png' % (args.savedir, image_name, name_i))

            outputs_argmax = torch.from_numpy(amax_output)
            
            inter, union = miou_class.get_iou(outputs_argmax, target)
            inter_meter.update(inter)
            union_meter.update(union)

            # measure elapsed time
            print("Batch {}/{} finished".format(i+1, len(val_loader)))
    
    iou = inter_meter.sum / (union_meter.sum + 1e-10) * 100
    miou = iou[[1, 2, 3]].mean()
    writer.add_scalar('label_eval/IoU', miou, 0)
    writer.add_scalar('label_eval/plant', iou[1], 0)
    writer.add_scalar('label_eval/artificial_object', iou[2], 0)
    writer.add_scalar('label_eval/ground', iou[3], 0)

    writer.close()

def colorize_mask(mask, palette):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

if __name__ == "__main__":
    from commons.general_details import segmentation_models, segmentation_schedulers, segmentation_loss_fns, \
        segmentation_datasets

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    # dataset and result directories
    parser.add_argument('--dataset', type=str, default='pascal', choices=segmentation_datasets, help='Datasets')
    parser.add_argument('--data-path', type=str, default='', help='dataset path')
    parser.add_argument('--savedir', type=str, default='./results_segmentation', help='Location to save the results')

    # input details
    parser.add_argument('--batch-size', type=int, default=40, help='list of batch sizes')
    parser.add_argument('--crop-size', type=int, nargs='+', default=[480, 256],
                        help='list of image crop sizes, with each item storing the crop size (should be a tuple).')

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
    parser.add_argument('--output-image', default=False, type=bool, help='Save images instead of writing in tensorboard log')
    parser.add_argument("--outsource1", type=str, default=None, dest='outsource1',
                        help="A dataset name that is used as a external dataset to provide initial pseudo-labels")
    parser.add_argument("--os-model1", type=str, default="espdnet", dest='os_model1',
                        help="Model for generating pseudo-labels")
    parser.add_argument("--os-weights1", type=str, default='./results_segmentation/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb_/20200420-095339/espdnet_2.0_480_best.pth', dest='os_weights1',
                        help="A dataset name that is used as a external dataset to provide initial pseudo-labels")
    parser.add_argument("--outsource2", type=str, default=None, dest='outsource2',
                        help="A dataset name that is used as a external dataset to provide initial pseudo-labels")
    parser.add_argument("--os-model2", type=str, default=None, dest='os_model2',
                        help="Model for generating pseudo-labels")
    parser.add_argument("--os-weights2", type=str, default=None, dest='os_weights2',
                        help="A dataset name that is used as a external dataset to provide initial pseudo-labels")
    parser.add_argument("--outsource3", type=str, default=None, dest='outsource3',
                        help="A dataset name that is used as a external dataset to provide initial pseudo-labels")
    parser.add_argument("--os-model3", type=str, default=None, dest='os_model3',
                        help="Model for generating pseudo-labels")
    parser.add_argument("--os-weights3", type=str, default=None, dest='os_weights3',
                        help="A dataset name that is used as a external dataset to provide initial pseudo-labels")
    parser.add_argument("--val-list", type=str, default='val_greenhouse_more.lst', dest='val_list',
                        help="Dataset to test the model")

    args = parser.parse_args()

    random.seed(1882)
    torch.manual_seed(1882)

    if not args.finetune:
        from model.weight_locations.classification import model_weight_map

        if args.model == 'espdnet' or args.model == 'espdnetue':
            weight_file_key = '{}_{}'.format('espnetv2', args.s)
            assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
            args.weights = model_weight_map[weight_file_key]
        elif args.model == 'deeplabv3':
            args.weights  = ''

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
    now = datetime.datetime.now()
    now += datetime.timedelta(hours=9) # JST = UTC + 9
    timestr = now.strftime("%Y%m%d-%H%M%S")

    args.savedir = '{}/{}_{}_{}'.format(
        args.savedir, args.model, args.dataset, timestr)

    main(args)
