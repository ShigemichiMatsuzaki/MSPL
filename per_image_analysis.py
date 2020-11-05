# ============================================
__author__ = "ShigemichiMatsuzaki"
__license__ = "MIT"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================
import os
import torch
import numpy as np
import argparse
from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation, GREENHOUSE_CLASS_LIST, color_encoding, color_palette
from data_loader.segmentation.greenhouse import id_camvid_to_greenhouse
from model.segmentation.espdnet_ue import espdnetue_seg2
from model.segmentation.espdnet import espdnet_seg_with_pre_rgbd
from utilities.utils import AverageMeter
from utilities.metrics.segmentation_miou import MIOU
from utilities.print_utils import *
import time
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='espdnet', 
                        help='Which model? basic= basic CNN model, res=resnet style)')
    parser.add_argument('--dataset', default='greenhouse', 
                        help='Which model? basic= basic CNN model, res=resnet style)')
    parser.add_argument('--data-path', default='./vision_datasets/greenhouse/', 
                        help='Directory of the data list')
#    parser.add_argument('--weights', type=str, default='/tmp/runs/results_segmentation/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200513-204023/espdnet_2.0_480_best.pth', 
    parser.add_argument('--weights', type=str, default='./results_segmentation/model_espdnet_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb_/20200420-095339/espdnet_2.0_480_best.pth', 
                        help='path to checkpoint to resume from')

    # dataset and result directories
    parser.add_argument('--savedir', type=str, default='./results_segmentation', help='Location to save the results')

    # input details
    parser.add_argument('--batch-size', type=int, default=40, help='list of batch sizes')
    parser.add_argument('--loss-type', default='ce',  help='Loss function (ce or miou)')

    # model related params
    parser.add_argument('--s', type=float, default=2.0, help='Factor by which channels will be scaled')
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
    parser.add_argument('--crop-size', type=int, nargs='+', default=[480, 288],
                        help='list of image crop sizes, with each item storing the crop size (should be a tuple).')

    return parser.parse_args()

def test(model, data_loader):
    pass

def val_seg_per_image(model, dataset_loader, criterion=None, num_classes=21, device='cuda', use_depth=False):
    model.eval()
    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    miou_class = MIOU(num_classes=num_classes-1)

    if criterion:
        losses = AverageMeter()

    accuracy_list = {}
    with torch.no_grad():
        for i, batch in enumerate(dataset_loader):
            inputs = batch[0].to(device=device)
            target = batch[1].to(device=device)
            
            if use_depth:
                depth = batch[2].to(device=device)
                outputs = model(inputs, depth)
            else:
                outputs = model(inputs)

            if criterion:
                if device == 'cuda':
                    loss = criterion(outputs, target).mean()
                    if isinstance(outputs, (list, tuple)):
                        target_dev = outputs[0].device
                        outputs = gather(outputs, target_device=target_dev)
                else:
                    loss = criterion(outputs, target)

                losses.update(loss.item(), inputs.size(0))

            inter, union = miou_class.get_iou(outputs, target)
            inter_meter.update(inter)
            union_meter.update(union)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            iou = inter_meter.sum / (union_meter.sum + 1e-10)
            miou = iou.mean() * 100
            loss_ = losses.avg if criterion is not None else 0
#            print_log_message("[%d/%d]\t\tBatch Time:%.4f\t\tLoss:%.4f\t\tmiou:%.4f" %
#                  (i, len(dataset_loader), batch_time.avg, loss_, miou))

            accuracy_list[i] = miou

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean() * 100

    print_info_message('Mean IoU: {0:.2f}'.format(miou))

    return accuracy_list

def tensor_to_image(tensor, palette):
    np_tensor = tensor.reshape(tensor.size(1), tensor.size(2)).cpu().numpy()
    np_tensor = id_camvid_to_greenhouse[np_tensor]
    image = Image.fromarray(np.uint8(np_tensor)).convert('P')
    image.putpalette(palette)

    return image
#    image.save()

def main():
    args = parse_args()
    args.crop_size = tuple(args.crop_size)

    # Import dataset
    val_dataset = GreenhouseRGBDSegmentation(root=args.data_path, list_name='val_cucumber.lst', train=False, size=args.crop_size, use_depth=args.use_depth)
    data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    seg_classes = 5
#    seg_classes = len(GREENHOUSE_CLASS_LIST)
    args.classes = 13

    # Import model
    #model = espdnetue_seg(args)
    #model = espdnetue_seg2(args, load_entire_weights=True, fix_pyr_plane_proj=True)
    model = espdnet_seg_with_pre_rgbd(args, load_entire_weights=True)

    model.to('cuda')

    # Test
    accuracy_list = val_seg_per_image(model, data_loader, criterion=None, num_classes=seg_classes, device='cuda', use_depth=args.use_depth)
    accuracy_list = sorted(accuracy_list.items(), key=lambda x:x[1])

    print(accuracy_list)

    for i in range(0, len(accuracy_list)):
        index = accuracy_list[i][0]
    
        #print(val_dataset[index])
        size = val_dataset[index][0].size()
        output = model(val_dataset[index][0].reshape(1, size[0], size[1], size[2]).to('cuda'))
        _, amax = torch.max(output, dim=1)
    
        image = tensor_to_image(amax, color_palette)

        image.save(os.path.join('bad_images', str(i) + '.png'))
        #print(val_dataset[index][2])

if __name__=='__main__':
    main()
