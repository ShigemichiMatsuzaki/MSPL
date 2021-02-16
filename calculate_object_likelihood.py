import torch
from torchvision import transforms
import argparse
import datetime

from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation, GREENHOUSE_CLASS_LIST
from commons.general_details import segmentation_models, segmentation_datasets

import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--ignore-idx', type=int, default=255, help='Index or label to be ignored during training')

parser.add_argument('--s', type=float, default=2.0, help='Factor by which channels will be scaled')
# model details
parser.add_argument('--freeze-bn', action='store_true', default=False, help='Freeze BN params or not')

# dataset and result directories
parser.add_argument('--dataset', type=str, default='greenhouse', choices=segmentation_datasets, help='Datasets')
parser.add_argument('--weights', type=str, default='/tmp/runs/uest/for-paper/trav/model_espdnetue_greenhouse/s_2.0_res_480_uest_rgb_os_camvid_cityscapes_forest_ue/20201204-152737/espdnetue_ep_1.pth', help='Name of weight file to load')
parser.add_argument('--data-test-list', type=str, default='./vision_datasets/greenhouse/val_greenhouse_merged.lst',
                    help='Location to save the results')
#parser.add_argument('--trav-module-weights', default='./espdnetue_tem.pth', 
#                    type=str, help='Weight file of traversability module')
# model related params
parser.add_argument('--model', default='espdnetue', 
                    help='Which model? basic= basic CNN model, res=resnet style)')
parser.add_argument('--model-name', default='espdnetue_trav', 
                    help='Which model? basic= basic CNN model, res=resnet style)')
parser.add_argument('--channels', default=3, type=int, help='Input channels')
parser.add_argument('--num-classes', default=1000, type=int,
                    help='ImageNet classes. Required for loading the base network')
parser.add_argument('--model-width', default=224, type=int, help='Model width')
parser.add_argument('--model-height', default=224, type=int, help='Model height')
parser.add_argument('--use-depth', default=False, type=bool, help='Use depth')
parser.add_argument('--trainable-fusion', default=False, type=bool, help='Use depth')
parser.add_argument('--dense-fuse', default=False, type=bool, help='Use depth')
parser.add_argument('--label-conversion', default=False, type=bool, help='Use label conversion in CamVid')
parser.add_argument('--use-uncertainty', default=True, type=bool, help='Use auxiliary loss')
parser.add_argument('--normalize', default=False, type=bool, help='Use auxiliary loss')

args = parser.parse_args()

# Main
def main():
    # Load model weights
    args.classes = len(GREENHOUSE_CLASS_LIST)
    from model.segmentation.espdnet_ue import espdnetue_seg2
    model = espdnetue_seg2(args, load_entire_weights=True, fix_pyr_plane_proj=True)

    model.to('cuda')
    model.eval()

    # Import a dataset
    trav_test_set = GreenhouseRGBDSegmentation(
        list_name=args.data_test_list, train=False, use_traversable=False, use_depth=False)

    testloader = torch.utils.data.DataLoader(trav_test_set, batch_size=32, shuffle=False, pin_memory=False)

    # Likelihood table
    lklhd_o_l = np.zeros((len(GREENHOUSE_CLASS_LIST)-1, len(GREENHOUSE_CLASS_LIST)-1))
    
    # Count the observations on the test images
    for batch in testloader:
        image = batch[0].to('cuda')
        label = batch[1].to('cuda').squeeze()
        output = model(image)
        output = output[0] + 0.5 * output[1]
        amax = torch.argmax(output, dim=1, keepdim=False)
        for l in range(len(GREENHOUSE_CLASS_LIST)-1):
            # amax_l : Predction on the pixels of true label l
            print(label.size())
            amax_l = amax[label==l]
            for o in range(len(GREENHOUSE_CLASS_LIST)-1):
                # Count the number of pixels predicted as o
                lklhd_o_l[l][o] += (amax_l == o).sum()

    for l in range(len(GREENHOUSE_CLASS_LIST)-1):
        if not math.isclose(lklhd_o_l[l].sum(), 0):
            lklhd_o_l[l] /= lklhd_o_l[l].sum()

    print(lklhd_o_l)

if __name__=='__main__':
    main()
