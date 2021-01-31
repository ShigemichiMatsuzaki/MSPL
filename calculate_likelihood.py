import torch
from torchvision import transforms
import argparse
import datetime

from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentationTrav, GREENHOUSE_CLASS_LIST
from commons.general_details import segmentation_models, segmentation_datasets

# UI
from tkinter import *
from PIL import Image, ImageTk

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--ignore-idx', type=int, default=255, help='Index or label to be ignored during training')

parser.add_argument('--s', type=float, default=2.0, help='Factor by which channels will be scaled')
# model details
parser.add_argument('--freeze-bn', action='store_true', default=False, help='Freeze BN params or not')

# dataset and result directories
parser.add_argument('--dataset', type=str, default='greenhouse', choices=segmentation_datasets, help='Datasets')
parser.add_argument('--weights', type=str, default='./espdnetue_ssm.pth', help='Name of weight file to load')
parser.add_argument('--data-test-list', type=str, default='./vision_datasets/traversability_mask/greenhouse_a_test_new.lst',
                    help='Location to save the results')
parser.add_argument('--data-train-list', type=str, default='./vision_datasets/traversability_mask/greenhouse_b_train.lst',
                    help='Location to save the results')
parser.add_argument('--trav-module-weights', default='./espdnetue_tem.pth', 
                    type=str, help='Weight file of traversability module')
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
    from model.segmentation.espdnet_ue_traversability import espdnetue_seg
    model = espdnetue_seg(args, load_entire_weights=True, fix_pyr_plane_proj=True, spatial=False)
    model.to('cuda')
    model.eval()

    # Import a dataset
    trav_test_set = GreenhouseRGBDSegmentationTrav(list_name=args.data_test_list, use_depth=args.use_depth)
    trav_train_set = GreenhouseRGBDSegmentationTrav(list_name=args.data_train_list, use_depth=args.use_depth)
    trav_test_loader = torch.utils.data.DataLoader(
        trav_test_set, batch_size=8, shuffle=False, num_workers=0)

    low_pos_sum = 0
    low_neg_sum = 0
    mid_pos_sum = 0
    mid_neg_sum = 0
    high_pos_sum = 0
    high_neg_sum = 0
    for batch in trav_test_loader:
        input_image_tensor = batch['rgb'].to('cuda')
        masks = batch['mask'].to('cuda')

        output = model(input_image_tensor)
        prob = torch.sigmoid(output[2]) / 0.3

#        output_tensor[output_tensor > 1.0] = 1.0
        if prob.max() > 1:
            prob /= prob.max()

        prob = prob.squeeze(dim=1)
        # Positive 
        prob_pos = prob[masks == 1]
        low_pos = (prob_pos >= 0.0) & (prob_pos < 0.15)
        mid_pos = (prob_pos >= 0.15) & (prob_pos < 0.30)
        high_pos = (prob_pos >= 0.3)

        low_pos_sum += low_pos.sum().item()
        mid_pos_sum += mid_pos.sum().item()
        high_pos_sum += high_pos.sum().item()

        # Negative 
        prob_neg = prob[masks == 0]
        low_neg = (prob_neg >= 0.0) & (prob_neg < 0.15)
        mid_neg = (prob_neg >= 0.15) & (prob_neg < 0.30)
        high_neg = (prob_neg >= 0.3)

        low_neg_sum += low_neg.sum().item()
        mid_neg_sum += mid_neg.sum().item()
        high_neg_sum += high_neg.sum().item()

    c = low_pos_sum + low_neg_sum + mid_pos_sum + mid_neg_sum + high_pos_sum + high_neg_sum

    print('joint')
    print(low_neg_sum/c, low_pos_sum/c)
    print(mid_neg_sum/c, mid_pos_sum/c)
    print(high_neg_sum/c, high_pos_sum/c)
    print('conditional')
    marginal_pos = low_pos_sum + mid_pos_sum + high_pos_sum
    marginal_neg = low_neg_sum + mid_neg_sum + high_neg_sum
    print(low_neg_sum/marginal_neg, low_pos_sum/marginal_pos)
    print(mid_neg_sum/marginal_neg, mid_pos_sum/marginal_pos)
    print(high_neg_sum/marginal_neg, high_pos_sum/marginal_pos)


if __name__=='__main__':
    main()
