import torch
import torchvision
from torchvision import transforms
import argparse
import datetime
import numpy as np
import cv2

from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentationTrav, GREENHOUSE_CLASS_LIST
from commons.general_details import segmentation_models, segmentation_datasets
from utilities.utils import get_metrics
import matplotlib.pyplot as plt

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
parser.add_argument('--trav-module-weights', default='./weights/espdnetue_c_0-1964.pth', 
                    type=str, help='Weight file of traversability module')
parser.add_argument('--data-test-list', type=str, default='./vision_datasets/traversability_mask/greenhouse_a_test_new.lst',
                    help='Location to save the results')
parser.add_argument('--data-train-list', type=str, default='./vision_datasets/traversability_mask/greenhouse_b_train.lst',
                    help='Location to save the results')
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
    
    # Set thresholds between 0.0 and 1.0
    thresholds = np.arange(0.0, 1.01, 0.05)
    iou_list = np.zeros_like(thresholds)
    acc_list = np.zeros_like(thresholds)
    pre_list = np.zeros_like(thresholds)
    rec_list = np.zeros_like(thresholds)
    for batch in trav_test_loader:
        input_image_tensor = batch['rgb'].to('cuda')
        masks = batch['mask'].to('cuda')

        # Get output
        output = model(input_image_tensor)
        # Convert the output to the label probability -> traversability
        prob = torch.sigmoid(output[2]) / 0.257970929145813
        # Limit the maximum value of the probabilities to 0
        prob[prob > 1.0] = 1.0

        # Match the tensor shape (B, C, H, W) to 'masks' (B, H, W)
        prob = prob.squeeze(dim=1)

        for thresh_index in range(len(thresholds)):
            # Set the threshold
            th = thresholds[thresh_index]

            # Get metrics(iou, acc, pre, rec)
            metrics = get_metrics(masks, prob, th)

            # Accumulate the results
            iou_list[thresh_index] += metrics['iou']
            acc_list[thresh_index] += metrics['acc']
            pre_list[thresh_index] += metrics['pre']
            rec_list[thresh_index] += metrics['rec']

    iou_list /= len(trav_test_loader)
    acc_list /= len(trav_test_loader)
    pre_list /= len(trav_test_loader)
    rec_list /= len(trav_test_loader)

    print(iou_list)
    print(acc_list)
    print(pre_list)
    print(rec_list)

    #
    # Draw a graph of PR curve
    #
    plt.figure()
    lw = 2
    plt.plot(rec_list[:20], pre_list[:20], color='darkorange',
             lw=lw, label='Precision-Recall curve', marker="o")
    plt.plot(rec_list[:20], iou_list[:20], color='c',
             lw=lw, label='IoU-Recall curve', marker="x")
    plt.xlim([0.4, 1.0])
    plt.ylim([0.0, 0.7])
    plt.xlabel('Recall')
    plt.ylabel('Precision / IoU')
    plt.legend(loc="lower right")

    # Save the image
    plt.savefig("roc.pdf")

    #
    # Visualize the mask of the best 
    #
    batch_iter = iter(trav_test_loader)
    batch = next(batch_iter)
    input_image_tensor = batch['rgb'].to('cuda')
    masks = (batch['mask']*255).byte().cpu()

    # Get output
    output = model(input_image_tensor)
    # Convert the output to the label probability -> traversability
    prob = torch.sigmoid(output[2]) / 0.257970929145813
    # Limit the maximum value of the probabilities to 0
    prob[prob > 1.0] = 1.0

    amax = np.argmax(iou_list)
    print("argmax: {}".format(amax))
    threshold = thresholds[amax]
    print("threshold: {}".format(threshold))

    pred_mask = torch.zeros_like(prob, dtype=torch.uint8).cpu()
    pred_mask[prob > threshold] = 255

    print("iou: {}".format(iou_list[amax]))
    print("acc: {}".format(acc_list[amax]))
    print("pre: {}".format(pre_list[amax]))
    print("rec: {}".format(rec_list[amax]))

    # Take a data from the batch and save as images one by one
    for i in range(pred_mask.size(0)):
        pil_image = transforms.ToPILImage()(pred_mask[i])
        pil_image.save('tem_vis_{}_pred.png'.format(i))
        pil_image = transforms.ToPILImage()(masks[i])
        pil_image.save('tem_vis_{}_gt.png'.format(i))
        pil_image = transforms.ToPILImage()(batch['rgb_orig'][i].cpu())
        pil_image.save('tem_vis_{}_input.png'.format(i))

#    im = cv2.imwrite('pred_mask_grid.png', pred_mask_grid)

if __name__=='__main__':
    main()
