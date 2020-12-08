import torch
import argparse
from model.segmentation.espdnet_ue import espdnetue_seg2
from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation, GREENHOUSE_CLASS_LIST

from commons.general_details import segmentation_models, segmentation_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--ignore-idx', type=int, default=255, help='Index or label to be ignored during training')

parser.add_argument('--s', type=float, default=2.0, help='Factor by which channels will be scaled')
# model details
parser.add_argument('--freeze-bn', action='store_true', default=False, help='Freeze BN params or not')

# dataset and result directories
parser.add_argument('--dataset', type=str, default='greenhouse', choices=segmentation_datasets, help='Datasets')
parser.add_argument('--weights', type=str, default='', help='Name of weight file to load')
# model related params
parser.add_argument('--model', default='espdnetue', choices=segmentation_models,
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

w = 480
h = 256
batch_size = 1
filename = "espdnet_ue_uest.pt"

args.classes = len(GREENHOUSE_CLASS_LIST)

# Intialize ENet
model = espdnetue_seg2(args, load_entire_weights=True, fix_pyr_plane_proj=True)

model.to('cuda')
model.eval()
# Trace the network with random data
inputs = torch.rand(1, 3, h, w).to('cuda')
print(inputs)
traced_net = torch.jit.trace(model, inputs)
print("Trace done")

# Save the module
traced_net.save(filename)
print(filename + " is exported")