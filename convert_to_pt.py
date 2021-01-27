import torch
import argparse
import datetime
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
parser.add_argument('--trav-module-weights', default='./results_segmentation/espdnetue_best.pth', type=str, help='Weight file of traversability module')

args = parser.parse_args()

w = 480
h = 256
batch_size = 1

args.classes = len(GREENHOUSE_CLASS_LIST)

now = datetime.datetime.now()
now += datetime.timedelta(hours=9)
timestr = now.strftime("%Y%m%d-%H%M%S")

# Intialize ENet
if args.model_name == 'espdnetue_trav':
    from model.segmentation.espdnet_ue_traversability import espdnetue_seg
    model = espdnetue_seg(args, load_entire_weights=True, fix_pyr_plane_proj=True, spatial=False)
    filename = "espdnet_ue_trav_{}.pt".format(timestr)
else:
    from model.segmentation.espdnet_ue import espdnetue_seg2
    model = espdnetue_seg2(args, load_entire_weights=True, fix_pyr_plane_proj=True)
    filename = "espdnet_ue_{}.pt".format(timestr)

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