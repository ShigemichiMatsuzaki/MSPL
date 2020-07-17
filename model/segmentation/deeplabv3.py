# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.segmentation import deeplabv3_resnet101
from collections import OrderedDict

import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
print(sys.path)

from model.segmentation.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from nn_layers.aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=20):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes

        ## self.project_dir = project_dir
        ## self.create_model_dirs()

        self.resnet = ResNet50_OS16() # NOTE! specify the type of ResNet here
        self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.interpolate(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
    
    def get_basenet_params(self):
        modules_base = [self.resnet]
        for i in range(len(modules_base)):
            for m in modules_base[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_segment_params(self):
        modules_seg = [self.aspp]
        for i in range(len(modules_seg)):
            for m in modules_seg[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
        pass

def deeplabv3_seg(num_classes=20, weights='pretrained_models/resnet/resnet50-19c8e357.pth', resnet_only=False):
    model = DeepLabV3(num_classes)
    
    import os
    if os.path.isfile(weights):
        num_gpus = torch.cuda.device_count()
        device = 'cuda' if num_gpus >= 1 else 'cpu'
        pretrained_dict = torch.load(weights, map_location=torch.device(device))
    else:
        print('Weight file does not exist at {}. Please check. Exiting!!'.format(weights))
        exit()

    # If 'resnet_only' flag is True, the target is only the feature extraction network i.e., ResNet
    #   else, the target is the entire DeepLab network
    model_dict = model.resnet.state_dict() if resnet_only else model.state_dict()
    overlap_dict = {k: v for k, v in pretrained_dict.items() 
                    if k in model_dict}

    model_dict.update(overlap_dict)
    model.load_state_dict(model_dict)

    print(overlap_dict.keys())
    print('{:.2f} % of weights copied from basenet to segnet'.format(len(overlap_dict) * 1.0/len(model_dict) * 100))

    return model

if __name__ == '__main__':
    # model = deeplabv3_seg(num_classes=13, weights='/tmp/runs/model_deeplabv3_camvid/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_rgb/20200704-230619/deeplabv3_2.0_480_best.pth')
    print(torch.backends.cudnn.version())

#    model = deeplabv3_resnet101(aux_loss=True)
#    model.eval()
#    model_dict = model.state_dict()
#
#    # print(model_dict.keys())
#
#    out = model(torch.rand(1, 3, 256, 480))
#
#    print(isinstance(out, OrderedDict))
#    print(out.values())