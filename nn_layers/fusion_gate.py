# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import torch
import torch.nn as nn
from torch.nn import init
from nn_layers.espnet_utils import C

class FusionGate(nn.Module):
    '''
    Gate to fuse RGB and depth features
    '''
    def __init__(self, nchannel, is_trainable=True):
        super().__init__()

        # Channel size of input features (must be the same)
        self.nchannel = nchannel

        self.conv_1x1 = C(nIn=2 * self.nchannel, nOut=self.nchannel, kSize=1)
        self.sigmoid  = nn.Sigmoid()

        self.is_trainable = is_trainable
        
    def forward(self, rgb, depth):
        if self.is_trainable:
          # 1. Concat RGB and depth features
          output = torch.cat((rgb, depth), 1)
  
          # 2. 1x1 convolution
          output = self.conv_1x1(output)
  
          # 3. Sigmoid
          weight = self.sigmoid(output)
  
          # 4. Multiply each features by the yielded weights
          size = weight.size()
          w_rgb = rgb * weight
          w_depth = depth * (torch.ones(size).to('cuda') - weight)
  
          output = w_rgb + w_depth

        else:
          output = rgb + depth

        return output