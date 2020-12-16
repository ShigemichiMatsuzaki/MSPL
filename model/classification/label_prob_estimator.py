#============================================
__author__ = "Shigemichi Matsuzaki"
__maintainer__ = "Shigemichi Matsuzaki"
#============================================

from torch import nn

class LabelProbEstimator(nn.Module):
    '''
    This class defines a simple architecture for estimating binary label probability of 1-d features
    '''

    def __init__(self, in_channels=16, use_sigmoid=False):
        '''
            Constructor
        '''
        super().__init__()

        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
        self.relu = nn.ReLU()

        self.use_sigmoid = use_sigmoid

        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        :param x: input (1-d tensor)
        '''
        output = self.conv1x1(x)

        if self.use_sigmoid:
            output = self.sigmoid(output)

        return output

        
