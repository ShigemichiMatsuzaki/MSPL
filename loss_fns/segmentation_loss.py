#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
import torch
from torch import nn
from torch.nn import functional as F
import math


class SegmentationLoss(nn.Module):
    def __init__(self, n_classes=21, loss_type='ce', device='cuda', ignore_idx=255, class_weights=None):
        super(SegmentationLoss, self).__init__()
        self.loss_type = loss_type
        self.n_classes = n_classes
        self.device = device
        self.ignore_idx = ignore_idx
        self.smooth = 1e-6
        self.class_wts = class_weights

        if self.loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(weight=self.class_wts)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_idx, weight=self.class_wts)

    def convert_to_one_hot(self, x):
        n, h, w = x.size()
        # remove the 255 index
        x[x == self.ignore_idx] = self.n_classes
        x = x.unsqueeze(1)

        # convert to one hot vector
        x_one_hot = torch.zeros(n, self.n_classes + 1, h, w).to(device=self.device)
        x_one_hot = x_one_hot.scatter_(1, x, 1)

        return x_one_hot[:, :self.n_classes, :, :].contiguous()

    def forward(self, inputs, target):
        if isinstance(inputs, tuple):
            tuple_len = len(inputs)
            assert tuple_len == 2
            loss = 0
            for i in range(tuple_len):
                if target.dim() == 3 and self.loss_type == 'bce':
                    target = self.convert_to_one_hot(target)
                loss_ = self.loss_fn(inputs[i], target)
                loss += loss_
        else:
            if target.dim() == 3 and self.loss_type == 'bce':
                target = self.convert_to_one_hot(target)
            return self.loss_fn(inputs, target)
        return loss

class NIDLoss(nn.Module):
    def __init__(self, image_bin=16, label_bin=4, bw_camera=0.005, bw_label=0.001):
        super(NIDLoss, self).__init__()
        self.K = image_bin
        self.C = label_bin
        self.soft_arg_max_layer = SoftArgMax()
        self.bw_camera = bw_camera
        self.bw_label  = bw_label

        print("K : {}, C : {}".format(self.K, self.C))

    def get_grayscale(self, camera):
        return torch.sum(camera, 1) / 3

    '''
        camera : Tensor of grayscaled camera images (B x H x W)
        label  : Tensor of predicted label images (B x H x W)
        ---
        return : Tensor of joint probabilities (K x C), prior probability for image intensity (K) and label (C)
    '''
    def get_probabilities(self, camera, label):
        # Calculate joint histogram
        num_pixel = camera.size()[1] * camera.size()[2]
        batch_size = camera.size()[0]
        camera_1d = camera.reshape(camera.size()[0], -1)
        label_1d = torch.reshape(label, (label.size()[0], -1) ).to('cuda')

        P_c = torch.zeros(self.K, num_pixel).to('cuda')
        P_l = torch.zeros(self.C, num_pixel).to('cuda')
        
        L_c = 1 / self.K # Size of a bin for the image intensity histogram
        L_l = 1 # Size of a bin for the label histogram
        for k in range(0, self.K):
            mu_k = L_c * (k + 1/2)
            PI_c = torch.sigmoid((camera_1d - mu_k + L_c/2)/self.bw_camera) - torch.sigmoid((camera_1d - mu_k - L_c/2)/self.bw_camera)
            P_c[k] = torch.sum(PI_c, 0)
            if k < self.C:
                PI_l = torch.sigmoid((label_1d - k + L_l/2)/self.bw_label) - torch.sigmoid((label_1d - k - L_l/2)/self.bw_label)
                P_l[k] = torch.sum(PI_l, 0)
       
        norm = num_pixel * batch_size

        return torch.mm(P_c, torch.t(P_l)) / norm, torch.sum(P_c, 1) / norm, torch.sum(P_l, 1) / norm

    def nid(self, p_cl, p_c, p_l, eps=1e-7):
        I = torch.sum(p_cl * (torch.log(p_cl + eps) - torch.log(torch.mm(p_c, torch.t(p_l)) + eps)))
        H = -torch.sum(p_cl * torch.log(p_cl + eps))

        return 1 - I/H
    
    def forward(self, camera, label):
        camera_gray = self.get_grayscale(camera)

        label_amax = self.soft_arg_max_layer(label)
        label_amax = label_amax.reshape(label_amax.size()[0], label_amax.size()[2], label_amax.size()[3]).to('cuda')

        p_cl, p_c, p_l = self.get_probabilities(camera_gray, label_amax)
        p_cl = p_cl / p_cl.sum()
        p_c = p_c / p_c.sum()
        p_l = p_l / p_l.sum()
        nid = self.nid(p_cl, p_c.reshape(p_c.size()[0], 1), p_l.reshape(p_l.size()[0], 1))

        # Rescale the NID value of [0.95, 1.00] to [0.0, 1.0]
        # Because the value range of NID between a camera image and a label image is roughly the range.
        return (nid - 0.95)*20

class SoftArgMax(nn.Module):
    def __init__(self):
        super(SoftArgMax, self).__init__()

    def soft_arg_max(self, A, beta=500, dim=1, epsilon=1e-12):
        '''
            applay softargmax on A and consider mask, return \sum_i ( i * (exp(A_i * beta) / \sum_i(exp(A_i * beta))))
            according to https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
            :param A:
            :param mask:
            :param dim:
            :param epsilon:
            :return:
        '''
        # According to https://discuss.pytorch.org/t/apply-mask-softmax/14212/7
        A_max = torch.max(A, dim=dim, keepdim=True)[0]
        A_exp = torch.exp((A - A_max)*beta)
        A_softmax = A_exp / (torch.sum(A_exp, dim=dim, keepdim=True) + epsilon)
        indices = torch.arange(start=0, end=A.size()[dim]).float().reshape(1, A.size()[dim], 1, 1)
        
    
        return F.conv2d(A_softmax.to('cuda'), indices.to('cuda'))

    def forward(self, x):
        return self.soft_arg_max(x)

class UncertaintyWeightedSegmentationLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, ignore_idx=None, device='cuda'):
        super(UncertaintyWeightedSegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights if class_weights is not None else torch.ones(self.num_classes).to(device)
        self.ignore_idx = ignore_idx
        if self.ignore_idx is not None:
            self.class_weights[self.ignore_idx] = 0.0

    def forward(self, pred, target, u_weight, epsilon=1e-12):
        torch.autograd.set_detect_anomaly(True)
        # Calculate softmax probability
        batch_size = pred.size()[0]
        H = pred.size()[2]
        W = pred.size()[3]

        logp = -F.log_softmax(pred, dim=1)

        # Losses on the pixels of ignore_idx do not contribute to the loss function
        logp = logp * self.class_weights.reshape(1, self.class_weights.size()[0], 1, 1).expand(batch_size, -1, H, W)

        # Gather log probabilities with respect to target
        logp = logp.gather(1, target.view(batch_size, 1, H, W))

        # Multiply with weights
        logp = logp * torch.exp(-u_weight.reshape(batch_size, 1, H, W))

        # Rescale so that loss is in approx. same interval
        
        return logp.mean()

class PixelwiseKLD(nn.Module):
    def __init__(self):
        super(PixelwiseKLD, self).__init__()

    def forward(self, dist1, dist2):
        # Calculate probability and log probability
        p1    = F.softmax(dist1, dim=1)
        logp1 = F.log_softmax(dist1, dim=1)
        logp2 = F.log_softmax(dist2, dim=1)

        kld_i = p1 * logp1 - p1 * logp2

        return torch.sum(kld_i, dim=1)

class SelectiveBCE(nn.Module):
    def __init__(self, with_logits=True, device='cuda'):
        super(SelectiveBCE, self).__init__()
        if with_logits:
            self.loss_func = nn.BCEWithLogitsLoss(reduction='none').to(device)
        else:
            self.loss_func = nn.BCELoss(reduction='none').to(device)

    def forward(self, input, target, bool_index_mask=None):
        '''
        :param input: Input (Tensor of size (B, 1, H, W))
        :param target: Target label (Tensor)
        :param mask: Mask indicating pixels to be used in loss calculation (Tensor)
        :return: loss value
        '''

        # Get non-reduced loss values
        loss = self.loss_func(input, torch.reshape(target, input.size()).float())

        if bool_index_mask is not None:
            # Extract the values of pixels whose mask is 1
            loss = loss[bool_index_mask]

        return loss.mean()
