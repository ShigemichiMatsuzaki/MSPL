# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import torch
from torch.nn import init
from nn_layers.espnet_utils import *
from nn_layers.efficient_pyramid_pool import EfficientPyrPool
from nn_layers.efficient_pt import EfficientPWConv
from nn_layers.espnet_utils import C, CBR
from model.classification.espnetv2 import EESPNet
from utilities.print_utils import *
from torch.nn import functional as F
import copy

class FusionGate(nn.Module):
    '''
    Gate to fuse RGB and depth features
    '''
    def __init__(self, nchannel, is_trainable=True):
        super().__init__()

        # Channel size of input features (must be the same)
        self.nchannel = nchannel

        self.conv_1x1 = C(nIn=2 * self.nchannel, nOut=self.nchannel, kSize=1)
        self.sigmoid  = torch.nn.Sigmoid()

        self.is_trainable = is_trainable
        
    def forward(self, rgb, depth):
        if self.is_trainable:
    #      print(rgb.size())
    #      print(depth.size())
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

class ESPDNetwithUncertaintyEstimation(nn.Module):
    '''
    This class defines the ESPDNet architecture for the Semantic Segmenation with uncertainty estimation

    aux_layer : A number of the layer from which the auxiliary branch is made.
      0: bu_dec_l1, 1: bu_dec_l2, 2: bu_dec_l3
    '''
    def __init__(self, args, classes=21, dataset='pascal', dense_fuse=False, trainable_fusion=True, aux_layer=2):
        super().__init__()

        # =============================================================
        #                       BASE NETWORK
        # =============================================================
        #
        # RGB
        #
        self.base_net = EESPNet(args) #imagenet model
        del self.base_net.classifier
        del self.base_net.level5
        del self.base_net.level5_0
        config = self.base_net.config

        #
        # Depth
        #
        tmp_args = copy.deepcopy(args)
        tmp_args.channels = 1
        self.depth_base_net = EESPNet(tmp_args)
        del self.depth_base_net.classifier
        del self.depth_base_net.level5
        del self.depth_base_net.level5_0

        self.fusion_gate_level1 = FusionGate(nchannel=32, is_trainable=trainable_fusion)
        self.fusion_gate_level2 = FusionGate(nchannel=128, is_trainable=trainable_fusion)
        self.fusion_gate_level3 = FusionGate(nchannel=256, is_trainable=trainable_fusion)
        self.fusion_gate_level4 = FusionGate(nchannel=512, is_trainable=trainable_fusion)

        #=============================================================
        #                   SEGMENTATION NETWORK
        #=============================================================
        dec_feat_dict={
            'pascal': 16,
            'city': 16,
            'coco': 32,
            'greenhouse': 16,
            'ishihara': 16,
            'sun': 16,
            'camvid': 16
        }
        base_dec_planes = dec_feat_dict[dataset]
        dec_planes = [4*base_dec_planes, 3*base_dec_planes, 2*base_dec_planes, classes]
        pyr_plane_proj = min(classes //2, base_dec_planes)

        self.bu_dec_l1 = EfficientPyrPool(in_planes=config[3], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[0])
        self.bu_dec_l2 = EfficientPyrPool(in_planes=dec_planes[0], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[1])
        self.bu_dec_l3 = EfficientPyrPool(in_planes=dec_planes[1], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[2])
        self.bu_dec_l4 = EfficientPyrPool(in_planes=dec_planes[2], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[3], last_layer_br=False)

        self.merge_enc_dec_l2 = EfficientPWConv(config[2], dec_planes[0])
        self.merge_enc_dec_l3 = EfficientPWConv(config[1], dec_planes[1])
        self.merge_enc_dec_l4 = EfficientPWConv(config[0], dec_planes[2])

        self.bu_br_l2 = nn.Sequential(nn.BatchNorm2d(dec_planes[0]),
                                      nn.PReLU(dec_planes[0])
                                      )
        self.bu_br_l3 = nn.Sequential(nn.BatchNorm2d(dec_planes[1]),
                                      nn.PReLU(dec_planes[1])
                                      )
        self.bu_br_l4 = nn.Sequential(nn.BatchNorm2d(dec_planes[2]),
                                      nn.PReLU(dec_planes[2])
                                      )

        # Auxiliary branch
        self.aux_layer = aux_layer
        if aux_layer >= 0 and aux_layer < 3:
            self.aux_decoder = EfficientPyrPool(in_planes=dec_planes[aux_layer], proj_planes=pyr_plane_proj,
                                                out_planes=dec_planes[3], last_layer_br=False)

        #self.upsample =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.init_params()
        self.dense_fuse = dense_fuse

    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def get_basenet_params(self):
        modules_base = [self.base_net]
        for i in range(len(modules_base)):
            for m in modules_base[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_depth_encoder_params(self):
        modules_depth = [self.depth_base_net]
        for i in range(len(modules_depth)):
            for m in modules_depth[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_segment_params(self):
        modules_seg = [self.bu_dec_l1, self.bu_dec_l2, self.bu_dec_l3, self.bu_dec_l4,
                       self.merge_enc_dec_l4, self.merge_enc_dec_l3, self.merge_enc_dec_l2,
                       self.bu_br_l4, self.bu_br_l3, self.bu_br_l2]
        for i in range(len(modules_seg)):
            for m in modules_seg[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def forward(self, x, x_d=None):
        '''
        :param x: Receives the input RGB image
        :param x_d: Receives the input Depth image
        :return: a C-dimensional vector, C=# of classes
        '''

        if x_d is not None:
            pass
#            print(x.size(), x_d.size())

        x_size = (x.size(2), x.size(3)) # Width and height

        # 
        # First conv
        #
        enc_out_l1 = self.base_net.level1(x)  # 112
        if not self.base_net.input_reinforcement:
            del x
            x = None

        if x_d is not None:
            d_enc_out_l1 = self.depth_base_net.level1(x_d) # Depth
    
            # Fusion level 1
            # enc_out_l1 += d_enc_out_l1
            enc_out_l1 = self.fusion_gate_level1(enc_out_l1, d_enc_out_l1)

        # 
        # Second layer (Strided EESP)
        #
        enc_out_l2 = self.base_net.level2_0(enc_out_l1, x)  # 56
        if x_d is not None:
            d_enc_out_l2 = self.depth_base_net.level2_0(d_enc_out_l1)
    
            # Fusion level 2
            # enc_out_l2 += d_enc_out_l2
            enc_out_l2 = self.fusion_gate_level2(enc_out_l2, d_enc_out_l2)

        # 
        # Third layer 1 (Strided EESP)
        #
        enc_out_l3_0 = self.base_net.level3_0(enc_out_l2, x)  # down-sample -> 28
        if x_d is not None: 
            d_enc_out_l3_0 = self.depth_base_net.level3_0(d_enc_out_l2)  # down-sample -> 28
        # 
        # EESP
        #
        for i, (layer, dlayer) in enumerate(zip(self.base_net.level3, self.depth_base_net.level3)):
            if i == 0:
                enc_out_l3 = layer(enc_out_l3_0)
                if x_d is not None: 
                    d_enc_out_l3 = dlayer(d_enc_out_l3_0)
                    if self.dense_fuse:
                        # enc_out_l3 += d_enc_out_l3
                        enc_out_l3 = self.fusion_gate_level3(enc_out_l3, d_enc_out_l3)
            else:
                enc_out_l3 = dlayer(enc_out_l3)
                if x_d is not None: 
                    d_enc_out_l3 = dlayer(d_enc_out_l3)
                    if self.dense_fuse:
                        # enc_out_l3 += d_enc_out_l3
                        enc_out_l3 = self.fusion_gate_level3(enc_out_l3, d_enc_out_l3)

        if x_d is not None and not self.dense_fuse:
            # Fusion level 3
            # enc_out_l3 += d_enc_out_l3
            enc_out_l3 = self.fusion_gate_level3(enc_out_l3, d_enc_out_l3)

        # 
        # Forth layer 1 (Strided EESP)
        #
        enc_out_l4_0 = self.base_net.level4_0(enc_out_l3, x)  # down-sample -> 14
        if x_d is not None: 
            d_enc_out_l4_0 = self.depth_base_net.level4_0(d_enc_out_l3)  # down-sample -> 14
        # 
        # EESP
        #
        for i, (layer, dlayer) in enumerate(zip(self.base_net.level4, self.depth_base_net.level4)):
            if i == 0:
                enc_out_l4 = layer(enc_out_l4_0)
                if x_d is not None: 
                    d_enc_out_l4 = dlayer(d_enc_out_l4_0)
                    if self.dense_fuse:
                        # enc_out_l4 += d_enc_out_l4
                        enc_out_l4 = self.fusion_gate_level4(enc_out_l4, d_enc_out_l4)
            else:
                enc_out_l4 = layer(enc_out_l4)
                if x_d is not None: 
                    d_enc_out_l4 = dlayer(d_enc_out_l4)
                    if self.dense_fuse:
                        # enc_out_l4 += d_enc_out_l4
                        enc_out_l4 = self.fusion_gate_level4(enc_out_l4, d_enc_out_l4)

        if x_d is not None and not self.dense_fuse:
            # Fusion level 4
            # enc_out_l4 += d_enc_out_l4
            enc_out_l4 = self.fusion_gate_level4(enc_out_l4, d_enc_out_l4)


        # *** 5th layer is for and classification and removed for segmentation ***

        # bottom-up decoding
        bu_out = self.bu_dec_l1(enc_out_l4)
        if self.aux_layer == 0:
            aux_out = self.aux_decoder(bu_out)

        # Decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l3_proj = self.merge_enc_dec_l2(enc_out_l3)
        bu_out = enc_out_l3_proj + bu_out
        bu_out = self.bu_br_l2(bu_out)
        bu_out = self.bu_dec_l2(bu_out)
        if self.aux_layer == 1:
            aux_out = self.aux_decoder(bu_out)

        #decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l2_proj = self.merge_enc_dec_l3(enc_out_l2)
        bu_out = enc_out_l2_proj + bu_out
        bu_out = self.bu_br_l3(bu_out)
        bu_out = self.bu_dec_l3(bu_out)
        if self.aux_layer == 2:
            aux_out = self.aux_decoder(bu_out)

        # decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l1_proj = self.merge_enc_dec_l4(enc_out_l1)
        bu_out = enc_out_l1_proj + bu_out
        bu_out = self.bu_br_l4(bu_out)
        bu_out  = self.bu_dec_l4(bu_out)

        return (F.interpolate(bu_out, size=x_size, mode='bilinear', align_corners=True), 
                F.interpolate(aux_out, size=x_size, mode='bilinear', align_corners=True))


def espdnetue_seg(args, load_entire_weights=False):
    classes = args.classes
    scale=args.s
    weights = args.weights
    print(weights)
    #depth_weights = 'results_segmentation/model_espnetv2_greenhouse/s_2.0_sch_hybrid_loss_ce_res_480_sc_0.5_2.0_autoenc/20200401-114045/espnetv2_2.0_480_checkpoint.pth.tar'
    depth_weights = weights
    dataset=args.dataset
    trainable_fusion=args.trainable_fusion
    dense_fuse=args.dense_fuse
    model = ESPDNetwithUncertaintyEstimation(args, classes=classes, dataset=dataset, dense_fuse=dense_fuse, trainable_fusion=trainable_fusion)
    if weights:
        import os
        if os.path.isfile(weights):
            num_gpus = torch.cuda.device_count()
            device = 'cuda' if num_gpus >= 1 else 'cpu'
            pretrained_dict = torch.load(weights, map_location=torch.device(device))
        else:
            print_error_message('Weight file does not exist at {}. Please check. Exiting!!'.format(weights))
            exit()

        print_info_message('Loading pretrained basenet model weights')
        # Load pretrained weights for RGB
        model_dict = model.state_dict()
        print(pretrained_dict.keys())
        if load_entire_weights:
            overlap_dict = {k: v for k, v in pretrained_dict.items() 
                            if k in model_dict}
        else:
            basenet_dict = model.base_net.state_dict()
            overlap_dict = {k.replace("base_net.", ""): v for k, v in pretrained_dict.items() 
                            if k.replace("base_net.", "") in basenet_dict}

        if len(overlap_dict) == 0:
            print_error_message('No overlaping weights between model file and pretrained weight file. Please check')
            exit()

        print_info_message('{:.2f} % of weights copied from basenet to segnet'.format(len(overlap_dict) * 1.0/len(model_dict) * 100))
        if load_entire_weights:
            model_dict.update(overlap_dict)
            model.load_state_dict(model_dict)
        else:
            basenet_dict.update(overlap_dict)
            model.base_net.load_state_dict(basenet_dict)

        print_info_message('Pretrained basenet model loaded!!')
    else:
        print_warning_message('Training from scratch!!')

    if depth_weights:
        import os
        if os.path.isfile(depth_weights):
            num_gpus = torch.cuda.device_count()
            device = 'cuda' if num_gpus >= 1 else 'cpu'
            pretrained_dict = torch.load(depth_weights, map_location=torch.device(device))
        else:
            print_error_message('Weight file does not exist at {}. Please check. Exiting!!'.format(depth_weights))
            exit()

        # Load pretrained weights for RGB
        dbasenet_dict = model.depth_base_net.state_dict()
        overlap_dict = {k.lstrip('base_net.'): v for k, v in pretrained_dict.items() if k.lstrip('base_net.') in dbasenet_dict}
        overlap_dict['level1.conv.weight'] = torch.mean(overlap_dict['level1.conv.weight'], dim=1, keepdim=True)
        if len(overlap_dict) == 0:
            print_error_message('No overlaping weights between model file and pretrained weight file. Please check')
            exit()
        dbasenet_dict.update(overlap_dict)
        model.depth_base_net.load_state_dict(dbasenet_dict)
        print_info_message('Pretrained depth basenet model loaded!!')

    return model

if __name__ == "__main__":

    from utilities.utils import compute_flops, model_parameters
    import torch
    import argparse

    parser = argparse.ArgumentParser(description='Testing')
    args = parser.parse_args()

    args.classes = 21
    args.s = 2.0
    args.weights='../classification/model_zoo/espnet/espnetv2_s_2.0_imagenet_224x224.pth'
    args.dataset='pascal'

    input = torch.Tensor(1, 3, 384, 384)
    model = espdnet_seg(args)
    from utilities.utils import compute_flops, model_parameters
    print_info_message(compute_flops(model, input=input))
    print_info_message(model_parameters(model))
    out = model(input)
    print_info_message(out.size())
