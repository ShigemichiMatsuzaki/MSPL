import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities.print_utils import *
import matplotlib.pyplot as plt

# Datasets
from data_loader.segmentation.cityscapes import CityscapesSegmentation, CITYSCAPE_CLASS_LIST
from data_loader.segmentation.camvid import CamVidSegmentation, CAMVID_CLASS_LIST
from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation, GREENHOUSE_CLASS_LIST
from data_loader.segmentation.ishihara_rgbd import IshiharaRGBDSegmentation, ISHIHARA_RGBD_CLASS_LIST
from data_loader.segmentation.gta5 import GTA5
from data_loader.segmentation.freiburg_forest import FreiburgForestDataset, FOREST_CLASS_LIST, color_encoding

# Network
from model.classification import espnetv2 as net

from tqdm import tqdm

class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        output_g = F.adaptive_avg_pool2d(module_out, output_size=1)
        output_g = F.dropout(output_g, p=0.2, training=False)
        output_1x1 = output_g.view(output_g.size(0), -1)
        output_1x1 = output_1x1.cpu().detach()
#        print(output_1x1.size())
        self.outputs.append(output_1x1.numpy()[0])
        
    def clear(self):
        self.outputs = []

def get_features(model, dataset, dataset_label, device='cuda'):
    model.eval()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                              pin_memory=True, num_workers=4)

    features = []
    dataset_labels = []

    save_output = SaveOutput()
    model.level5[4].register_forward_hook(save_output)

    with tqdm(total=len(data_loader)) as pbar:
        for i, batch in enumerate(tqdm(data_loader)):
    #        print("Let's go {}".format(i))
            image = batch[0].to(device)
            output = model(image)
    
    #        features.append(feature.numpy())
            dataset_labels.append(dataset_label)
    
    pbar.close()

    return np.array(save_output.outputs), np.array(dataset_labels)

def main(args):

    dataset_cityscapes = CityscapesSegmentation(
        root='./vision_datasets/cityscapes', train=False, coarse=False, scale=1.0, size=(224, 224))
    dataset_camvid = CamVidSegmentation(
        root='./vision_datasets/camvid', list_name='train_camvid.txt', train=True, scale=1.0, size=(224, 224))
    dataset_greenhouse = GreenhouseRGBDSegmentation(
        root='./vision_datasets/greenhouse', list_name='train_greenhouse_no_gamma.lst', train=True, use_depth=False,
        scale=1.0, size=(224, 224))
    dataset_greenhouse2 = GreenhouseRGBDSegmentation(
        root='./vision_datasets/greenhouse', list_name='val_greenhouse2.lst', train=True, use_depth=False,
        scale=1.0, size=(224, 224))
#    dataset_ishihara = IshiharaRGBDSegmentation(
#        root='./vision_datasets/ishihara_rgbd', list_name='ishihara_rgbd_val.txt', train=False, 
#        scale=1.0, size=(224,224), use_depth=False)
    dataset_gta5 = GTA5(
        root='./vision_datasets/gta5', list_name='val_small.lst', scale=1.0, size=(224,224))
    dataset_forest = FreiburgForestDataset(
        train=True, size=(224, 224), scale=1.0, normalize=True)

    model = net.EESPNet(args)

    if not args.weights:
        print_info_message('Grabbing location of the ImageNet weights from the weight dictionary')
        from model.weight_locations.classification import model_weight_map

        weight_file_key = '{}_{}'.format(args.model, args.s)
        assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
        args.weights = model_weight_map[weight_file_key]

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus >=1 else 'cpu'
    weight_dict = torch.load(args.weights, map_location=torch.device(device))
    model.load_state_dict(weight_dict)

    model = model.cuda()
#    if num_gpus >= 1:
#        model = torch.nn.DataParallel(model)
#        model = model.cuda()
#        if torch.backends.cudnn.is_available():
#            import torch.backends.cudnn as cudnn
#            cudnn.benchmark = True
#            cudnn.deterministic = True
    
    features_gta5, labels_gta5 = get_features(model, dataset_gta5, 0)
    features_cityscapes, labels_cityscapes = get_features(model, dataset_cityscapes, 1)
    features_camvid, labels_camvid = get_features(model, dataset_camvid, 2)
    features_greenhouse, labels_greenhouse = get_features(model, dataset_greenhouse, 3)
    features_greenhouse2, labels_greenhouse2 = get_features(model, dataset_greenhouse2, 4)
    features_forest, labels_forest = get_features(model, dataset_forest, 5)

#    features_ishihara, labels_ishihara = get_features(model, dataset_ishihara, 0)

#    features = np.concatenate([features_cityscapes, features_camvid, features_greenhouse])
    features = np.concatenate([features_cityscapes, features_greenhouse2, features_camvid, features_greenhouse, features_gta5, features_forest])
#    features = np.concatenate([features_camvid, features_greenhouse])
#    target = np.concatenate([labels_cityscapes, labels_camvid, labels_greenhouse])
    target = np.concatenate([labels_cityscapes, labels_greenhouse2, labels_camvid, labels_greenhouse, labels_gta5, labels_forest])
#    target = np.concatenate([labels_camvid, labels_greenhouse])

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state = 0, perplexity = 20, n_iter = 1000)
    feature_embedded = tsne.fit_transform(features)

    plt.scatter(feature_embedded[:, 0], feature_embedded[:, 1],
                c=target, cmap='jet')
    plt.colorbar()
    plt.savefig('figure.png')


if __name__ == '__main__':
    from commons.general_details import classification_models, classification_datasets

    parser = argparse.ArgumentParser(description='Testing efficient networks')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--data', default='', help='path to dataset')
    parser.add_argument('--dataset', default='imagenet', help='Name of the dataset', choices=classification_datasets)
    parser.add_argument('--batch-size', default=512, type=int, help='mini-batch size (default: 512)')
    parser.add_argument('--num-classes', default=1000, type=int, help='# of classes in the dataset')
    parser.add_argument('--s', default=2.0, type=float, help='Width scaling factor')
    parser.add_argument('--weights', type=str, default='', help='weight file')
    parser.add_argument('--inpSize', default=224, type=int, help='Input size')
    ##Select a model
    parser.add_argument('--model', default='espnetv2', choices=classification_models, help='Which model?')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')

    args = parser.parse_args()
    main(args)