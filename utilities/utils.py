
import os
import torch
from utilities.print_utils import print_info_message
import numpy as np
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import logging
from loss_fns.segmentation_loss import PixelwiseKLD

#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

'''
This file is mostly adapted from the PyTorch ImageNet example
'''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

'''
Utility to save checkpoint or not
'''
def save_checkpoint(state, is_best, dir, extra_info='model', epoch=-1):
    check_pt_file = dir + os.sep + str(extra_info) + '_checkpoint.pth.tar'
    torch.save(state, check_pt_file)
    if is_best:
        #We only need best models weight and not check point states, etc.
        torch.save(state['state_dict'], dir + os.sep + str(extra_info) + '_best.pth')
    if epoch != -1:
        torch.save(state['state_dict'], dir + os.sep + str(extra_info) + '_ep_' + str(epoch) + '.pth')

    print_info_message('Checkpoint saved at: {}'.format(check_pt_file))


'''
Function to compute model parameters
'''
def model_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters()])/ 1e6

'''
function to compute flops
'''
def compute_flops(model, input=None):
    from utilities.flops_compute import add_flops_counting_methods
    input = input if input is not None else torch.Tensor(1, 3, 224, 224)
    model = add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()

    _ = model(input)

    flops = model.compute_average_flops_cost()  # + (model.classifier.in_features * model.classifier.out_features)
    flops = flops / 1e6 / 2
    return flops

def in_training_visualization_img(model, images, depths=None, labels=None, predictions=None, class_encoding=None, writer=None, epoch=None, data=None, device=None):
    # Make predictions!
    if predictions is None:
        model.eval()
        with torch.no_grad():
            if depths is not None:
                print("Eval. depths:{}".format(depths.size()))
                predictions = model(images, depths)
            else:
                predictions = model(images)
    
    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    if type(predictions) is tuple:
        kld_layer = PixelwiseKLD()
        f_pred = predictions[0] + 0.5 * predictions[1]
        kld = kld_layer(predictions[0], predictions[1])
        kld = (-kld / torch.max(kld).item() + 1)# * 255# Scale to [0, 255]
        kld = torch.reshape(kld, (kld.size(0), 1, kld.size(1), kld.size(2)))
        kld = torchvision.utils.make_grid(kld.cpu()).numpy()
        writer.add_image(data + '/kld', kld, epoch)
        _, predictions = torch.max(f_pred, dim=1)
    elif isinstance(predictions, OrderedDict):
        f_pred = predictions['out']
        if len(predictions) == 2:
            kld_layer = PixelwiseKLD()
            f_pred += 0.5 * predictions['aux']
            kld = kld_layer(predictions['out'], predictions['aux'])
            kld = (-kld / torch.max(kld).item() + 1)# * 255# Scale to [0, 255]
            kld = torch.reshape(kld, (kld.size(0), 1, kld.size(1), kld.size(2)))
            kld = torchvision.utils.make_grid(kld.cpu()).numpy()
            writer.add_image(data + '/kld', kld, epoch)

        _, predictions = torch.max(f_pred, dim=1)       
    elif len(predictions.size()) == 3:
        pass
    else:
        _, predictions = torch.max(predictions.data, dim=1)


    print(predictions.size())
    print(labels.size())
    
    # label_to_rgb : Sequence of processes
    #  1. LongTensorToRGBPIL(tensor) -> PIL Image : Convert label tensor to color map
    #  2. transforms.ToTensor() -> Tensor : Convert PIL Image to a tensor
    label_to_rgb = transforms.Compose([
        LongTensorToRGBPIL(class_encoding)#,
#        transforms.ToTensor()
    ])

    # Do transformation of label tensor and prediction tensor
    color_train       = batch_transform(labels.data.cpu(), label_to_rgb)
    color_predictions = batch_transform(predictions.data.cpu(), label_to_rgb)

    write_summary_batch(images.data.cpu(), color_train, color_predictions, writer, epoch, data)

def in_training_visualization_2(model, images, depths=None, labels=None, class_encoding=None, writer=None, epoch=None, data=None, device=None):
    # Make predictions!
    model.eval()
    with torch.no_grad():
        if depths is not None:
            predictions = model(images, depths)
        else:
            predictions = model(images)
    
    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions[0].data, 1)
    
       # label_to_rgb : Sequence of processes
    #  1. LongTensorToRGBPIL(tensor) -> PIL Image : Convert label tensor to color map
    #  2. transforms.ToTensor() -> Tensor : Convert PIL Image to a tensor
    label_to_rgb = transforms.Compose([
        LongTensorToRGBPIL(class_encoding)#,
#        transforms.ToTensor()
    ])

    # Do transformation of label tensor and prediction tensor
    color_train       = batch_transform(labels.data.cpu(), label_to_rgb)
    color_predictions = batch_transform(predictions.cpu(), label_to_rgb)

    write_summary_batch(images.data.cpu(), color_train, color_predictions, writer, epoch, data)

def batch_transform(batch, transform):
    """Applies a transform to a batch of samples.
    Keyword arguments:
    - batch (): a batch os samples
    - transform (callable): A function/transform to apply to ``batch``
    """

    # Convert the single channel label to RGB in tensor form
    # 1. torch.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]

    return torch.stack(transf_slices)

def write_summary_batch(images, train_labels, pred_labels, writer, epoch, data):
    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    train_labels = torchvision.utils.make_grid(train_labels).numpy()
    pred_labels = torchvision.utils.make_grid(pred_labels).numpy()

#    images = np.transpose(images, (1, 2, 0))
#    train_labels = np.transpose(train_labels, (1, 2, 0))
#    pred_labels = np.transpose(pred_labels, (1, 2, 0))

    writer.add_image(data + '/images', images, epoch)
    writer.add_image(data + '/train_labels', train_labels, epoch)
    writer.add_image(data + '/pred_labels', pred_labels, epoch)

class LongTensorToRGBPIL(object):
    """Converts a ``torch.LongTensor`` to a ``PIL image``.

    The input is a ``torch.LongTensor`` where each pixel's value identifies the
    class.

    Keyword arguments:
    - rgb_encoding (``OrderedDict``): An ``OrderedDict`` that relates pixel
    values, class names, and class colors.

    """
    def __init__(self, rgb_encoding):
        self.rgb_encoding = rgb_encoding

    def __call__(self, tensor):
        """Performs the conversion from ``torch.LongTensor`` to a ``PIL image``

        Keyword arguments:
        - tensor (``torch.LongTensor``): the tensor to convert

        Returns:
        A ``PIL.Image``.

        """
        # Check if label_tensor is a LongTensor
        if not isinstance(tensor, torch.LongTensor):
            raise TypeError("label_tensor should be torch.LongTensor. Got {}"
                            .format(type(tensor)))
        # Check if encoding is a ordered dictionary
        if not isinstance(self.rgb_encoding, OrderedDict):
            raise TypeError("encoding should be an OrderedDict. Got {}".format(
                type(self.rgb_encoding)))

        # label_tensor might be an image without a channel dimension, in this
        # case unsqueeze it
        if len(tensor.size()) == 2:
            tensor.unsqueeze_(0)

        # Initialize
        color_tensor = torch.ByteTensor(3, tensor.size(1), tensor.size(2))

        for index, (class_name, color) in enumerate(self.rgb_encoding.items()):
            # Get a mask of elements equal to index
            mask = torch.eq(tensor, index).squeeze_()
            # Fill color_tensor with corresponding colors
            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value)

#        return ToPILImage()(color_tensor)
        return color_tensor

def calc_cls_class_weight(data_loader, class_num, inverted=False):
    class_array = np.zeros(class_num).astype(np.float32)

    for n, batch in enumerate(data_loader):
        cls_ids = batch[1].numpy()
        for i in range(0, class_num):
            class_array[i] += (cls_ids == i).sum()

    class_array /= class_array.sum() # normalized
#    class_array = 1 - class_array 

    if inverted:
        return np.exp(class_array)/np.sum(np.exp(class_array)) #/ class_array.sum()
    else:
        return 1/(class_array + 1e-10)

def set_logger(output_dir=None, log_file=None, debug=False):
    head = '%(asctime)-15s Host %(message)s'
    logger_level = logging.INFO if not debug else logging.DEBUG
    if all((output_dir, log_file)) and len(log_file) > 0:
        logger = logging.getLogger()
        log_path = os.path.join(output_dir, log_file)
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logger_level)
    else:
        logging.basicConfig(level=logger_level, format=head)
        logger = logging.getLogger()
    return logger

