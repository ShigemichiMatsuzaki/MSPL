import torch
from PIL import Image
import numpy as np
from segmentation_loss import SegmentationLoss, NIDLoss
from torchvision.transforms import functional as F

def main():
    weight = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    print(weight)
    print(weight.reshape(1, 5, 1, 1).expand(3, -1, 5, 5))
#    image_orig = Image.open('/tmp/dataset/greenhouse/train/26_0_000000.png')
#    label_orig = Image.open('/tmp/dataset/greenhouse/trainannot/26_0_000000.png')

#    nid = NIDLoss(image_bin=32, label_bin=4)
#    for i in range(-3, 4):
#        print()
#        print("==== angle : {} ====".format(30*i))
#        if i != 0:
#            label = label_orig.rotate(30*i)
#        else:
#            label = label_orig
#
#        image = np.array(image_orig)#.transpose(2, 0, 1)
#        label = np.array(label)
#    
##        image = torch.tensor(image, dtype=torch.float)#.reshape(1, image.shape[0], image.shape[1], image.shape[2])
#        image = F.to_tensor(image) # convert to tensor (values between 0 and 1)
##        print("image type : {}".format(image.dtype))
##        print("image max : {}".format(image.mean()))
##        label = torch.tensor(label, dtype=torch.float)#.reshape(1, label.shape[0], label.shape[1], label.shape[2])
#        label = torch.Tensor(np.array(label).astype(np.int64))
##        print(image.size(), label.size()) 
#    
#        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).to('cuda')#.float()
#        label = label.reshape(1, label.shape[0], label.shape[1]).to('cuda')#.float()
#
#        gray = nid.get_grayscale(image)# / 255
##        gray_o = nid.get_grayscale(image_o)# / 255
##        print("----orig----")
##        print(image)
##        print("----gray----")
##        print(gray)
#        p_cl, p_c, p_l = nid.get_probabilities(gray, label)
##        p_table_c = nid.prior_probability_table(gray, bins=32, value_range=1, bw=0.005)
##        p_table_l = nid.prior_probability_table(gray_o, bins=32, value_range=1, bw=0.005)
##        p_cl = p_cl / p_cl.sum()
##        p_c = torch.sum(p_table_c, 1) / p_table_c.sum()
##        p_l = torch.sum(p_table_l, 1) / p_table_l.sum()
##        p_cl = torch.mm(p_table_c, torch.t(p_table_l))
##        p_cl /= p_cl.sum()
#
#        print(p_c)
#        print(p_l)
#
##        print("p_cl : {}, p_c : {}, p_l : {}".format(p_cl.sum(), p_c.sum(), p_l.sum()))
##        print(p_cl.size(), p_c.size(), p_l.size()) 
#        loss = nid.nid(p_cl, p_c.reshape(p_c.size()[0], 1), p_l.reshape(p_l.size()[0], 1))
#        print("loss : {}".format(loss.item()))

if __name__ == '__main__':
    main()
