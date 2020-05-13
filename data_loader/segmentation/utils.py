from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage.data
import skimage.color
import skimage.filters
import skimage.util
import skimage.segmentation

'''
input : sp -> Pixel label values in a superpixel
output : max_label -> A label value that should be assigned to the superpixel
'''
def propagate_max_label_in_sp(sp):
    label_hist = np.bincount(sp, minlength=5)

    valid_label_num = label_hist[0:4].sum()
    argmax = np.argmax(label_hist[0:4])
#    print(valid_label_num[valid_label_num==argmax].sum() / float(sp.size))
#    print("portion : {}".format(label_hist[argmax] / float(sp.size)))
    if valid_label_num and label_hist[argmax] / float(sp.size) > 0.3:
        return argmax
    else:
        return 4

def get_label_from_superpixel(rgb_img_np, label_img_np, sp_type='watershed'):
    rgb_img_np = skimage.util.img_as_float(rgb_img_np)
#    print(rgb_img_np.shape) 

    # Superpixel segmentation
    if sp_type == 'watershed':
        superpixels = skimage.segmentation.watershed(
            skimage.filters.sobel(skimage.color.rgb2gray(rgb_img_np)), markers=250, compactness=0.001) 
    elif sp_type == 'quickshift':
        superpixels = skimage.segmentation.quickshift(rgb_img_np, kernel_size=3, max_dist=6, ratio=0.5) 
    elif sp_type == 'felzenszwalb':
        superpixels = skimage.segmentation.felzenszwalb(rgb_img_np,scale=100, sigma=0.5, min_size=50)
    elif sp_type == 'slic':
        superpixels = skimage.segmentation.slic(rgb_img_np, n_segments=250, compactness=10, sigma=1)

    # Define a variable for a new label image
    new_label_img = np.zeros(label_img_np.shape) 
    for i in range(0, superpixels.max()):
        # Get indeces of pixels in i+1 th superpixel
        index = superpixels == (i+1)
    
        # Get labels within the superpixel
        labels_in_sp = label_img_np[index]
    
        # Get a label id that should be propagated within the superpixel
        max_label = propagate_max_label_in_sp(labels_in_sp)
    
        # Substitute the label in all pixels in the superpixel
        if max_label != 4:
            new_label_img[index] = max_label
        else:
            new_label_img[index] = labels_in_sp
            
    return new_label_img

def main():
    filename = "26_0_000000.png"
    img = skimage.util.img_as_float( plt.imread('/media/data/dataset/matsuzaki/greenhouse/train/' + filename) )
    print(img.shape)
    superpixels = skimage.segmentation.watershed( skimage.filters.sobel( skimage.color.rgb2gray( img ) ), markers=250, compactness=0.001) 
    #superpixels = skimage.segmentation.quickshift(img, kernel_size=3, max_dist=6, ratio=0.5) 
    #superpixels = skimage.segmentation.felzenszwalb(img,scale=100, sigma=0.5, min_size=50)
    #superpixels = skimage.segmentation.slic(img, n_segments=250, compactness=10, sigma=1)
    
    # label_img = skimage.util.img_as_float( plt.imread('trainannot/' + filename) )
    label_img = np.array(Image.open('/media/data/dataset/matsuzaki/greenhouse/trainannot/' + filename))
    
    new_label_img = np.zeros(label_img.shape) 
    for i in range(0, superpixels.max()):
        index = superpixels == (i+1)
    
        labels_in_sp = label_img[index]
    
        max_label = propagate_max_label_in_sp(labels_in_sp)
    
    #    print("==========")
    #    print(labels_in_sp)
    #    print(max_label)
    
        new_label_img[index] = max_label
    
    palette = [0, 255, 0,
               0, 255, 255,
               255, 0, 0,
               255, 255, 0,
               0, 0, 0]
    
#    print(new_label_img)
    index_image = Image.fromarray(np.uint8(new_label_img)).convert("P")
    index_image.putpalette(palette)
    
    plt.imsave('sp_img.png', skimage.segmentation.mark_boundaries(img, superpixels))
    index_image.save('hoge.png')

if __name__=='__main__':
    main()
