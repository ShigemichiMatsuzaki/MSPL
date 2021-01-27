import torch
import argparse
import datetime
from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentation, GREENHOUSE_CLASS_LIST

from commons.general_details import segmentation_models, segmentation_datasets

# UI
from tkinter import *
from PIL import Image, ImageTk

from tkinter import *
from PIL import Image, ImageTk

from tkinter import *
from PIL import Image, ImageTk

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--ignore-idx', type=int, default=255, help='Index or label to be ignored during training')

parser.add_argument('--s', type=float, default=2.0, help='Factor by which channels will be scaled')
# model details
parser.add_argument('--freeze-bn', action='store_true', default=False, help='Freeze BN params or not')

# dataset and result directories
parser.add_argument('--dataset', type=str, default='greenhouse', choices=segmentation_datasets, help='Datasets')
parser.add_argument('--weights', type=str, default='/tmp/runs/uest/for-paper/trav/model_espdnetue_greenhouse/s_2.0_res_480_uest_rgb_os_camvid_cityscapes_forest_ue/20201204-152737/espdnetue_ep_1.pth', help='Name of weight file to load')
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
parser.add_argument('--trav-module-weights', default='/tmp/runs/uest/trav/model_espdnetue_greenhouse/20210115-145605/espdnetue_best.pth', 
                    type=str, help='Weight file of traversability module')

args = parser.parse_args()

class App(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.frame1 = Frame(self)
        self.frame2 = Frame(self)
        self.frame3 = Frame(self)
        self.original = Image.open('26_0_000000.png')

        self.image = ImageTk.PhotoImage(self.original)
        self.display = Canvas(self.frame1)

        # Button to display the previous image
        self.button1 = Button(self.frame2, text='Previous')
        self.button1.grid(row=0, column=0)
        self.button1.bind("<Button-1>", self.show_previous_image)

        # Slide bar to change the threshold
        self.xscale = Scale(self.frame2, from_=1, to=1000, orient=HORIZONTAL, command=self.resize)
        self.xscale.grid(row=0, column=1)

        # Button to display the next image
        self.button2 = Button(self.frame2, text='Next')
        self.button2.grid(row=0, column=2)
        self.button2.bind("<Button-1>", self.show_next_image)

        self.display.pack(fill=BOTH, expand=1)
#        self.xscale.pack()
        self.pack(fill=BOTH, expand=1)
        self.frame1.pack(fill=BOTH, expand=1)
        self.frame2.pack()
        self.bind("<Configure>", self.resize)

    def show_previous_image(self, *args):
        print("show previous image")

    def show_next_image(self, *args):
        print("show next image")

    def resize(self, *args):
        size = (self.xscale.get(), self.xscale.get())
        resized = self.original.resize(size,Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(resized)
        self.display.delete("IMG")
        self.display.create_image(self.display.winfo_width()/2, self.display.winfo_height()/2, anchor=CENTER, image=self.image, tags="IMG")

# Main
def main():
    # Load model weights
    args.classes = len(GREENHOUSE_CLASS_LIST)
    from model.segmentation.espdnet_ue_traversability import espdnetue_seg
    model = espdnetue_seg(args, load_entire_weights=True, fix_pyr_plane_proj=True, spatial=False)

    # Import a dataset
    val_dataset = GreenhouseRGBDSegmentation(root='./vision_datasets/greenhouse/', list_name=args.val_list, use_traversable=False, 
                                             train=False, size=crop_size, use_depth=args.use_depth,
                                             normalize=args.normalize)

    root = Tk()
    app = App(root)
    app.mainloop()

if __name__=='__main__':
    main()
