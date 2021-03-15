import torch
from torchvision import transforms
import argparse
import datetime

from data_loader.segmentation.greenhouse import GreenhouseRGBDSegmentationTrav, GREENHOUSE_CLASS_LIST
from commons.general_details import segmentation_models, segmentation_datasets

# UI
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
parser.add_argument('--weights', type=str, default='./espdnetue_ssm.pth', help='Name of weight file to load')
parser.add_argument('--data-test-list', type=str, default='./vision_datasets/traversability_mask/greenhouse_a_test.lst',
                    help='Location to save the results')
parser.add_argument('--trav-module-weights', default='./espdnetue_tem.pth', 
                    type=str, help='Weight file of traversability module')
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

args = parser.parse_args()

class App(Frame):
    def __init__(self, master, model, dataset):
        Frame.__init__(self, master)
        self.master.geometry("1500x600")

        self.model = model.to('cuda')
        self.dataset = dataset
        self.index = 0

        # Define the window
        self.frame1 = Frame(self)
        self.frame2 = Frame(self)
        self.frame3 = Frame(self)

        self.display = Canvas(self.frame1)

        # Button to display the previous image
        self.button1 = Button(self.frame2, text='Previous')
        self.button1.grid(row=0, column=0)
        self.button1.bind("<Button-1>", self.show_previous_image)

        # Slide bar to change the threshold
        self.xscale = Scale(self.frame2, from_=0, to=100, orient=HORIZONTAL, command=self.resize)
        self.xscale.set(50)
        self.xscale.grid(row=0, column=1)

        # Button to display the next image
        self.button2 = Button(self.frame2, text='Next')
        self.button2.grid(row=0, column=2)
        self.button2.bind("<Button-1>", self.show_next_image)

        # Button to display the next image
        self.button3 = Button(self.frame2, text='Save image')
        self.button3.grid(row=0, column=3)
        self.button3.bind("<Button-1>", self.save_image)

        self.update_current_images()

        self.display.pack(fill=BOTH, expand=1)
#        self.xscale.pack()
        self.pack(fill=BOTH, expand=1)
        self.frame1.pack(fill=BOTH, expand=1)
        self.frame2.pack()
        self.bind("<Configure>", self.resize)

    def get_image(self, index=0):
        image_tensor = self.dataset[index]['rgb']
        image_pil = transforms.ToPILImage()(image_tensor)

        return {'pil': image_pil, 'tensor': image_tensor}
    
    def get_output(self, image_tensor):
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        output_tensor = self.model(image_tensor.to('cuda'))[2]
        output_tensor = torch.sigmoid(output_tensor) / 0.3
        output_tensor = torch.squeeze(output_tensor).cpu()

#        output_tensor[output_tensor > 1.0] = 1.0
        if output_tensor.max() > 1:
            output_tensor /= output_tensor.max()

        output_pil = transforms.ToPILImage()(output_tensor)

        self.current_output_tensor = output_tensor

        return {'pil': output_pil, 'tensor': output_tensor}

    def update_current_images(self, index=0):
        # Get image tensor
        images = self.get_image(self.index)
        self.current_image_pil = images['pil']
        self.current_image_tensor = images['tensor']

        outputs = self.get_output(self.current_image_tensor)
        self.current_prob_tensor = outputs['tensor']
        self.current_prob_pil = outputs['pil']
        self.current_binary_pil = self.get_binary_image(self.current_prob_tensor, self.xscale.get() / 100)

        self.tk_image = ImageTk.PhotoImage(self.current_image_pil)
        self.tk_prob = ImageTk.PhotoImage(self.current_prob_pil)
        self.tk_binary = ImageTk.PhotoImage(self.current_binary_pil)

        image_width = self.current_image_pil.size[0]
        self.display.create_image(0, 0, anchor='nw', image=self.tk_image, tags="IMG1")
        self.display.create_image(image_width+10, 0, anchor='nw', image=self.tk_prob, tags="IMG2")
        self.display.create_image(image_width * 2 + 20, 0, anchor='nw', image=self.tk_binary, tags="IMG3")

    def show_previous_image(self, *args):
        self.index = (self.index - 1) % len(self.dataset)

        self.update_current_images(index=self.index)

    def show_next_image(self, *args):
        self.index = (self.index + 1) % len(self.dataset)

        self.update_current_images(index=self.index)

    def get_binary_image(self, prob_tensor, threshold=0.5):
        binary = torch.zeros_like(prob_tensor)
        binary[prob_tensor > threshold] = 1.0
        binary_pil = transforms.ToPILImage()(binary)

        return binary_pil

    def resize(self, *args):
        binary_pil = self.get_binary_image(self.current_prob_tensor, self.xscale.get() / 100)
#        size = (self.xscale.get(), self.xscale.get())
#        resized = self.original.resize(size,Image.ANTIALIAS)
        self.tk_binary = ImageTk.PhotoImage(binary_pil)
        self.display.delete("IMG3")

        image_width = binary_pil.size[0]
        self.display.create_image(image_width * 2 + 20, 0, anchor='nw', image=self.tk_binary, tags="IMG3")
    
    def save_image(self, *args):
        self.current_prob_pil.save('prob_img.png')

# Main
def main():
    # Load model weights
    args.classes = len(GREENHOUSE_CLASS_LIST)
    from model.segmentation.espdnet_ue_traversability import espdnetue_seg
    model = espdnetue_seg(args, load_entire_weights=True, fix_pyr_plane_proj=True, spatial=False)
    model.eval()

    # Import a dataset
    trav_test_set = GreenhouseRGBDSegmentationTrav(list_name=args.data_test_list, use_depth=args.use_depth)

    root = Tk()
    app = App(root, model, trav_test_set)
    app.mainloop()

if __name__=='__main__':
    main()
