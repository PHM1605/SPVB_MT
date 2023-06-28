# import the necessary packages
from tkinter import Button, Label, Tk
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
from stitch import StitchingClip

def select_image():
    # grab a reference to the image panels
    global panelA
    # open a file chooser dialog and allow the user to select an input image
    path = filedialog.askopenfilename()
    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk
        image = cv2.imread(path)        
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert the images to PIL format...
        image = Image.fromarray(image)
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        # if the panels are None, initialize them
        if panelA is None:
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
        # otherwise, update the image panels
        else:
            panelA.configure(image=image)
            panelA.image = image

class DemoSPVBApp():
    def __init__(self):
        self.root = Tk()
        self.init_view()
        self.root.mainloop()
    
    def select_vid(self):
        self.path = filedialog.askopenfilename()
        self.run()
            
    def init_view(self):
        self.btn = Button(self.root, text="Select an image", command=self.select_vid)
        self.btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
        self.panel_img = Label()
        self.panel_img.pack(side="left", padx=10, pady=10)
    
    def display_img(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1280, 720))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.panel_img.configure(image=image)
        self.panel_img.image = image
    
    def run(self):
        if len(self.path) > 0:
            self.stitch_app = StitchingClip(self.path)
            #self.stitch_app.extract_frames(rotate=cv2.ROTATE_90_CLOCKWISE)
            self.stitch_app.extract_frames(rotate=None)
            self.stitched_img = self.stitch_app.run()
            self.display_img(self.stitched_img)
            
if __name__ == '__main__':
    app = DemoSPVBApp()

