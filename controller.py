import cv2
import tkinter as tk
from model import Yolov7Model, TestModel
from stitch import StitchingClip
from tkinter import filedialog
from view import View

class Controller():
    def __init__(self):
        self.root = tk.Tk()
        self.view = View(self, self.root)
        self.model = Yolov7Model()
        #self.model = TestModel()
        self.root.mainloop()
        
    # when pressing the ANALYZE button
    def on_menu_open_clip(self): # evt is the event object
        self.path = filedialog.askopenfilename()
        if len(self.path) > 0:
            self.stitch_app = StitchingClip(self.path, return_img_flag=True)
            #self.stitch_app.extract_frames(rotate=cv2.ROTATE_90_CLOCKWISE)
            self.stitch_app.extract_frames(rotate=None)
            result_dict = self.stitch_app.run()
            img = result_dict['out_image']
            self.view.curr_img = img 
            self.view.update_canvas(img)
            sos_dict, ana_img = self.model.analyze_one_image(img)
            self.view.update_canvas(ana_img)
            self.view.update_result(sos_dict)
            
    def on_menu_open_image(self):
        self.path = filedialog.askopenfilename()
        if len(self.path) > 0:
            img = cv2.imread(self.path)
            self.view.curr_img = img 
            self.view.update_canvas(img)
            sos_dict, ana_img = self.model.analyze_one_image(img)
            self.view.update_canvas(ana_img)
            self.view.update_result(sos_dict)
            