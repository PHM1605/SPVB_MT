import cv2
import tkinter as tk
from model import Yolov8Model, TestModel
from stitch import StitchingClip
from tkinter import filedialog
from view import View

class Controller():
    def __init__(self):
        self.root = tk.Tk()
        self.view = View(self, self.root)
        # self.model = Yolov8Model()
        self.model = TestModel()
        self.root.mainloop()
        
    # when pressing the ANALYZE button
    def on_btn_analyze(self, evt): # evt is the event object
        self.path = filedialog.askopenfilename()
        if len(self.path) > 0:
            self.stitch_app = StitchingClip(self.path)
            #self.stitch_app.extract_frames(rotate=cv2.ROTATE_90_CLOCKWISE)
            self.stitch_app.extract_frames(rotate=None)
            result_dict = self.stitch_app.run(return_img_flag=True)
            self.stitched_img = result_dict['out_image']
            msg = result_dict['message']
            self.view.status['text'] = msg
            self.view.curr_img = self.stitched_img
            
            # Update yolo and sos result
            sos_dict, res_img = self.model.analyze_one_image(self.stitched_img)
            self.view.update_canvas(res_img)
            self.view.update_result(sos_dict)
            
    def on_menu_open(self):
        self.path = filedialog.askopenfilename()
        if len(self.path) > 0:
            img = cv2.imread(self.path)
        self.view.curr_img = img 
        self.view.update_canvas(img)
        sos_dict, ana_img = self.model.analyze_one_image(img)
        self.view.update_canvas(ana_img)
        self.view.update_result(sos_dict)
            