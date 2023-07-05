import cv2, time
import tkinter as tk
from model import Yolov7Model, TestModel
from m_utils import update_datetime_to_img
from stitch import StitchingClip
from tkinter import filedialog
from time import strftime, gmtime
from view import View

class Controller():
    def __init__(self):
        self.root = tk.Tk()
        self.view = View(self, self.root)
        self.view.update_canvas(cv2.imread('data/SPVB web-banner545a.png'))
        self.model = Yolov7Model()
        #self.model = TestModel()
        self.root.mainloop()
        
    def on_menu_open_image(self):
        self.path = filedialog.askopenfilename()
        if len(self.path) > 0:
            img = cv2.imread(self.path)
            self.view.curr_img = img 
            #self.view.update_canvas(img)
            sos_dict, ana_img = self.model.analyze_one_image(img)
            ana_img = update_datetime_to_img(ana_img)
            self.view.update_canvas(ana_img)
            self.view.update_result(sos_dict)
            self.view.update_progress_bar(100)
            
    def on_menu_open_clip(self): # evt is the event object
        self.path = filedialog.askopenfilename()
        if len(self.path) > 0:
            self.stitch_app = StitchingClip(self.path, rewind=2, slope_thr=0.8, stride=30, \
                                            return_img_flag=True, progress_interface=self.view)
                
            time_start = time.time()
            #self.stitch_app.extract_frames(rotate=cv2.ROTATE_90_CLOCKWISE)
            self.stitch_app.extract_frames(rotate=None)
            result_dict = self.stitch_app.run()
            img = result_dict['out_image']
            self.view.curr_img = img 
            sos_dict, ana_img = self.model.analyze_one_image(img)
            time_stop = time.time()
            elapse = time_stop-time_start
            elapse = strftime("%M'%S''", gmtime(elapse))
            ana_img = update_datetime_to_img(ana_img, msg=f' ({elapse})')
            self.view.update_canvas(ana_img)
            self.view.update_result(sos_dict)
            self.view.update_progress_bar(100)
            

            
    
            