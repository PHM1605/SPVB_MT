import cv2, copy
import tkinter as tk
from tkinter import Button, Canvas, Label, LabelFrame
from PIL import Image, ImageTk

class View():
    def __init__(self, controller, root):
        self.root = root
        self.controller = controller
        self.curr_img = None
        self.img_shape = [1280, 720]
        self.create_status_bar()
        self.create_left_bar()
        self.create_drawing_canvas()
        

    def create_status_bar(self):
        self.top_bar = LabelFrame(self.root)
        self.top_bar.grid(row = 0, column = 0, columnspan = 2, sticky = 'W')
        Label(self.top_bar, text = 'STATUS: ', font = ('Courier', 15, 'bold')).grid(row=0, column=0, sticky='W')
        self.status = Label(self.top_bar, text = '', font = ('Courier', 15))
        self.status.grid(row=0, column=1, sticky='W')
        
        for child in self.top_bar.winfo_children():
            child.grid_configure(padx = 5, pady = 5, ipadx = 5, ipady = 5)

    def create_left_bar(self):
        self.left_bar = LabelFrame(self.root)
        self.left_bar.grid(row=1, column=0)
        # Analyze button
        self.analyze_btn = Button(self.left_bar, text='Analyze', width=16, height=4, font=('Courier', 16, 'bold'))
        self.analyze_btn.bind('<Button-1>', self.controller.on_btn_analyze)
        self.analyze_btn.grid(row=0, column=0)
        # Display result
        self.stat_frame = LabelFrame(self.left_bar, width=16, height = 6)
        self.stat_frame.grid(row=1, column=0)
        fontSize = 10
        Label(self.stat_frame, text = 'SPVB: ', font = ('Courier', fontSize)).grid(row = 0, column = 0)
        Label(self.stat_frame, text = 'NonSPVB: ', font = ('Courier', fontSize)).grid(row = 1, column = 0)
        Label(self.stat_frame, text = 'SOS (%): ', font = ('Courier', fontSize)).grid(row = 2, column = 0)
        self.num_spvb = Label(self.stat_frame, text = '', font = ('Courier', fontSize))
        self.num_spvb.grid(row = 0, column = 1)
        self.num_others = Label(self.stat_frame, text = '', font = ('Courier', fontSize))
        self.num_others.grid(row = 1, column = 1)
        self.sos = Label(self.stat_frame, text = '', font = ('Courier', fontSize))
        self.sos.grid(row = 2, column = 1)
        
        for child in self.left_bar.winfo_children():
            child.grid_configure(padx = 5, pady = 5, ipadx = 5, ipady = 5)
        
    def create_drawing_canvas(self):        
        self.img_canvas = Canvas(self.root, width = self.img_shape[0], height = self.img_shape[1])
        self.img_canvas.grid(row = 1, column = 1, sticky = 'NSWE')

    # resize and convert from cv2 image to image suitable for tk
    def convert_image_to_display(self, img):
        img = cv2.resize(img, tuple(self.img_shape))
        conv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(conv_img)
        tk_img = ImageTk.PhotoImage(pil_img)
        return tk_img
    
    def update_canvas(self, img):
        global tk_img
        tk_img = self.convert_image_to_display(img)
        self.img_canvas.create_image(0, 0, anchor = 'nw', image = tk_img)
        self.img_canvas.update()
    
    def update_result(self, sos_dict):
        self.num_spvb['text'] = str(sos_dict['SPVB'])
        self.num_others['text'] = str(sos_dict['NonSPVB'])
        self.sos['text'] = str(sos_dict['sos'])
