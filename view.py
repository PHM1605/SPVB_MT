import cv2, copy
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk

class View():
    def __init__(self, controller, root):
        self.controller = controller
        self.root = root
        self.frame = tk.Frame(root)
        self.frame.pack()
        self.curr_img = None

        self.create_top_bar(root)
        self.create_left_bar(root)
        self.create_drawing_canvas(root)
            

    def create_left_bar(self, root):
        self.left_bar = tk.Frame(root, width = 150, relief = 'raised', bg = '#65A8E1')
        self.left_bar.pack(fill = 'y', side = 'left')
        self.structure_left_bar()
        
    def structure_left_bar(self):        
        # CAPTURE button
        self.button_capture = Button(self.left_bar, text = 'CAPTURE', height = 6, width = 21)
        self.button_capture.bind('<Button-1>', self.controller.on_btn_capture)
        self.button_capture.pack(side = 'top', padx = 2, pady = 6)        

        # RESULT label
        self.result_lab = Label(self.left_bar, borderwidth=2, relief="groove", text = 'PASS/ FAIL', height = 6, width = 21)
        self.result_lab.pack(side="top", padx = 2, pady = 6)          
    
    def create_top_bar(self, root):
        self.top_bar = tk.Frame(root, height=25, relief="raised", bg = '#65A8E1')
        self.top_bar.pack( fill="x", side="top")
        self.structure_top_bar()

    def structure_top_bar(self):
        self.barcode_lb = Label(self.top_bar, text = 'Status', height = 1, width = 8)
        self.barcode_lb.pack(side="left",padx = 2, pady = 1)
        
        # BARCODE label
        self.barcode_text = Label(self.top_bar, borderwidth=2, relief="groove", text = '', height = 1, width = 30)
        self.barcode_text.pack(side="left", padx = 2, pady = 1) 
        
    def create_drawing_canvas(self, root):
        self.canvas = tk.Frame(root)
        self.canvas.pack(side = 'right', expand = 'yes', fill = 'both')
        
        self.img_canvas = tk.Canvas(self.canvas, background = 'Lavender', width = 864, height = 648)
        self.img_canvas.pack(side = 'right', expand = 'yes', fill = 'both')
    
    def update_canvas(self, img):
        global tk_img
        tk_img = self.convert_image_to_display(img)
        self.img_canvas.create_image(0, 0, anchor = 'nw', image = tk_img)
        self.img_canvas.update()

    # resize and convert from cv2 image to image suitable for tk
    def convert_image_to_display(self, img):
        img = cv2.resize(img, (864, 648))
        conv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(conv_img)
        tk_img = ImageTk.PhotoImage(pil_img)
        return tk_img
    
    def draw_result(self):
        img1 = copy.deepcopy(self.curr_img)
        dh, dw = self.controller.img_shape
        
        radiatorList = [comp for comp in self.controller.comps if comp.name == 'Radiator']
        barcodeList = [comp for comp in self.controller.comps if comp.name == 'BarCode']
        for iRadiator, radiator in enumerate(radiatorList):
            if self.controller.preds[iRadiator] == radiator.label:
                color = (0, 255, 0) # green
            else:
                color = (0, 0, 255) # red
            x, y, w, h = radiator.box[0], radiator.box[1], radiator.box[2], radiator.box[3]   
            image = cv2.rectangle(img1, (x, y), (x + w, y + h), color, 5) 
            cv2.putText(img1,'#' + str(radiator.label) , (x, y-5) , cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
        
        for iBarcode, barcode in enumerate(barcodeList):
            color = (255, 0, 0)
            x, y, w, h = barcode.box[0], barcode.box[1], barcode.box[2], barcode.box[3]
            image = cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 5)
        
        self.update_canvas(image)
        return image
    
    def update_result(self, ok_flag):
        if ok_flag:
            self.result_lab['text'] = 'PASS'
            self.result_lab['bg'] = 'green'
        else:
            self.result_lab['text'] = 'FAIL'
            self.result_lab['bg'] = 'red'     