import cv2, copy
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Button, Canvas, END, Entry, Label, LabelFrame, Menu, PhotoImage, ttk
from PIL import Image, ImageTk


class View():
    def __init__(self, controller, root):
        self.root = root
        self.root.iconphoto(False, PhotoImage(file='data/dms_logo.png'))
        self.root.title("DMSPro demo software")
        self.root.resizable(False,False)
        self.controller = controller
        self.curr_img = None
        self.img_shape = (1400, 400)
        self.pie_chart_shape = (250, 200)
        self.table_shape = (self.img_shape[0]-self.pie_chart_shape[0], 200)
        
        
        self.create_menu()
        self.create_drawing_canvas()
        self.create_pie_chart()
        self.create_progress_bar()
        self.create_tables()

    def create_menu(self):
        self.menu_bar = Menu(self.root)
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Choose image...", command=self.controller.on_menu_open_image)
        self.file_menu.add_command(label="Choose clip...", command=self.controller.on_menu_open_clip)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.destroy)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.root.config(menu=self.menu_bar)
        
    # def create_status_bar(self):
    #     self.top_bar = LabelFrame(self.root)
    #     self.top_bar.grid(row = 0, column = 0, columnspan = 2, sticky = 'W')
    #     Label(self.top_bar, text = 'STATUS: ', font = ('Courier', 15, 'bold')).grid(row=0, column=0, sticky='W')
    #     self.status = Label(self.top_bar, text = '', font = ('Courier', 15))
    #     self.status.grid(row=0, column=1, sticky='W')
        
    #     for child in self.top_bar.winfo_children():
    #         child.grid_configure(padx = 5, pady = 5, ipadx = 5, ipady = 5)

    def create_drawing_canvas(self):        
        self.img_canvas = Canvas(self.root, width = self.img_shape[0], height = self.img_shape[1])
        self.img_canvas.grid(row=0, column=0, columnspan=2)

    def create_pie_chart(self):
        self.pie_chart = Canvas(self.root, width = self.pie_chart_shape[0], height = self.pie_chart_shape[1])
        self.pie_chart.grid(row=1, column=0, rowspan=3)
    
    def create_progress_bar(self):
        self.pb = ttk.Progressbar(self.root, orient='horizontal', mode='determinate', length=self.table_shape[0])
        self.pb.grid(row=1, column=1)
        self.pb.grid_configure(padx=20)
    
    def create_tables(self):
        self.table = LabelFrame(self.root, width = self.table_shape[0], height=self.table_shape[1], borderwidth=0, highlightthickness=0)
        self.table.grid(row=2, column=1)
        self.table_2 = LabelFrame(self.root, width = self.table_shape[0], height=self.table_shape[1], borderwidth=0, highlightthickness=0)
        self.table_2.grid(row=3, column=1)
    
    # resize and convert from cv2 image to image suitable for tk
    def convert_image_to_display(self, img, size=None):
        if size is None:
            img = cv2.resize(img, tuple(self.img_shape))
        else:
            img = cv2.resize(img, size)
        conv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(conv_img)
        tk_img = ImageTk.PhotoImage(pil_img)
        return tk_img
    
    def update_canvas(self, img):
        global tk_img
        tk_img = self.convert_image_to_display(img)
        self.img_canvas.create_image(0, 0, anchor = 'nw', image = tk_img)
        self.img_canvas.update()
    
    def update_progress_bar(self, val):
        global pb_value
        pb_value = val
        self.pb['value'] = pb_value
        self.root.update()
    
    def update_result(self, sos_dict):
        """ Update pie chart """
        path_pie_chart = 'data/output/pie_chart.png'
        labels = list(sos_dict['percent'].keys())
        drink_types = list(sos_dict['percent']['SPVB'].keys())
        colors = ['blue', 'red']
        sizes = [sos_dict['percent_skus']['SPVB'], sos_dict['percent_skus']['NON_SPVB']]
        plt.clf()
        plt.pie(sizes, labels = labels, colors = colors, startangle=90, shadow = True, explode = (0.1, 0.1), autopct = '%1.0f%%')
        #plt.title('Beverage distribution')
        plt.axis('equal')
        plt.savefig(path_pie_chart)
        global pie_chart_img
        pie_chart_img = cv2.imread(path_pie_chart)
        pie_chart_img = self.convert_image_to_display(pie_chart_img, size=self.pie_chart_shape)
        self.pie_chart.create_image(0, 0, anchor = 'nw', image = pie_chart_img)
        self.pie_chart.update()
        
        """ Update table """
        # Initialize tables
        width_cell = 15
        font_size = 10
        e = Entry(self.table, width=width_cell, fg='black', font=('Arial', font_size,'bold') )
        e.grid(row=0, column=0)
        e.insert(END, 'Theo SOS')
        for i, row in enumerate(labels):
            color = 'blue' if row == 'SPVB' else 'red'
            e = Entry(self.table, width=width_cell, fg=color, font=('Arial', font_size,'bold') )
            e.grid(row=i+1, column=0)
            e.insert(END, row)
        for j, col in enumerate(drink_types):
            e = Entry(self.table, width=width_cell, fg='black', font=('Arial', font_size,'bold') )
            e.grid(row=0, column=j+1)
            e.insert(END, col)
        # Update table
        for i, row in enumerate(labels):
            for j, col in enumerate(drink_types):
                e = Entry(self.table, width=width_cell, fg='black', font=('Arial', font_size,'bold') )
                e.grid(row=i+1, column=j+1)
                output_stats = f"{sos_dict['percent'][row][col]}%"
                e.insert(END, output_stats)
                
        # Table 2
        e = Entry(self.table_2, width=width_cell, fg='black', font=('Arial', font_size,'bold') )
        e.grid(row=0, column=0)
        e.insert(END, 'Theo SKU')
        for i, row in enumerate(labels):
            color = 'blue' if row == 'SPVB' else 'red'
            e = Entry(self.table_2, width=width_cell, fg=color, font=('Arial', font_size,'bold') )
            e.grid(row=i+1, column=0)
            e.insert(END, row)
        for j, col in enumerate(drink_types):
            e = Entry(self.table_2, width=width_cell, fg='black', font=('Arial', font_size,'bold') )
            e.grid(row=0, column=j+1)
            e.insert(END, col)
            
        # Update table 2
        count_dict = {}
        for j, col in enumerate(drink_types):
            count_dict[col] = sos_dict['total_num_boxes']['SPVB'][col] + sos_dict['total_num_boxes']['NON_SPVB'][col]
            
        for i, row in enumerate(labels):
            for j, col in enumerate(drink_types):
                e = Entry(self.table_2, width=width_cell, fg='black', font=('Arial', font_size,'bold') )
                e.grid(row=i+1, column=j+1)
                if count_dict[col] > 0:
                    percent = round(sos_dict['total_num_boxes'][row][col] / count_dict[col] * 100, 1)
                    output_stats = f"{sos_dict['total_num_boxes'][row][col]} ({percent}%)"
                else:
                    output_stats = 'NA'
                e.insert(END, output_stats)