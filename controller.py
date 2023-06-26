import tkinter as tk
from view import View

class Controller():
    def __init__(self):
        self.root = tk.Tk()
        self.view = View(self, self.root)
        self.root.mainloop()