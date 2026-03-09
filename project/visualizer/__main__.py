import sys
import os
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project.visualizer.controller import AppController


def main() -> None:
    root = tk.Tk()
    AppController(root)
    root.mainloop()


if __name__=="__main__":
    main()
