# Entry point — wires the MVC triad together and starts the Tkinter event loop.
# Controller loads data sources and manages all background threads;
# View owns the window and all widgets; root.mainloop() blocks until the user closes.
from tkinter import Tk
from controller.controller import Controller
from ui.view import View

def main():
    root = Tk()
    controller = Controller()
    view = View(root, controller)
    # bring window to front on launch (Tkinter doesn't always do this on Windows)
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.mainloop()

if __name__ == "__main__":
    main()
