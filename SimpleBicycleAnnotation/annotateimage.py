import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Menu
import cv2
from PIL import Image, ImageTk

class VideoAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Annotator")
        
        # Menu bar
        menubar = Menu(root)
        
        # File menu
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Video", command=self.load_video, accelerator="Ctrl+L")
        filemenu.add_command(label="Save Annotations", command=self.save_annotations, accelerator="Ctrl+S")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # Edit menu
        editmenu = Menu(menubar, tearoff=0)
        editmenu.add_command(label="Clear Annotations", command=self.clear_annotations, accelerator="Ctrl+C")
        editmenu.add_command(label="Clear Specific ID", command=self.clear_specific_id, accelerator="Ctrl+D")
        menubar.add_cascade(label="Edit", menu=editmenu)

        # Help menu
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="Controls", command=self.show_controls)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        root.config(menu=menubar)
        
        # Keyboard shortcuts
        root.bind("<Control-l>", lambda event: self.load_video())
        root.bind("<Control-s>", lambda event: self.save_annotations())
        root.bind("<Control-c>", lambda event: self.clear_annotations())
        root.bind("<Control-d>", lambda event: self.clear_specific_id())
        root.bind("<Left>", lambda event: self.prev_frame())
        root.bind("<Right>", lambda event: self.next_frame())
        
        self.canvas = tk.Canvas(root)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.canvas.bind("<Button-1>", self.annotate_position)
        
        self.prev_button = tk.Button(root, text="Previous Frame", command=self.prev_frame)
        self.prev_button.pack(side=tk.LEFT)
        
        self.next_button = tk.Button(root, text="Next Frame", command=self.next_frame)
        self.next_button.pack(side=tk.RIGHT)
        
        self.label = tk.Label(root, text="")
        self.label.pack(side=tk.TOP)
        
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.annotations = {}
        self.current_time = 0

    def load_video(self):
        self.video_path = filedialog.askopenfilename()
        if not self.video_path:
            return
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.show_frame()

    def show_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.current_time * 1000)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.current_frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        if self.current_time in self.annotations:
            for coord, bid, wheel_type in self.annotations[self.current_time]:
                x, y = coord
                color = 'blue' if wheel_type == 'front' else 'red'
                self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=color)
                self.canvas.create_text(x, y-10, text=str(bid), fill='white')
        
        # Update the label with frame number and time
        self.label.config(text=f"Frame: {self.current_time}, Time: {self.current_time} seconds")

    def annotate_position(self, event):
        x, y = event.x, event.y
        bid = simpledialog.askinteger("Input", "Enter Bicycle ID:")
        if bid is None:
            return
        wheel_type = simpledialog.askstring("Input", "Enter Wheel Type (f for front/r for rear):")
        if wheel_type == 'f':
            wheel_type = 'front'
        elif wheel_type == 'r':
            wheel_type = 'rear'
        if wheel_type not in ['front', 'rear']:
            messagebox.showerror("Error", "Invalid wheel type. Please enter 'f' for front or 'r' for rear.")
            return
        if self.current_time not in self.annotations:
            self.annotations[self.current_time] = []
        self.annotations[self.current_time].append(((x, y), bid, wheel_type))
        color = 'blue' if wheel_type == 'front' else 'red'
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=color)
        self.canvas.create_text(x, y-10, text=str(bid), fill='white')

    def prev_frame(self):
        if self.current_time > 0:
            self.current_time -= 1
            self.show_frame()

    def next_frame(self):
        self.current_time += 1
        self.show_frame()

    def clear_annotations(self):
        if self.current_time in self.annotations:
            del self.annotations[self.current_time]
        self.show_frame()

    def clear_specific_id(self):
        bid_to_remove = simpledialog.askinteger("Input", "Enter Bicycle ID to remove:")
        if bid_to_remove is None:
            return
        if self.current_time in self.annotations:
            self.annotations[self.current_time] = [(coord, bid, wheel_type) for coord, bid, wheel_type in self.annotations[self.current_time] if bid != bid_to_remove]
        self.show_frame()

    def save_annotations(self):
        with open("annotations.txt", "w") as f:
            for time, coords in self.annotations.items():
                f.write(f"Time: {time} seconds\n")
                for coord, bid, wheel_type in coords:
                    f.write(f"ID: {bid}, Position: {coord}, Wheel: {wheel_type}\n")
                f.write("\n")
        messagebox.showinfo("Info", "Annotations saved successfully!")

    def show_controls(self):

        about_window = tk.Toplevel(self.root)
        about_window.title("Controls")

        info = (
            """
        Controls:
        - Load Video: Ctrl+L
        - Save Annotations: Ctrl+S
        - Clear Annotations: Ctrl+C
        - Clear Specific ID: Ctrl+D
        - Navigate to Previous Frame: Left Arrow
        - Navigate to Next Frame: Right Arrow
        """
        )

        label = tk.Label(about_window, text=info, font=("Arial", 8))
        label.pack(pady=15)


    def show_about(self):
        about_window = tk.Toplevel(self.root)
        about_window.title("About")

        info = (
            "Developed by Kevin Gildea, Ph.D.\n"
            "Faculty of Engineering, LTH\n"
            "Lund University\n"
            "Email: kevin.gildea@tft.lth.se"
        )

        label = tk.Label(about_window, text=info, font=("Arial", 8))
        label.pack(pady=15)

root = tk.Tk()
app = VideoAnnotator(root)
root.mainloop()
