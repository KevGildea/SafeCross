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
        editmenu.add_command(label="Clear Bicycle Annotations", command=self.clear_annotations, accelerator="Ctrl+C")
        editmenu.add_command(label="Clear Specific Bicycle ID", command=self.clear_specific_id, accelerator="Ctrl+D")
        editmenu.add_command(label="Clear Tram Tracks", command=self.clear_tram_tracks, accelerator="Ctrl+T")
        editmenu.add_command(label="Clear Specific Tram ID", command=self.clear_specific_tram_id, accelerator="Ctrl+R")
        menubar.add_cascade(label="Edit", menu=editmenu)

        # Mode menu
        modemenu = Menu(menubar, tearoff=0)
        modemenu.add_command(label="Bicycle Mode", command=self.set_bicycle_mode)
        modemenu.add_command(label="Tram Track Mode", command=self.set_tram_track_mode)
        menubar.add_cascade(label="Mode", menu=modemenu)

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
        root.bind("<Control-t>", lambda event: self.clear_tram_tracks())
        root.bind("<Control-r>", lambda event: self.clear_specific_tram_id())

        
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
        self.mode = "Bicycle"  # Default mode

        self.tram_tracks = {}  # Dictionary to store tram tracks with frame number as key and list of coordinates as value
        self.tram_id = 1  # Starting ID for tram tracks

    def set_bicycle_mode(self):
        self.mode = "Bicycle"
        self.show_frame()

    def set_tram_track_mode(self):
        self.mode = "Tram Track"
        self.show_frame()

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
            
        # Display tram track annotations
        if self.current_time in self.tram_tracks:
            for tid, x, y in self.tram_tracks[self.current_time]:
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill='green')
                self.canvas.create_text(x, y-10, text=str(tid), fill='white')
            
        # Update the label with frame number and time
        self.label.config(text=f"Frame: {self.current_time}, Time: {self.current_time} seconds, Mode: {self.mode}")


    def annotate_position(self, event):
        x, y = event.x, event.y
        if self.mode == "Bicycle":
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
        elif self.mode == "Tram Track":
            if self.current_time not in self.tram_tracks:
                self.tram_tracks[self.current_time] = []
            self.tram_tracks[self.current_time].append((self.tram_id, x, y))
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill='green')
            self.canvas.create_text(x, y-10, text=str(self.tram_id), fill='white')
            
            # Ask the user if they want to annotate a new track or continue with the current one
            choice = messagebox.askquestion("Annotate Track", "Do you want to annotate a new track?\n\nYes: Annotate a new track\nNo: Continue annotating the current track")
            if choice == 'yes':
                self.tram_id += 1  # Increment tram ID for next track

    def prev_frame(self):
        if self.current_time > 0:
            self.current_time -= 1
            self.tram_points = []  # Clear tram points when navigating frames
            self.show_frame()

    def next_frame(self):
        self.current_time += 1
        self.tram_points = []  # Clear tram points when navigating frames
        self.show_frame()

    def clear_annotations(self):
        if self.current_time in self.annotations:
            del self.annotations[self.current_time]
        self.tram_points = []  # Clear tram points
        self.show_frame()

    def clear_specific_id(self):
        bid = simpledialog.askinteger("Input", "Enter Bicycle ID to clear:")
        if bid is None:
            return
        if self.current_time in self.annotations:
            self.annotations[self.current_time] = [a for a in self.annotations[self.current_time] if a[1] != bid]
            self.show_frame()

    def clear_tram_tracks(self):
        """Clear all tram track annotations for the current frame."""
        if self.current_time in self.tram_tracks:
            del self.tram_tracks[self.current_time]
        self.show_frame()

    def clear_specific_tram_id(self):
        """Clear annotations for a specific tram track ID in the current frame."""
        tid = simpledialog.askinteger("Input", "Enter Tram Track ID to clear:")
        if tid is None:
            return
        if self.current_time in self.tram_tracks:
            self.tram_tracks[self.current_time] = [t for t in self.tram_tracks[self.current_time] if t[0] != tid]
            self.show_frame()

    def save_annotations(self):
        with open("annotations.txt", "w") as f:
            # Write column titles for bicycles
            f.write("Type,Time,X,Y,BicycleID,WheelType\n")
            for time, annotations in self.annotations.items():
                for coord, bid, wheel_type in annotations:
                    x, y = coord
                    f.write(f"Bicycle,{time},{x},{y},{bid},{wheel_type}\n")

        with open("tram_tracks.txt", "w") as f:
            # Write column titles for tram tracks
            f.write("Type,Time,X,Y,TramTrackID\n")
            for time, tracks in self.tram_tracks.items():
                for tid, x, y in tracks:
                    f.write(f"TramTrack,{time},{x},{y},{tid}\n")

        messagebox.showinfo("Info", "Annotations saved successfully!")

    def show_controls(self):
        controls = """
        Keyboard Shortcuts:
        Ctrl+L - Load Video
        Ctrl+S - Save Annotations
        Ctrl+C - Clear Annotations
        Ctrl+D - Clear Specific ID
        Left Arrow - Previous Frame
        Right Arrow - Next Frame
        """
        messagebox.showinfo("Controls", controls)


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
