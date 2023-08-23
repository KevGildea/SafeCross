import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Menu
import cv2
from PIL import Image, ImageTk
import numpy as np

class VideoAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Annotator")
        
        # Menu bar
        menubar = Menu(root)
        
        # File menu
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Video", command=self.load_video, accelerator="Ctrl+L")
        filemenu.add_command(label="Load TACAL File", command=self.load_tacal_file, accelerator="Ctrl+T")
        root.bind("<Control-t>", lambda event: self.load_tacal_file())
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
        
        self.calibration_params = {}

    def load_tacal_file(self):
        tacal_path = filedialog.askopenfilename(filetypes=[("TACAL files", "*.tacal")])
        if not tacal_path:
            return

        with open(tacal_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                key, value = line.strip().split(":")
                self.calibration_params[key.strip()] = float(value.strip())
        print(self.calibration_params)
        

    def groundProjectPoint(self, image_point, int_mat, ext_mat, z=0.0):
        camMat = np.array(int_mat)
        rotMat = np.array([
            [ext_mat[0][0], ext_mat[0][1], ext_mat[0][2]], 
            [ext_mat[1][0], ext_mat[1][1], ext_mat[1][2]],
            [ext_mat[2][0], ext_mat[2][1], ext_mat[2][2]]
        ])

        iRot = np.linalg.inv(rotMat)
        iCam = np.linalg.inv(camMat)
        tvec = [[ext_mat[0][3]], [ext_mat[1][3]], [ext_mat[2][3]]]

        uvPoint = np.ones((3, 1))
        uvPoint[0, 0] = image_point[0]
        uvPoint[1, 0] = image_point[1]

        tempMat = iRot @ iCam @ uvPoint
        tempMat2 = iRot @ tvec

        s = (z + tempMat2[2, 0]) / tempMat[2, 0]
        wcPoint = iRot @ ((s * iCam @ uvPoint) - tvec)

        assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
        wcPoint[2] = z

        return wcPoint

    def pixel_to_world(self, x, y):
        # Construct the intrinsic matrix from self.calibration_params
        int_mat = [
            [self.calibration_params['f'], 0, self.calibration_params['Cx']],
            [0, self.calibration_params['f'], self.calibration_params['Cy']],
            [0, 0, 1]
        ]

        # Construct the extrinsic matrix from self.calibration_params
        ext_mat = [
            [self.calibration_params['r1'], self.calibration_params['r2'], self.calibration_params['r3'], self.calibration_params['Tx']],
            [self.calibration_params['r4'], self.calibration_params['r5'], self.calibration_params['r6'], self.calibration_params['Ty']],
            [self.calibration_params['r7'], self.calibration_params['r8'], self.calibration_params['r9'], self.calibration_params['Tz']]
        ]

        world_coords = self.groundProjectPoint((x, y), int_mat, ext_mat)
        return world_coords[0], world_coords[1]


    def world_to_pixel(self, world_x, world_y, world_z=0.0):
        """Convert world coordinates to pixel coordinates."""
        # Construct the intrinsic matrix from self.calibration_params
        int_mat = np.array([
            [self.calibration_params['f'], 0, self.calibration_params['Cx']],
            [0, self.calibration_params['f'], self.calibration_params['Cy']],
            [0, 0, 1]
        ])

        # Construct the extrinsic matrix from self.calibration_params
        ext_mat = np.array([
            [self.calibration_params['r1'], self.calibration_params['r2'], self.calibration_params['r3'], self.calibration_params['Tx']],
            [self.calibration_params['r4'], self.calibration_params['r5'], self.calibration_params['r6'], self.calibration_params['Ty']],
            [self.calibration_params['r7'], self.calibration_params['r8'], self.calibration_params['r9'], self.calibration_params['Tz']]
        ])

        world_point = np.array([[world_x], [world_y], [world_z], [1]])
        image_point = int_mat @ ext_mat @ world_point
        image_point /= image_point[2]  # Homogeneous to cartesian coordinates

        return int(image_point[0]), int(image_point[1])



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
                if wheel_type == 'front':
                    color = 'blue'
                elif wheel_type == 'rear':
                    color = 'red'
                elif wheel_type == 'center':
                    color = 'yellow'
                self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=color)
                self.canvas.create_text(x, y-10, text=str(bid), fill='white')


        # Draw lines for bicycles
        if self.current_time in self.annotations:
            bicycle_points = {}
            for coord, bid, wheel_type in self.annotations[self.current_time]:
                if bid not in bicycle_points:
                    bicycle_points[bid] = {}
                bicycle_points[bid][wheel_type] = coord

            for bid, points in bicycle_points.items():
                if 'front' in points and 'rear' in points:
                    self.canvas.create_line(points['front'], points['rear'], fill='purple')
                if 'center' in points:
                    if 'front' in points:
                        self.canvas.create_line(points['center'], points['front'], fill='purple')
                    if 'rear' in points:
                        self.canvas.create_line(points['center'], points['rear'], fill='purple')


        # Display tram track annotations
        if self.current_time in self.tram_tracks:
            for tid, x, y in self.tram_tracks[self.current_time]:
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill='green')
                self.canvas.create_text(x, y-10, text=str(tid), fill='white')


        # Draw lines for tram tracks
        if self.current_time in self.tram_tracks:
            tram_points = sorted(self.tram_tracks[self.current_time], key=lambda x: x[0])
            for i in range(len(tram_points) - 1):
                _, x1, y1 = tram_points[i]
                _, x2, y2 = tram_points[i + 1]
                self.canvas.create_line(x1, y1, x2, y2, fill='green')


        # Plot the origin of the world coordinate system
        origin_x, origin_y = self.world_to_pixel(0, 0)
        self.canvas.create_oval(origin_x-5, origin_y-5, origin_x+5, origin_y+5, fill='white')
        self.canvas.create_text(origin_x, origin_y-10, text="Origin", fill='white')

        # Plot X, Y, Z axes
        x_end_x, x_end_y = self.world_to_pixel(1, 0)
        y_end_x, y_end_y = self.world_to_pixel(0, 1)
        z_end_x, z_end_y = self.world_to_pixel(0, 0, 1)

        self.canvas.create_line(origin_x, origin_y, x_end_x, x_end_y, fill='red', arrow=tk.LAST)  # X-axis in red
        self.canvas.create_line(origin_x, origin_y, y_end_x, y_end_y, fill='green', arrow=tk.LAST)  # Y-axis in green
        self.canvas.create_line(origin_x, origin_y, z_end_x, z_end_y, fill='blue', arrow=tk.LAST)  # Z-axis in blue


        # Update the label with frame number and time
        self.label.config(text=f"Frame: {self.current_time}, Time: {self.current_time} seconds, Mode: {self.mode}")


    def annotate_position(self, event):
        x, y = event.x, event.y
        if self.mode == "Bicycle":
            bid = simpledialog.askinteger("Input", "Enter Bicycle ID:")
            if bid is None:
                return
            wheel_type = simpledialog.askstring("Input", "Enter Wheel Type (f for front/r for rear/c for center):")
            if wheel_type == 'f':
                wheel_type = 'front'
            elif wheel_type == 'r':
                wheel_type = 'rear'
            elif wheel_type == 'c':
                wheel_type = 'center'
            if wheel_type not in ['front', 'rear', 'center']:
                messagebox.showerror("Error", "Invalid wheel type. Please enter 'f' for front, 'r' for rear, or 'c' for center.")
                return
            if self.current_time not in self.annotations:
                self.annotations[self.current_time] = []
            self.annotations[self.current_time].append(((x, y), bid, wheel_type))
            color = 'blue' if wheel_type == 'front' else ('red' if wheel_type == 'rear' else 'yellow')
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
            f.write("Type,Time,X,Y,BicycleID,WheelType,WorldX,WorldY\n")
            for time, annotations in self.annotations.items():
                for coord, bid, wheel_type in annotations:
                    x, y = coord
                    world_x, world_y = self.pixel_to_world(x, y)
                    f.write(f"Bicycle,{time},{x},{y},{bid},{wheel_type},{world_x},{world_y}\n")

        with open("tram_tracks.txt", "w") as f:
            # Write column titles for tram tracks
            f.write("Type,Time,X,Y,TramTrackID,WorldX,WorldY\n")
            for time, tracks in self.tram_tracks.items():
                for tid, x, y in tracks:
                    world_x, world_y = self.pixel_to_world(x, y)
                    f.write(f"TramTrack,{time},{x},{y},{tid},{world_x},{world_y}\n")

        messagebox.showinfo("Info", "Annotations with world coordinates saved successfully!")

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
