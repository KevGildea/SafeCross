import torch
import cv2
from sort.sort import Sort
import tkinter as tk
from tkinter import filedialog, simpledialog  # Import simpledialog
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Prompt user to select video file
root = tk.Tk()
root.title("SafeCross AUTO")
#root.withdraw()  # Hide the main window
root.geometry("300x300")

def show_about():
    about_window = tk.Toplevel(root)
    about_window.title("About")

    info = (
        "Developed by Kevin Gildea, Ph.D.\n"
        "Faculty of Engineering, LTH\n"
        "Lund University\n"
        "Email: kevin.gildea@tft.lth.se"
    )

    label = tk.Label(about_window, text=info, font=("Arial", 8))
    label.pack(pady=15)


# Create menu bar
menu_bar = tk.Menu(root)

# Create 'Help' menu
help_menu = tk.Menu(menu_bar, tearoff=0)
help_menu.add_command(label="About", command=show_about)

# Add 'Help' menu to menu bar
menu_bar.add_cascade(label="Help", menu=help_menu)

# Add menu bar to root window
root.config(menu=menu_bar)



# Prompt user to select tram_tracks.xlsx file
tram_tracks_path = filedialog.askopenfilename(title="Select the tram_tracks.xlsx file", filetypes=[("Excel files", "*.xlsx")])
if not tram_tracks_path:
    print("No tram_tracks.xlsx file selected. Exiting.")
    exit()
df = pd.read_excel(tram_tracks_path, engine='openpyxl')
# Read the Excel file
#df = pd.read_excel('tram_tracks.xlsx', engine='openpyxl')


video_path = filedialog.askopenfilename(title="Select the video file")
if not video_path:
    print("No video file selected. Exiting.")
    exit()



# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
#model = torch.hub.load('yolov5', 'yolov5x', source='local', pretrained=True)

# Prompt user to select custom YOLOv5x model file
model_path = filedialog.askopenfilename(title="Select the YOLOv5x model file", filetypes=[("PyTorch files", "*.pt")])
if not model_path:
    print("No YOLOv5x model file selected. Exiting.")
    exit()
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')


# Prompt user to specify the confidence level for YOLO
confidence_threshold = simpledialog.askfloat("Input", "Enter confidence level for YOLO (e.g., 0.5):", minvalue=0.0, maxvalue=1.0)
if confidence_threshold is None:  # If user closes the dialog or cancels
    print("No confidence level specified. Using default value of 0.5.")
    confidence_threshold = 0.5

# Prompt user to specify the tracking point within the bounding box
tracking_point_options = ['centre', 'centre-left', 'centre-right', 'top-centre', 'top-left', 'top-right', 'bottom-centre', 'bottom-left', 'bottom-right']
tracking_point = simpledialog.askstring("Input", "Specify the tracking point within the bounding box:", initialvalue="bottom-centre")
if tracking_point not in tracking_point_options:
    print(f"Invalid tracking point specified. Using default value of 'bottom-centre'.")
    tracking_point = "bottom-centre"

# Initialize SORT tracker
tracker = Sort()

# Dictionary to store historical centers for each bicycle
historical_centers = {}


# Open video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# List of road users to track
road_users = ['bicycle']  #, 'car', 'motorcycle', 'bus', 'truck', 'person'



world_coordinates_list = []

class CameraCalibration:
    def __init__(self):
        self.calibration_params = {}
        self.load_tacal_file()

    def load_tacal_file(self):
        tacal_path = filedialog.askopenfilename(filetypes=[("TACAL files", "*.tacal")])
        if not tacal_path:
            return

        with open(tacal_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                key, value = line.strip().split(":")
                self.calibration_params[key.strip()] = float(value.strip())
        

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


# Initialize CameraCalibration
camera_calib = CameraCalibration()




while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            break



        # Inference : DO NOT DRAW ANYTHING ON THE IMAGE UNTIL AFTER
        results = model(frame)

        # Filter out detections that are road users and meet the confidence threshold
        road_user_detections = [d for d in results.xyxy[0] if results.names[int(d[5])] in road_users and d[4] > confidence_threshold]

        # Extract bounding boxes for road users
        road_user_boxes = [d[:4].cpu().numpy() for d in road_user_detections]

        # Draw raw YOLO detections in red
        for d in road_user_detections:
            box = d[:4].cpu().numpy()
            confidence = d[4].item()  # Extract confidence value
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Display confidence

        # Convert road_user_boxes to a numpy array and update the tracker
        if road_user_boxes:  # Check if there are any detections
            road_user_boxes_np = np.array(road_user_boxes)
            trackers = tracker.update(road_user_boxes_np)

            # Draw tracked road users on the frame in green
            for track in trackers:
                cv2.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {int(track[4])}", (int(track[0]), int(track[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Plot tram tracks
        # Group by TramTrackID
        grouped = df.groupby('TramTrackID')

        for _, group in grouped:
            # Sort the group if necessary (e.g., by index)
            sorted_group = group.sort_values(by='TramTrackID')  # Replace 'index' with the appropriate column if needed
            
            # Extract the coordinates
            coords = sorted_group[['X', 'Y']].values
            
            # Plot lines between consecutive points
            for i in range(1, len(coords)):
                cv2.line(frame, tuple(coords[i-1]), tuple(coords[i]), (0, 128, 0), 2)  # Draw a green line

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        for track in trackers:
            width = track[2] - track[0]
            height = track[3] - track[1]
            
            if tracking_point == "centre":
                center_x = track[0] + width / 2
                center_y = track[1] + height / 2
            elif tracking_point == "centre-left":
                center_x = track[0] + width / 4
                center_y = track[1] + height / 2
            elif tracking_point == "centre-right":
                center_x = track[0] + 3 * width / 4
                center_y = track[1] + height / 2
            elif tracking_point == "top-centre":
                center_x = track[0] + width / 2
                center_y = track[1] + height / 4
            elif tracking_point == "top-left":
                center_x = track[0] + width / 4
                center_y = track[1] + height / 4
            elif tracking_point == "top-right":
                center_x = track[0] + 3 * width / 4
                center_y = track[1] + height / 4
            elif tracking_point == "bottom-centre":
                center_x = track[0] + width / 2
                center_y = track[1] + 3 * height / 4
            elif tracking_point == "bottom-left":
                center_x = track[0] + width / 4
                center_y = track[1] + 3 * height / 4
            elif tracking_point == "bottom-right":
                center_x = track[0] + 3 * width / 4
                center_y = track[1] + 3 * height / 4

            # Store the center in the historical_centers dictionary
            track_id = int(track[4])
            if track_id not in historical_centers:
                historical_centers[track_id] = []
            historical_centers[track_id].append((int(center_x), int(center_y)))

            world_x, world_y = camera_calib.pixel_to_world(center_x, center_y)
            world_coordinates_list.append(["Bicycle", frame_number, center_x, center_y, int(track[4]), world_x[0], world_y[0]])

        # Plot historical centers for each tracked bicycle
        for track_id, centers in historical_centers.items():
            for center in centers:
                cv2.circle(frame, center, 1, (255, 255, 255), -1)  # Draw a small white dot

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow('Tracked cyclists', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error occurred: {e}")
        break

df_output = pd.DataFrame(world_coordinates_list, columns=["Type", "Time", "X", "Y", "BicycleID", "WorldX", "WorldY"])
df_output.to_excel("annotations.xlsx", index=False)



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def ccw(A, B, C):
    """Check if points are counterclockwise."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    """Check if line segments AB and CD intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def angle_between(v1, v2):
    """Calculate the angle between two vectors."""
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = dot_product / magnitude_product
    angle = np.arccos(cos_theta) * (180 / np.pi)  # Convert to degrees
    return angle if angle <= 90 else 180 - angle

# Load the data
annotations = pd.read_excel('annotations.xlsx')
tram_tracks = pd.read_excel('tram_tracks.xlsx')

# Group by BicycleID and Time
grouped_bikes = annotations.groupby(['BicycleID', 'Time'])

# Directly use the world coordinates for the center of each bicycle
bike_centers = []

for (bike_id, time), group in grouped_bikes:
    center_coords = group[['WorldX', 'WorldY']].values[0]
    bike_centers.append((bike_id, time, center_coords[0], center_coords[1]))

bike_centers_df = pd.DataFrame(bike_centers, columns=['BicycleID', 'Time', 'WorldX', 'WorldY'])
grouped_trams = tram_tracks.groupby('TramTrackID')


# Prompt user to specify the window length for Savitzky-Golay filter
window_length = simpledialog.askinteger("Savitzky-Golay filter", "Specify window length (must be an odd number):", initialvalue=31)
if window_length is None or window_length % 2 == 0:  # Ensure it's an odd number
    print("Invalid window length specified. Using default value of 31.")
    window_length = 31


# Apply Savitzky-Golay filter to the bicycle trajectories
polynomial_order = 2  

# Create a list to store indices of rows to be dropped
drop_indices = []

for bike_id in bike_centers_df['BicycleID'].unique():
    bike_data = bike_centers_df[bike_centers_df['BicycleID'] == bike_id]
    
    # Check the size of the group
    if len(bike_data) < window_length:
        # Store indices of this group to be dropped later
        drop_indices.extend(bike_data.index.tolist())
    else:
        bike_centers_df.loc[bike_data.index, 'WorldX'] = savgol_filter(bike_data['WorldX'], window_length, polynomial_order)
        bike_centers_df.loc[bike_data.index, 'WorldY'] = savgol_filter(bike_data['WorldY'], window_length, polynomial_order)

# Drop the rows corresponding to the small groups
bike_centers_df.drop(drop_indices, inplace=True)

# Prepare to collect crossing data
crossings = []
crossing_id = 0

# Plot tram tracks and check for intersections with bicycle trajectories
for tram_id, tram_group in grouped_trams:
    tram_coords = list(zip(tram_group['WorldX'], tram_group['WorldY']))

    for bike_id in bike_centers_df['BicycleID'].unique():
        bike_data = bike_centers_df[bike_centers_df['BicycleID'] == bike_id]
        bike_coords = list(zip(bike_data['WorldX'], bike_data['WorldY']))

        for i in range(len(bike_coords) - 1):
            for j in range(len(tram_coords) - 1):
                if intersect(bike_coords[i], bike_coords[i+1], tram_coords[j], tram_coords[j+1]):
                    # Calculate intersection point using numpy's line intersection function
                    A = np.array(bike_coords[i])
                    B = np.array(bike_coords[i+1])
                    C = np.array(tram_coords[j])
                    D = np.array(tram_coords[j+1])
                    X = np.linalg.solve([[B[0] - A[0], C[0] - D[0]], [B[1] - A[1], C[1] - D[1]]], [C[0] - A[0], C[1] - A[1]])
                    intersection = A + X[0] * (B - A)

                    # Calculate the crossing angle
                    bike_vector = B - A
                    tram_vector = D - C
                    crossing_angle = angle_between(bike_vector, tram_vector)

                    # Append to crossings list
                    crossings.append([crossing_angle, bike_id, tram_id, intersection[0], intersection[1]])

# Convert crossings to DataFrame and save to xlsx
crossings_df = pd.DataFrame(crossings, columns=['Angle', 'BicycleID', 'TrackID', 'WorldX', 'WorldY'])
crossings_df.to_excel('crossings_data.xlsx', index=False)

# Plotting
fig, ax = plt.subplots()

# Add grid with spacing of 1 meter
ax.grid(which='both', linestyle='--', linewidth=0.5)
ax.set_xticks(np.arange(int(min(tram_tracks['WorldX'].min(), annotations['WorldX'].min())), int(max(tram_tracks['WorldX'].max(), annotations['WorldX'].max())) + 1, 1))
ax.set_yticks(np.arange(int(min(tram_tracks['WorldY'].min(), annotations['WorldY'].min())), int(max(tram_tracks['WorldY'].max(), annotations['WorldY'].max())) + 1, 1))

# Adjust the fontsize of the x-axis and y-axis tick labels
ax.tick_params(axis='both', which='major', labelsize=8)


# Plot tram tracks
for tram_id, tram_group in grouped_trams:
    ax.plot(tram_group['WorldX'], tram_group['WorldY'], linestyle='--', color='green')
    # Plot tram ID
    tram_midpoint = tram_group.iloc[len(tram_group) // 2]
    ax.text(tram_midpoint['WorldX'], tram_midpoint['WorldY'], f'Track {tram_id}', fontsize=7, color='green')

# Plot bicycle trajectories and intersections
for bike_id in bike_centers_df['BicycleID'].unique():
    bike_data = bike_centers_df[bike_centers_df['BicycleID'] == bike_id]
    ax.plot(bike_data['WorldX'], bike_data['WorldY'])
    # Plot bike ID
    bike_midpoint = bike_data.iloc[len(bike_data) // 2]
    #ax.text(bike_midpoint['WorldX'], bike_midpoint['WorldY'], f'Bike {bike_id}', fontsize=7, color='black')

for crossing in crossings:
    ax.plot(crossing[3], crossing[4], 'rx')
    ax.text(crossing[3], crossing[4], f"{crossing[0]:.2f}°", fontsize=7, verticalalignment='bottom')

ax.set_xlabel('WorldX', fontsize=10)
ax.set_ylabel('WorldY', fontsize=10)
ax.set_aspect('equal')
ax.invert_yaxis()  # This will reverse the y-axis for visualisation (an artifact of calibration)
plt.savefig('Sceneplot_WorldCoords.png', dpi=300)  # Save with a resolution of 300 dpi


cap.release()
out.release()  # Release the VideoWriter
cv2.destroyAllWindows()
