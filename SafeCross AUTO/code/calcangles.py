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
bike_centers = []

# Calculate the center of the bicycle for each group
for (bike_id, time), group in grouped_bikes:
    if 'front' in group['WheelType'].values and 'rear' in group['WheelType'].values:
        front_coords = group[group['WheelType'] == 'front'][['WorldX', 'WorldY']].values[0]
        rear_coords = group[group['WheelType'] == 'rear'][['WorldX', 'WorldY']].values[0]
        center_x = (front_coords[0] + rear_coords[0]) / 2
        center_y = (front_coords[1] + rear_coords[1]) / 2
        bike_centers.append((bike_id, time, center_x, center_y))
    elif 'center' in group['WheelType'].values:
        center_coords = group[group['WheelType'] == 'center'][['WorldX', 'WorldY']].values[0]
        bike_centers.append((bike_id, time, center_coords[0], center_coords[1]))

bike_centers_df = pd.DataFrame(bike_centers, columns=['BicycleID', 'Time', 'WorldX', 'WorldY'])
grouped_trams = tram_tracks.groupby('TramTrackID')

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
    ax.text(bike_midpoint['WorldX'], bike_midpoint['WorldY'], f'Bike {bike_id}', fontsize=7, color='black')

for crossing in crossings:
    ax.plot(crossing[3], crossing[4], 'ro')
    ax.text(crossing[3], crossing[4], f"{crossing[0]:.2f}°", fontsize=7, verticalalignment='bottom')

ax.set_xlabel('WorldX')
ax.set_ylabel('WorldY')
ax.set_aspect('equal')
plt.show()
