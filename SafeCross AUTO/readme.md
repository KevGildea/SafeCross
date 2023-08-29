

## **SafeCross AUTO**


<div align="center">
    <img src="../images/xlsx_tracks.png" width="150" />
</div>

<div align="center">
    <img src="../images/down-arrow-png-down-arrow-sketch-free-icon-512.png" width="50" />
</div>



<div align="center">
    <table>
        <tr>
            <td><img src="../images/SafeCross AUTO.gif" width="500" /></td> 
        </tr>
        <tr>
            <th>Applies retrained YOLOv5x (red bounding boxes), and SORT algorithm for tracking (green bounding boxes)</th>
        </tr>
    </table>
</div>


<div align="center">
    <img src="../images/down-arrow-png-down-arrow-sketch-free-icon-512.png" width="50" />
</div>




<div align="center">
    <table>
        <tr>
            <td><img src="../images/xlsx_annotationsandcrossings.png" width="300" /></td>
            <td><img src="../SafeCross AUTO/example output/287ALL/Sceneplot_WorldCoords.png" width="600" /></td> 
        </tr>
        <tr>
            <th>Output all data in .xlsx files</th>
            <th>Automatically convert to world coordinates, and process to calculate crossing angles</th>
        </tr>
    </table>
</div>


## Requirements

To run this code, you'll need to install the following Python packages:

### Libraries
- **torch**: For deep learning tasks, specifically [YOLOv5](https://github.com/ultralytics/yolov5) model.
- **cv2 (OpenCV)**: For video and image processing.
- **numpy**: For numerical operations.
- **pandas**: For data manipulation and analysis.
- **tkinter**: For GUI dialogs.
- **scipy**: Specifically for the Savitzky-Golay filter.
- **matplotlib**: For plotting and data visualization.

### Custom Modules
- **sort**: A custom module for SORT ([Simple Online and Realtime Tracking](https://github.com/abewley/sort)).

### File Formats
- **Excel files (.xlsx)**: For reading tram track data and writing annotations.
- **TACAL files (.tacal)**: For camera calibration parameters for tomography transformation - can be obtained from the [T-calibration](https://bitbucket.org/TrafficAndRoads/tanalyst/downloads/) tool, or similar, so long as the format is of the form [example.tacal](https://github.com/KevGildea/SafeCross/blob/main/SafeCross%20AUTO/example%20data/Dublin/tacal/example.tacal).

### **Retrained YOLO model**
- <>




_short term:_
1. incorporate YOLOv5xDCC and SORT ✔
2. add option for user to specify confidence level threshold ✔
3. use projection to estimate point to use within bounding box
5. plot coordinate system
6. read in tram track annotations ✔
7. plot world coordinate system and tram tracks on the images ✔
8. incorporate risk models for plotting heat map




_Long term:_
1. Try retraining YOLOv8, with discussed enhancements for bicycle/rider tracking

