

## **SafeCross AUTO**



<div align="center">
<img src="../SafeCross AUTO/example data/Dublin/output/example_output.gif" width="300" />
</div>


<div align="center">
<img src="../images/down-arrow-png-down-arrow-sketch-free-icon-512.png" width="50" />
</div>


<div align="center">
<img src="../SafeCross AUTO/example data/Dublin/output/example_Sceneplot_WorldCoords.png" width="500" />
</div>


<div align="center">
<img src="../images/down-arrow-png-down-arrow-sketch-free-icon-512.png" width="50" />
</div>


<div align="center">
<img src="../SafeCross AUTO/example data/Dublin/output/example_blended_heatmap.png" width="300" />
</div>


### Directory format


```plaintext
.
├── SafeCross AUTO
│   ├── SafeCross AUTO.exe
│   └── yolov5
│       ├── .github
│       └── classify
│       ...
├── data
│   ├── tram_tracks.xlsx (obtained using SafeCross TA)
│   ├── example.mp4
│   └── example.tacal
├── model
    └── yolov5x.pt
```

### Downloads
The standalone application can be downloaded from this Google Drive [link](https://drive.google.com/file/d/12KIOzM5GUcvQv9erIQcXPCmKm7LO0QnP/view?usp=sharing), `yolov5` is the GitHub repository available [here](https://github.com/ultralytics/yolov5/), and `yolov5x.pt` (retrained on Dublin data) can be downloaded from this Google Drive [link](https://drive.google.com/file/d/1gZQsXaXeRKGgOVOmG5X0W-ma7uFE3N6w/view?usp=sharing).

Alternatively, a .zip file with the entire directory contents shown above can be downloaded from this Google Drive [link](https://drive.google.com/file/d/1y8onklZKutPQU_los4LH7xFEDsoNO_YQ/view?usp=sharing).


### File Formats
- **Excel files (.xlsx)**: For reading tram track data and writing annotations.
- **TACAL files (.tacal)**: For camera calibration parameters for homography transformation - can be obtained from the [T-calibration](https://bitbucket.org/TrafficAndRoads/tanalyst/downloads/) tool, or similar, so long as the format is of the form [example.tacal](https://github.com/KevGildea/SafeCross/blob/main/SafeCross%20AUTO/example%20data/Dublin/tacal/example.tacal).


## Acknowledgments
Special thanks to the developers of the [YOLO (You Only Look Once)](https://arxiv.org/abs/1506.02640) object detection algorithm at [Ultralytics](https://github.com/ultralytics/), and to the developers of the [SORT (Simple Online and Realtime Tracking)](https://arxiv.org/abs/1602.00763) algorithm. These tools have been essential in the development of this project.




### To-do

_short term:_
1. incorporate YOLOv5xDCC and SORT ✔
2. add option for user to specify confidence level threshold ✔
3. read in tram track annotations ✔
4. plot world coordinate system and tram tracks on the images ✔
5. incorporate risk models for plotting heat map ✔
6. make a how-to video demonstrating functionality and features
7. update manuals
8. include smaller models, and compare 

_Long term:_
1. try to create GPU version
2. try retraining YOLOv8, with discussed enhancements for bicycle/rider tracking, using oriented bounding boxes ([https://docs.ultralytics.com/tasks/obb/](https://docs.ultralytics.com/tasks/obb/)).

