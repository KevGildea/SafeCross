

## **SafeCross AUTO**



<div align="center">
<img src="../images/SafeCross AUTO.gif" width="300" />
</div>


<div align="center">
<img src="../images/down-arrow-png-down-arrow-sketch-free-icon-512.png" width="50" />
</div>


<div align="center">
<img src="../SafeCross AUTO/example output/287ALL/Sceneplot_WorldCoords.png" width="500" />
</div>

<div align="center">
<img src="../SafeCross AUTO/example output/287ALL/blended_heatmap.png" width="500" />
</div>


### Directory format


```plaintext
.
├── src
│   ├── SafeCross AUTO.exe
│   └── yolov5
│       ├── .github
│       └── classify
│       ...
├── data
│   ├── tram_tracks.xlsx
│   ├── example.mp4
│   └── example.tacal
├── model
    └── yolov5x.pt (retrained on Dublin data)
```

### Downloads
The standalone application can be downloaded from this Google Drive [link](https://drive.google.com/file/d/1XtshKne4uYplc2j4_M1CqLpofHj9cTMJ/view?usp=sharing), `yolov5` is the GitHub repository available [here](https://github.com/ultralytics/yolov5/), and `yolov5x.pt` can be downloaded from this Google Drive [link](https://drive.google.com/file/d/1QWWPHsp7amwtncWRP3atRaDncN9ohmbK/view?usp=sharing).


### File Formats
- **Excel files (.xlsx)**: For reading tram track data and writing annotations.
- **TACAL files (.tacal)**: For camera calibration parameters for tomography transformation - can be obtained from the [T-calibration](https://bitbucket.org/TrafficAndRoads/tanalyst/downloads/) tool, or similar, so long as the format is of the form [example.tacal](https://github.com/KevGildea/SafeCross/blob/main/SafeCross%20AUTO/example%20data/Dublin/tacal/example.tacal).


## Acknowledgments
Special thanks to the developers of the [YOLO (You Only Look Once)](https://arxiv.org/abs/1506.02640) object detection algorithm at [Ultralytics](https://github.com/ultralytics/), and to Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, and Ben Upcroft for the [SORT (Simple Online and Realtime Tracking)](https://arxiv.org/abs/1602.00763) algorithm. These tools have been essential in the development of this project.




### To-do

_short term:_
1. incorporate YOLOv5xDCC and SORT ✔
2. add option for user to specify confidence level threshold ✔
3. read in tram track annotations ✔
4. plot world coordinate system and tram tracks on the images ✔
5. incorporate risk models for plotting heat map

_Long term:_
1. Try retraining YOLOv8, with discussed enhancements for bicycle/rider tracking

