Tool for annotating and tracking bicycles crossing tram tracks, given 

1) a stationary video,
2) a .tcal calibration file ([T-Calibration](https://bitbucket.org/TrafficAndRoads/tanalyst/downloads/))



Built SafeCross TA tool: <https://drive.google.com/file/d/1xjvP95p90opQc1klEcRlylIqykLPMiam/view?usp=sharing>



**To do:**

_Short term:_
1. annotate bicycle front and rear wheels while tracking IDs accross frames ✔
2. add option for tracking of central position of bicycle only ✔
3. annotate tracks ✔
4. read .tcal file and calculate pixel-to-world coordinates ✔
5. plot global coordinate system at the origin using world-to-pixel transformation ✔
6. output as xlsx ✔
7. Plot in world coordinates ✔
9. automatically calculate crossing angles ✔
10. make How-to video demonstrating functionality and features


_Long term:_
1. incorporate YOLOv5xDCC
2. use projection to estimate point to use within bounding box
3. Try retraining YOLOv8, with discussed enhancements for bicycle/rider tracking
4. plot a heat map of riskiest areas
5. output image of heatmap
