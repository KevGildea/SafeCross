GUI for annotating and tracking bicycles crossing tram tracks.

To create a standalone app: pyinstaller --onefile annotateimage.py

Built app: <https://drive.google.com/file/d/1hlCrIXd2YjaA4d4PlfwtATVJ92DEjl6t/view?usp=sharing>

Example video: <https://drive.google.com/file/d/1Yp2f51HBcForeNRBcXFL9Lmdvtc7cQqn/view?usp=sharing>


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
10. calculate N_UC
11. plot a heat map of riskiest areas
12. output image of heatmap
13. make How-to video demonstrating functionality and features


_Long term:_
1. incorporate YOLOv5xDCC
2. use projection to estimate point to use within bounding box
3. Try retraining YOLOv8, with discussed enhancements for bicycle/rider tracking
