GUI for annotating and tracking bicycles crossing tram tracks.

To create a standalone app: pyinstaller --onefile annotateimage.py

Built app: <https://drive.google.com/file/d/1sNqW4D8gRolIi4EgPBPlplcI2O_4Rxvw/view?usp=sharing> 

Example video: <https://drive.google.com/file/d/1Yp2f51HBcForeNRBcXFL9Lmdvtc7cQqn/view?usp=sharing>


**To do:**

_Short term:_
1. annotate bicycle front and rear wheels while tracking IDs accross frames ✔
2. add option for tracking of central position of bicycle only
3. annotate tracks ✔
4. read .tcal file
5. filter bicycle tracks (fixed distance between points)
6. automatically calculate crossing angles
7. display 3D plot
8. calculate N_UC
9. plot a heat map of riskiest areas
10. output image of heatmap
11. make How-to video demonstrating functionality and features


_Long term:_
1. incorporate automatic tracking
