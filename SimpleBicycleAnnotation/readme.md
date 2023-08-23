GUI for annotating and tracking bicycles crossing tram tracks.

To create a standalone app: pyinstaller --onefile annotateimage.py

Built app: <https://drive.google.com/file/d/1sNqW4D8gRolIi4EgPBPlplcI2O_4Rxvw/view?usp=sharing> 

Example video: <https://drive.google.com/file/d/1Yp2f51HBcForeNRBcXFL9Lmdvtc7cQqn/view?usp=sharing>


**To do:**

_Short term:_
1. annotate bicycle front and rear wheels while tracking IDs accross frames ✔
2. annotate tracks ✔
3. read .tcal file
4. filter bicycle tracks (fixed distance between points)
5. automatically calculate crossing angles
6. display 3D plot
7. calculate N_UC
8. plot a heat map of riskiest areas
9. output image of heatmap
10. Make How-to video


_Long term:_
1. incorporate automatic tracking
