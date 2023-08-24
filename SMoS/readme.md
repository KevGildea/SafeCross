GUI for reading in calculated crossing angles, and applying Eqn 2:


\[ N_{UC} = N_{C} \times \frac{\sum_{m=1}^{M}\left\{ 1-\frac{e^{\alpha+\beta x_m}}{1+e^{\alpha+\beta x_m}} \right\}}{M} \]

\[ \alpha, \beta \text{ are taken from the modelling,} \]
\[ \underline{x} = \underline{\theta} \text{ for model (b),} \]
\[ \underline{x} = \underline{EW} = \left[ \frac{Gap}{\sin(\theta_1)}, \dots, \frac{Gap}{\sin(\theta_M)} \right] \text{ for model (c).} \]




To create a standalone app: pyinstaller --onefile _.py

Built app: <>

Example video: <>


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
11. make How-to video demonstrating functionality and features


_Long term:_
1. incorporate YOLOv5xDCC
2. use projection to estimate point to use within bounding box
3. Try retraining YOLOv8, with discussed enhancements for bicycle/rider tracking
4. plot a heat map of riskiest areas
5. output image of heatmap
