GUI for reading in calculated crossing angles, and applying Eqn 2:


![equation](https://latex.codecogs.com/svg.latex?N_{UC}=N_{C}\times\frac{\sum_{m=1}^{M}{1-\frac{e^{\alpha+\beta%20x_m}}{1+e^{\alpha+\beta%20x_m}}}}{M})




![equation](https://latex.codecogs.com/svg.latex?\alpha,\beta\text{are%20taken%20from%20the%20modelling,})

![equation](https://latex.codecogs.com/svg.latex?\underline{x}=\underline{\theta}\text{for%20model%20(b),})

![equation](https://latex.codecogs.com/svg.latex?\underline{x}=\underline{EW}=\left[\frac{Gap}{\sin(\theta_1)},\dots,\frac{Gap}{\sin(\theta_M)}\right]\text{for%20model%20(c).})





To create a standalone app: pyinstaller --onefile _.py

Built app: <https://drive.google.com/file/d/1ABynPvPQmtPVHS3FKaWbubzpv3U1sL3p/view?usp=sharing>



**To do:**

_Short term:_ 
1. read in crossing angles from an excel file with a column marked 'Angle' ✔
2. calculate and plots using model b and/or c  ✔
3. calculate and output N_UC for a user provided count of cyclists ✔
4. make How-to video demonstrating functionality and features
5. add example data


_Long term:_
1. automatically count cyclists for N_UC calculation (in conjunction with SafeCross TA or similar)
2. plot a heat map of riskiest areas (in conjunction with SafeCross TA or similar)
3. output image of heatmap (in conjunction with SafeCross TA or similar)