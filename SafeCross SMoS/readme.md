
## **SafeCross SMoS**


<div align="center">
    <img src="../images/xlsx_crossings.png" width="150" />
</div>


<div align="center">
    <img src="../images/down-arrow-png-down-arrow-sketch-free-icon-512.png" width="50" />
</div>


<div align="center">
    <img src="../images/SafeCross SMoS.PNG" width="400" />
</div>



<div align="center">
    <img src="../images/down-arrow-png-down-arrow-sketch-free-icon-512.png" width="50" />
</div>



<div align="center">
<table>
<tr>
    <td><img src="../SafeCross SMoS/example output/model_b_plot.png" width="400" /></td>
    <td><img src="../SafeCross SMoS/example output/model_c_plot.png" width="400" /></td>
</tr>
</table>
</div>


![equation](https://latex.codecogs.com/svg.latex?N_{UC}=N_{C}\times\frac{\sum_{m=1}^{M}{1-\frac{e^{\alpha+\beta%20x_m}}{1+e^{\alpha+\beta%20x_m}}}}{M})




![equation](https://latex.codecogs.com/svg.latex?\alpha,\beta\text{%20are%20taken%20from%20the%20modelling,})

![equation](https://latex.codecogs.com/svg.latex?\underline{x}=\underline{\theta}\text{%20for%20model%20(b),})

![equation](https://latex.codecogs.com/svg.latex?\underline{x}=\underline{EW}=\left[\frac{Gap}{\sin(\theta_1)},\dots,\frac{Gap}{\sin(\theta_M)}\right]\text{%20for%20model%20(c).})



### Downloads
The standalone application can be downloaded from this Google Drive [link](https://drive.google.com/file/d/1ABynPvPQmtPVHS3FKaWbubzpv3U1sL3p/view?usp=sharing).




### To-do

_Short term:_ 
1. read in crossing angles from an excel file with a column marked 'Angle' ✔
2. calculate and plots using model b and/or c  ✔
3. calculate and output N_UC for a user provided count of cyclists ✔
4. make How-to video demonstrating functionality and features


_Long term:_
1. automatically count cyclists for N_UC calculation (in conjunction with SafeCross TA or similar)
2. plot a heat map of riskiest areas (in conjunction with SafeCross TA or similar)
3. output image of heatmap (in conjunction with SafeCross TA or similar)
