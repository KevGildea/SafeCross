

## **1) SafeCross TA**
[View Details](../trajectory%20annotation)

<div align="center">
    <img src="./images/SafeCross TA.gif" width="400" />
</div>

<div align="center">
    <img src="./images/down-arrow-png-down-arrow-sketch-free-icon-512.png" width="50" />
</div>




<div align="center">
    <table>
        <tr>
            <td><img src="./images/xlsx_files.png" width="400" /></td>
            <td><img src="./trajectory annotation/example output/Sceneplot_WorldCoords.png" width="400" /></td>
        </tr>
        <tr>
            <!-- Add your headings here -->
            <th>Output all data in .xlsx files</th>
            <th>Automatically convert to world coordinates, and process to calculate crossing angles</th>
        </tr>
    </table>
</div>









Built SafeCross TA tool: <https://drive.google.com/file/d/1kptsozS1HeLlvq52q8bqFGerR2BdwloJ/view?usp=sharing>
[T-Calibration](https://bitbucket.org/TrafficAndRoads/tanalyst/downloads/)



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
10. plot y axis from negative to positive ✔
11. add filtering and polynomial fitting functionalities
12. add option to change framerate
13. make How-to video demonstrating functionality and features


_Long term:_
1. incorporate YOLOv5xDCC
2. use projection to estimate point to use within bounding box
3. Try retraining YOLOv8, with discussed enhancements for bicycle/rider tracking
4. plot a heat map of riskiest areas
5. output image of heatmap
