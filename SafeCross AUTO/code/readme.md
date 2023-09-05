Source code and files used to generate the application.


To create a standalone app:

1. create venv
2. download [SafeCrossAUTO.py](https://github.com/KevGildea/SafeCross/blob/main/SafeCross%20AUTO/code/SafeCrossAUTO.py)
3. clone [YOLOv5](https://github.com/ultralytics/yolov5/) and [SORT](https://github.com/abewley/sort) into the same folder
4. install requirements for all three 
5. run:
```bash
pyinstaller --add-data "your\path\SafeCrossAUTOvenv\Lib\site-packages\ultralytics\cfg\default.yaml;ultralytics/cfg/" --hidden-import=ultralytics --hidden-import=git --hidden-import=PIL --hidden-import=yaml --onefile SafeCrossAUTO.py
```

[test](https://drive.google.com/file/d/12KIOzM5GUcvQv9erIQcXPCmKm7LO0QnP/view?usp=sharing)https://drive.google.com/file/d/12KIOzM5GUcvQv9erIQcXPCmKm7LO0QnP/view?usp=sharing
