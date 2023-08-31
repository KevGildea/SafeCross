Source code and files used to generate the application.


To create a standalone app:

1. create venv
2. download SafeCrossAUTO.py
3. clone Yolov5 and SORT into the same folder
4. install requirements
5. run:
'''
pyinstaller --add-data "D:\repositories\Windows\SafeCrossAUTO\SafeCrossAUTOvenv\Lib\site-packages\ultralytics\cfg\default.yaml;ultralytics/cfg/" --hidden-import=ultralytics --hidden-import=git --hidden-import=PIL --hidden-import=yaml --onefile SafeCrossAUTO.py
'''
