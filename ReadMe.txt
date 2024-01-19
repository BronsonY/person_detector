# Install the env requirements.
pip install -r ./requirements.txt

# Get the model_data and pjreddie.com from the official yoloV3 doc, please follow the ReadMe.md to get the data.
Or
# yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

# yolov3-tiny
wget -P model_data https://pjreddie.com/media/files/yolov3-tiny.weights

# yolov4
wget -P model_data https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

# yolov4-tiny
wget -P model_data https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```
# RUN
# Run main.py to get the personDetector app.
- python main.py

# detection_window.py is the actual program to run the detection of person and opening up of the camera.
- python detection_window.py 

# BUILD .exe
pip install pyinstaller

## run in the console
pyinstaller --name humanDetector --onefile --windowed --icon=icon.ico --add-data "yoloenv\Lib\site-packages\keras;./keras" --hidden-import tensorflow --hidden-import keras --console main.py

After the exe file is built, use inno setup to make the installation file.
Visit:  jrsoftware.org


# Comparison/System Information

| Sl. No. | Processor                                       | Installed RAM | System Type                                      | Status  | Duration | Hardware Component Type |
|---------|-------------------------------------------------|----------------|--------------------------------------------------|---------|----------|-------------------------|
| 1       | Intel(R) Core(TM) i3-5005U CPU@ 2.00GHz 2.00GHz | 12.0 GB        | 64-bit operating system, x64-based processor    | Working | Slow     | CPU                     |
| 2       | Intel(R) Core(TM) i5-7200U CPU@ 2.50GHz 2.70GHz | 16 GB          | 64-bit operating system, x64-based processor    | Working | Smooth   | CPU                     |
| 3       | Intel(R) Core(TM) i5-8250U CPU@ 1.60GHz 1.80GHz | 8 GB           | 64-bit operating system, x64-based processor    | Working | Slow     | CPU                     |
| 4       | Intel(R) Core(TM) i5-8265U CPU@ 1.60GHz 1.80GHz | 8 GB           | 64-bit operating system, x64-based processor    | Working | Slow     | CPU                     |
| 5       | 11th Gen Intel(R) Core(TM) i3-1115 G4 @ 3.00GHz 2.90GHz | 8 GB  | 64-bit operating system, x64-based processor | Working | Smooth   | CPU                     |

**Note:** Sometimes it takes more time to open up the camera.

# To Do List
Linux build
Deploy in Raspberry Pi 
Check in GPU system

