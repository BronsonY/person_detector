import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QComboBox, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import cv2
from yolov3.utils import Load_Yolo_model
from yolov3.configs import *
from detection_window import ObjectTracking  # Assuming you have a separate file for the DetectionWindow
import os

class WebcamWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.video_capture = cv2.VideoCapture(0, cv2.CAP_ANY )
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Person Detector')
        self.setGeometry(100, 100, 640, 480)

        image_path = "bg-img.jpg"

        # Set an image as the background of the QLabel
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label = QLabel(self)
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.webcam_combo = QComboBox(self)
        self.populate_webcams()

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_webcam)

        # Set button color and size
        self.start_button.setStyleSheet("background-color: #073763; color: white;")
        self.start_button.setFixedHeight(30)
        self.webcam_combo.setStyleSheet("background-color: #073763; color: white;")
        self.webcam_combo.setFixedHeight(30)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video_label)
        layout.addWidget(self.webcam_combo)
        layout.addWidget(self.start_button)

        self.show()

    def populate_webcams(self):
        i = 0
        while True:
            temp_capture = cv2.VideoCapture(cv2.CAP_ANY  + i)
            if not temp_capture.isOpened():
                break
            temp_capture.release()
            self.webcam_combo.addItem(f'Camera {i + 1}')
            i += 1

    def start_webcam(self):
        selected_webcam_index = self.webcam_combo.currentIndex()
        if selected_webcam_index >= 0:
            self.video_capture.release()
            self.video_capture = cv2.VideoCapture(cv2.CAP_ANY + selected_webcam_index)
            if not self.video_capture.isOpened():
                print("Error: Unable to open the selected webcam.")
                return
            self.timer.stop()  # Stop the timer before opening the new window
            self.open_detection_window(selected_webcam_index)
        else:
            print("Please select a Camera.")

    def open_detection_window(self, webcam_index):
        detection_window = ObjectTracking(video_path=webcam_index, output_path="detection2.avi",
                                  input_size=YOLO_INPUT_SIZE, show=True, iou_threshold=0.1,
                                  Track_only=["person"], CLASSES=YOLO_COCO_CLASSES)
        detection_window.run()

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.timer.stop()
        self.video_capture.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WebcamWindow()
    sys.exit(app.exec_())