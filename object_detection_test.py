import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
import pygame
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, read_class_names
from yolov3.configs import YOLO_FRAMEWORK, YOLO_COCO_CLASSES, YOLO_INPUT_SIZE


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS2
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Initialize pygame mixer
pygame.mixer.init()

# Load alarm sound file
alarm_sound_file = resource_path('sound\\beep-warning-6387.mp3')  # Replace with the path to your alarm sound file
pygame.mixer.music.load(alarm_sound_file)

def play_alarm():
    pygame.mixer.music.play()

class VideoWorker(QThread):
    frame_processed = pyqtSignal(np.ndarray, int)

    def __init__(self, parent=None):
        super(VideoWorker, self).__init__(parent)
        self.yolo = Load_Yolo_model()
        self.max_cosine_distance = 0.7
        self.nn_budget = None
        self.model_filename = resource_path('model_data\\mars-small128.pb')
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)
        self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        self.output_width = 640
        self.output_height = 480
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('output_video.mp4', self.fourcc, 20.0, (self.output_width, self.output_height))
        self.is_running = False

        # Skip frames to improve performance
        self.frame_skip = 12
        self.frame_count = 0

    def run(self):
        self.is_running = True
        while self.is_running:
            _, frame = self.cap.read()
            if frame is not None:
                # Skip frames
                if self.frame_count % self.frame_skip == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    original_frame = frame.copy()
                    image_data = image_preprocess(np.copy(original_frame), [YOLO_INPUT_SIZE, YOLO_INPUT_SIZE])
                    image_data = image_data[np.newaxis, ...].astype(np.float32)

                    if YOLO_FRAMEWORK == "tf":
                        pred_bbox = self.yolo.predict(image_data)
                    elif YOLO_FRAMEWORK == "trt":
                        batched_input = tf.constant(image_data)
                        result = self.yolo(batched_input)
                        pred_bbox = [value.numpy() for value in result.values()]

                    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
                    pred_bbox = tf.concat(pred_bbox, axis=0)

                    bboxes = postprocess_boxes(pred_bbox, original_frame, YOLO_INPUT_SIZE, 0.3)
                    bboxes = nms(bboxes, 0.45, method='nms')

                    person_count = 0

                    for bbox in bboxes:
                        if read_class_names(YOLO_COCO_CLASSES)[int(bbox[5])] == "person":
                            boxes = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                            scores = bbox[4]
                            names = read_class_names(YOLO_COCO_CLASSES)[int(bbox[5])]

                            if names == "person":
                                cv2.rectangle(original_frame, (int(boxes[0]), int(boxes[1])),
                                              (int(boxes[0] + boxes[2]), int(boxes[1] + boxes[3])), (0, 255, 0), 2)
                                person_count += 1

                    self.out.write(original_frame)

                    self.frame_processed.emit(original_frame, person_count)

                    if person_count > 0:
                        play_alarm()

                self.frame_count += 1
                if self.frame_count > self.frame_skip:
                    self.frame_count = 0

    def stop(self):
        self.is_running = False
        self.wait()

class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super(VideoWidget, self).__init__(parent)
        self.video_label = QLabel(self)
        self.video_label.setMinimumSize(750, 550)
        self.count_label = QLabel(self)
        self.count_label.setFont(QFont('Arial', 16))

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.count_label, alignment=Qt.AlignTop | Qt.AlignHCenter)
        self.layout.addWidget(self.video_label)
        self.setLayout(self.layout)

        self.worker = VideoWorker()
        self.worker.frame_processed.connect(self.update_frame)
        self.worker.start()

    def update_frame(self, frame, person_count):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
        self.count_label.setText(f"Person Count: {person_count}")

    def closeEvent(self, event):
        self.worker.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = QWidget()
    video_widget = VideoWidget(mainWin)
    mainWin.setGeometry(100, 100, 800, 600)
    mainWin.show()
    sys.exit(app.exec_())