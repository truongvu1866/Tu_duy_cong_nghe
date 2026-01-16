import cv2
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from process.camera_thread import CameraThread


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition")
        self.resize(800, 600)

        self.video_label = QLabel("Camera chưa chạy")
        self.video_label.setFixedSize(640, 480)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        self.setLayout(layout)

        self.thread = None

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)

    def start_camera(self):
        if self.thread is None:
            self.thread = CameraThread()
            self.thread.frame_ready.connect(self.update_frame)
            self.thread.start()

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
            self.video_label.clear()

    def update_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(
            frame.data, w, h, ch * w, QImage.Format.Format_RGB888
        )
        self.video_label.setPixmap(QPixmap.fromImage(qimg))
