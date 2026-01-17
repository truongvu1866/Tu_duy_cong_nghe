import cv2
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtGui import QImage, QPixmap

from UI.mainui import Ui_Form
from db_code.faissdb import FaceDatabase
from process.camera_thread import CameraThread


class MainWindow(QtWidgets.QMainWindow, Ui_Form):
    msg_signal = QtCore.pyqtSignal(str)
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.db = FaceDatabase()
        self.thread = None
        self.steps = [self.OpenCam, self.Detect, self.Embedding, self.Query, self.Draw]

        self.msg_signal.connect(self.log_message)
        self.startbutton.clicked.connect(self.start_camera)
        self.stopbutton.clicked.connect(self.stop_camera)
        self.radio1.toggled.connect(self.change_mode)
        self.radio2.toggled.connect(self.change_mode)
        self.Nextstepbutton.clicked.connect(self.trigger_next)

    def start_camera(self):
        if self.thread is None:
            self.log_message("Đang khởi động hệ thống...")
            self.log_message("Đang tái cấu trúc Database, vui lòng đợi...")
            new_status = self.db.load()
            if self.radio1.isChecked():
                mode = "EACH"
            else:
                mode = "REAL_TIME"
            self.log_message(new_status)
            self.thread = CameraThread(self.db, "EACH")
            try:
                self.thread.notification.connect(self.log_message)
                self.thread.step_signal.connect(self.set_progress_step)
            except:
                print('Khong the ket su dung Notetext')
            self.thread.frame_ready.connect(self.update_frame)
            self.thread.start()

            self.log_message(f"Hệ thống bắt đầu ở chế độ: {mode}")

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

    def log_message(self, text):
        """Hàm này sẽ thêm dòng mới vào QTextBrowser"""
        self.Notext.append(f"> {text}")
        # Tự động cuộn xuống cuối cùng
        self.Notext.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def set_progress_step(self, step_index):
        """
        step_index: 0, 1, 2...
        """
        for i, label in enumerate(self.steps):
            if i == step_index:
                # Ô đang thực hiện: Màu xanh dương
                label.setStyleSheet("""
                    background-color: #3498db; 
                    color: white; 
                    border: 2px solid #2980b9;
                    font-weight: bold;
                """)
            elif i < step_index:
                # Ô đã hoàn thành: Màu xanh lá
                label.setStyleSheet("""
                    background-color: #2ecc71; 
                    color: white; 
                    border: 2px solid #27ae60;
                """)
            else:
                # Ô chưa tới: Màu xám
                label.setStyleSheet("""
                    background-color: #d1d1d1; 
                    color: #555555;
                    border: 2px solid #a1a1a1;
                """)

    def change_mode(self):
        if self.thread and self.thread.isRunning():
            if self.radio1.isChecked():
                new_mode = "EACH"
            else :
                new_mode = "REAL_TIME"

            self.thread.setMode(new_mode)
            self.log_message(f"Đã chuyển sang chế độ ")

    def trigger_next(self):
        if self.thread:
            self.thread.next_step()