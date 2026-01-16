import cv2
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtGui import QImage, QPixmap

from db_code.faissdb import FaceDatabase
from process.camera_thread import CameraThread
from .mainui import Ui_Form  # Import class giao diện của bạn


class MainController(QtWidgets.QWidget, Ui_Form):
    msg_signal = QtCore.pyqtSignal(str)
    def __init__(self):
        super().__init__()
        # 1. Thiết lập giao diện từ file maingui
        self.setupUi(self)

        # 2. Khởi tạo các biến bổ sung
        self.db = FaceDatabase()
        self.thread = None


        # 3. Kết nối lại các tín hiệu (Signals) nếu maingui.py chưa làm chuẩn
        # Lưu ý: Bỏ ngoặc () để truyền hàm chứ không phải thực thi hàm
        self.msg_signal.connect(self.log_message)
        self.startbutton.clicked.connect(self.start_camera)

        self.stopbutton.clicked.connect(self.stop_camera)

    # Ghi đè hoặc bổ sung các hàm xử lý logic
    def start_camera(self):
        if self.thread is None:
            self.log_message("Đang khởi động hệ thống...")
            self.log_message("Đang tái cấu trúc Database, vui lòng đợi...")
            new_status = self.db.load()
            self.log_message(new_status)
            self.thread = CameraThread(self.db)
            try:
                self.thread.notification.connect(self.log_message)
            except:
                print('Khong the ket su dung Notetext')
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

    def log_message(self, text):
        """Hàm này sẽ thêm dòng mới vào QTextBrowser"""
        self.Notext.append(f"> {text}")
        # Tự động cuộn xuống cuối cùng
        self.Notext.moveCursor(QtGui.QTextCursor.MoveOperation.End)