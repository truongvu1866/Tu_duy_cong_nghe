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
        self.running = False
        self.stopbutton.setEnabled(False)
        self.InButton.setEnabled(False)
        self.steps = [self.OpenCam, self.Detect, self.Embedding, self.Query, self.Draw]
        self.in_mode = True

        self.msg_signal.connect(self.log_message)
        self.startbutton.clicked.connect(self.start_button_logic)
        self.stopbutton.clicked.connect(self.stop_camera)
        self.radio1.toggled.connect(self.change_mode)
        self.radio2.toggled.connect(self.change_mode)
        self.InButton.clicked.connect(self.change_in_mode)
        self.OutButton.clicked.connect(self.change_in_mode)

    def change_in_mode(self):
        self.in_mode = not self.in_mode
        self.in_out_change()
        if self.in_mode:
            self.log_message(f"change into in mode complete")
        else:
            self.log_message(f"change in out mode complete")

    def start_button_logic(self):
        if not self.running:
            self.start_process()
        else:
            self.trigger_next()

    def start_process(self):
        self.stopbutton.setEnabled(True)
        self.InButton.setEnabled(False)
        self.OutButton.setEnabled(False)
        if self.radio1.isChecked():
            self.running = True

            # Thay đổi giao diện nút
            self.startbutton.setText("NEXT STEP")
            self.startbutton.setStyleSheet("background-color: #3498db; color: white; font-weight: bold")
            self.stopbutton.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")

            # Vô hiệu hóa vùng chọn chế độ
            self.radio2.setEnabled(False)
            self.radio1.setEnabled(False)

            # Khởi chạy CameraThread
            self.start_camera()  # Hàm khởi tạo thread của bạn
        else:
            # Nếu là Realtime thì chỉ khởi chạy bình thường, không đổi tên nút thành Next
            self.start_camera()
            self.startbutton.setEnabled(False)

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
            self.thread = CameraThread(self.db, mode, self.in_mode)
            try:
                self.thread.notification.connect(self.log_message)
                self.thread.step_signal.connect(self.set_progress_step)
            except:
                print('Khong the ket su dung Notetext')
            self.thread.frame_ready.connect(self.update_frame)
            self.thread.start()
            self.log_message(f"In mode = {self.thread.in_mode}")
            self.log_message(f"Hệ thống bắt đầu ở chế độ: {mode}")

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
            self.video_label.clear()
        self.log_message("Hệ thống đã dừng.")
        # Reset trạng thái nút về ban đầu
        self.running = False
        self.startbutton.setText("START")
        self.startbutton.setEnabled(True)
        self.startbutton.setStyleSheet("border:2px solid #8f8f91;border-radius: 5px;font-weight:bold;font-size:15")  # Reset màu về mặc định

        # Kích hoạt lại vùng chọn chế độ
        self.radio2.setEnabled(True)
        self.radio1.setEnabled(True)

        self.set_progress_step(0)  # Hàm reset màu các ô vuông bước
        self.stopbutton.setEnabled(False)
        self.stopbutton.setStyleSheet("border:2px solid #8f8f91;border-radius: 5px;font-weight:bold;font-size:15")
        self.in_out_change()

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

    def in_out_change(self):
        if self.in_mode:
            self.InButton.setEnabled(False)
            self.InButton.setStyleSheet("border: 2px solid #8f8f91 ; border-radius: 10px;background-color: #3498db;color: white;font-size: 15;font-weight: bold")
            self.OutButton.setEnabled(True)
            self.OutButton.setStyleSheet("border: 2px solid #8f8f91;border-radius: 10px;font-size: 15;font-weight: bold")
        else:
            self.InButton.setEnabled(True)
            self.InButton.setStyleSheet("border: 2px solid #8f8f91;border-radius: 10px;font-size: 15;font-weight: bold")
            self.OutButton.setEnabled(False)
            self.OutButton.setStyleSheet("border: 2px solid #8f8f91 ; border-radius: 10px;background-color: #3498db;color: white;font-size: 15;font-weight: bold")
