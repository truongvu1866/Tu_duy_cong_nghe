import os, cv2, numpy as np
import time

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from ultralytics import YOLO

from .face_align import align_face
from db_code.faissdb import FaceDatabase

class CameraThread(QThread):
    notification = pyqtSignal(str)
    frame_ready = pyqtSignal(np.ndarray)
    step_signal = pyqtSignal(int)

    def __init__(self, db: FaceDatabase, mode):
        super().__init__()

        self.mutex = QMutex()
        self.condition = QWaitCondition()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Đường dẫn tới các thư mục
        MODELS_DIR = os.path.join(BASE_DIR, "models")
        DB_DIR = os.path.join(BASE_DIR, "faiss_db")

        # Đảm bảo thư mục lưu DB tồn tại
        os.makedirs(DB_DIR, exist_ok=True)

        # Đường dẫn file model cụ thể
        YOLO_PATH = os.path.join(MODELS_DIR, "yolov8n-face.pt")
        SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

        # Đường dẫn file database đầu ra
        DB_INDEX_FILE = os.path.join(DB_DIR, "face_db.index")
        DB_NAMES_FILE = os.path.join(DB_DIR, "names.pkl")

        self.embedder = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

        self.model = YOLO(YOLO_PATH)
        self.size_process = 416
        self.conf_threshold = 0.5
        self.THRESHOLD = 0.6

        self.db = db

        self.mode = mode

        self.detect_interval = 13
        self.running = True
        self.frame_id = 0
        self.last_results = []
        self.last_id = '0'

        self.prev_time = time.time()
        self.fps = 0.0
        self.frame_count = 0
        self.fps_update_interval = 0.5  # giây

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:break
            if self.mode == "REAL_TIME" :
                self.real_time(frame)
            elif self.mode == "EACH":
                self.each_process()
            if not self.running:break

        self.cap.release()

    def real_time(self, frame):
        self.frame_count += 1
        if self.frame_id % self.detect_interval == 0:
            try:
                self.last_results = self.process(frame)
            except:
                print("An error in processing frame")
        if self.frame_id == 60:
            self.frame_id = 0
        self.update_fps()
        try:
            self.draw(frame)
            self.draw_fps(frame)
        except:
            print("An error when drawing frame")

    def each_process(self):
        # --- STEP 0: Get Image ---
        if not self.wait_for_user(0): return
        self.notification.emit(f"Bắt đầu bước lấy ảnh")
        frame = self.capture_image()  # Hàm lấy ảnh từ ESP32
        self.notification.emit(f"Lấy ảnh thành công")
        self.frame_ready.emit(frame)

        # --- STEP 1: Detect ---
        if not self.wait_for_user(1):return
        self.notification.emit(f"Bắt đầu detect ảnh")
        face_box = self.detect_face(frame)
        self.notification.emit(f"Đã hoàn thành detect ảnh")

        # --- STEP 2: Embedding ---
        if not self.wait_for_user(2): return
        self.notification.emit(f"Bắt đầu embedding mặt")
        vector = self.get_embedding(face_box)
        self.notification.emit(f"Hoàn thành embedding")

        # --- STEP 3: Query Database ---
        if not self.wait_for_user(3):return
        self.notification.emit(f"Bắt đầu truy vấn cơ sở dữ liệu")
        self.last_results = self.query_db(vector)
        self.notification.emit(f"Đã hoàn thành truy vấn cơ sở dữ liệu")
        for x1, y1, x2, y2, user_id, score in self.last_results:
            self.notification.emit(f"user_id: {user_id}")

        # --- STEP 4: Draw Result ---
        if not self.wait_for_user(4):return
        self.notification.emit(f"Bắt dầu vẽ kết quả")
        try:
            self.draw(frame)
        except:
            print("An error when draw info")
        self.frame_ready.emit(f"Đã hoàn tất quá trình")

    def update_fps(self):
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.prev_time

        if elapsed >= self.fps_update_interval:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.prev_time = now

    def draw(self, frame):
        for x1,y1,x2,y2,user_id,score in self.last_results:
            color = (0, 255, 0) if user_id != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{user_id} {score:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

    def draw_fps(self, frame):
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    def stop(self):
        self.running = False
        self.condition.wakeAll()
        if self.cap.isOpened():
            self.cap.release()

    def process(self, image):
        h, w = image.shape[:2]

        img = cv2.resize(image, (self.size_process, self.size_process))
        r = self.model(
            img,
            conf=self.conf_threshold,
            device="cpu",
            verbose=False
        )[0]

        if r.boxes is None:
            return np.empty((0, 128), dtype=np.float32)

        sx, sy = w / self.size_process, h / self.size_process
        boxes = r.boxes.xyxy.cpu().numpy()
        kpts = r.keypoints.xy.cpu().numpy()

        results= []

        for box, lm in zip(boxes, kpts):
            x1, y1, x2, y2 = (box * [sx, sy, sx, sy]).astype(int)

            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue

            lm = [(int(x * sx), int(y * sy)) for x, y in lm]
            lm_local = [(x - x1, y - y1) for x, y in lm]

            aligned = align_face(face, lm_local)
            if aligned is None:
                continue
            emb = self.embedder.feature(aligned)
            flatten = emb.flatten()  # Flatten để đảm bảo là vector 1 chiều
            user_id, score = self.db.search(flatten)
            if user_id != "Unknown":
                if user_id != self.last_id:
                    try:
                        self.notification.emit(f"Xin chào {user_id}")
                    except:
                        print("khong the gui thong bao ve Notetext")
                    self.last_id = user_id
            results.append((x1, y1, x2, y2, user_id, score))

        return results

    def next_step(self):
        self.condition.wakeAll()

    def wait_for_user(self, current_step):
        if not self.running:
            return
        self.step_signal.emit(current_step)  # Thông báo UI đổi màu ô vuông
        self.mutex.lock()
        self.condition.wait(self.mutex)  # Dừng Thread tại đây cho đến khi wakeAll()
        self.mutex.unlock()
        return self.running

    def capture_image(self):
        """Lấy khung hình mới nhất từ Webcam"""
        if not self.cap.isOpened():
            self.cap.open(0)

        # Giải phóng các khung hình cũ trong buffer để lấy ảnh thực tế tại thời điểm nhấn nút
        # Webcam thường có buffer 3-5 khung hình, ta sẽ 'đọc bỏ' chúng
        for _ in range(5):
            self.cap.grab()

        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def detect_face(self, image):
        h, w = image.shape[:2]

        img = cv2.resize(image, (self.size_process, self.size_process))
        r = self.model(
            img,
            conf=self.conf_threshold,
            device="cpu",
            verbose=False
        )[0]

        if r.boxes is None:
            return np.empty((0, 128), dtype=np.float32)

        sx, sy = w / self.size_process, h / self.size_process
        boxes = r.boxes.xyxy.cpu().numpy()
        kpts = r.keypoints.xy.cpu().numpy()

        results = []

        for box, lm in zip(boxes, kpts):
            x1, y1, x2, y2 = (box * [sx, sy, sx, sy]).astype(int)

            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue

            lm = [(int(x * sx), int(y * sy)) for x, y in lm]
            lm_local = [(x - x1, y - y1) for x, y in lm]

            aligned = align_face(face, lm_local)
            results.append((x1, y1, x2, y2, aligned))
        return results

    def get_embedding(self, face):
        results = []
        for x1, y1, x2, y2, aligned in face:
            emb = self.embedder.feature(aligned)
            flatten = emb.flatten()  # Flatten để đảm bảo là vector 1 chiều
            results.append((x1, y1, x2, y2, flatten))
        return results

    def query_db(self, vector):
        results = []
        for x1, y1, x2, y2, flatten in vector:
            user_id, score = self.db.search(flatten)
            results.append((x1, y1, x2, y2, user_id, score))
        return results

