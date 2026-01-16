import os, cv2, numpy as np
import time

from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

from .face_align import align_face
from db_code.faissdb import FaceDatabase

class CameraThread(QThread):
    notification = pyqtSignal(str)
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, db: FaceDatabase):
        super().__init__()

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
            if not ret:
                break

            self.frame_id += 1

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
            except:
                print("An error when drawing frame")

            self.frame_ready.emit(frame)

        self.cap.release()

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
            color = (0, 255, 0) if user_id != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{user_id} {score:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

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
        self.wait()

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