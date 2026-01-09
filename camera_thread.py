import os, cv2, numpy as np
import time

from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
from ultralytics import YOLO

from .embedding import FaceEmbedding
from .face_align import align_face
from db_code.faissdb import FaceFaissDB

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        MODEL_YOLO_PATH = Path(__file__).resolve().parent / "yolov8n-face.pt"
        self.model = YOLO(str(MODEL_YOLO_PATH))
        self.size_process = 416
        self.conf_threshold = 0.5
        self.THRESHOLD = 0.6

        MODEL_ARC_PATH = Path(__file__).resolve().parent / "arcface_ir_se50.onnx"
        self.embedder = FaceEmbedding(
            str(MODEL_ARC_PATH),
            num_threads=max(2, os.cpu_count()),
        )

        self.last_user_id = '0'
        self.faiss_db = FaceFaissDB(dim=512)
        self.faiss_db.load("faiss_db")  #

        self.detect_interval = 13
        self.running = True
        self.frame_id = 0
        self.last_results = []

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
            return np.empty((0, 512), dtype=np.float32)

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
            emb = self.embedder.get_embedding(aligned)
            print("Embedding norm:", np.linalg.norm(emb))
            db_search = self.faiss_db.search(emb, top_k=3)
            if not db_search:
                user_id = "unknown"
                score = 0.0
            else:
                user_id, score = db_search[0]
                if score < self.THRESHOLD:
                    user_id = "unknown"
                else:
                    if user_id != self.last_user_id:
                        self.last_user_id = user_id
                        self.return_new_box()

            results.append((x1, y1, x2, y2, user_id, score))

        return results
    def return_new_box(self):
        print("returning new box")