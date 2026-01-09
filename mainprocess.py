import cv2, os, numpy as np

from pathlib import Path

from .embedding import FaceEmbedding
from .face_align import align_face

from ultralytics import YOLO
class Processer:
    def __init__(self):
        MODEL_YOLO_PATH = Path(__file__).resolve().parent / "yolov8n-face.pt"
        self.model = YOLO(str(MODEL_YOLO_PATH))
        self.size_process = 416
        self.conf_threshold = 0.5

        MODEL_ARC_PATH = Path(__file__).resolve().parent / "arcface_ir_se50.onnx"
        self.embedder = FaceEmbedding(
            str(MODEL_ARC_PATH),
            num_threads=max(2, os.cpu_count()),
        )

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

        embeddings = []

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
            embeddings.append(emb)

        if len(embeddings) == 0:
            return np.empty((0, 512), dtype=np.float32)

        return np.vstack(embeddings)