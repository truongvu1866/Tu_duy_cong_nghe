import os
import cv2
import numpy as np
import faiss
import pickle
from ultralytics import YOLO

# --- CẤU HÌNH ĐƯỜNG DẪN ĐỘNG ---
# Lấy đường dẫn của thư mục gốc (thư mục chứa db_code, data, models...)
# os.path.abspath(__file__) lấy đường dẫn file hiện tại
# .dirname 2 lần để nhảy từ db_code/ ra ngoài thư mục gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn tới các thư mục
DATA_DIR = os.path.join(BASE_DIR, "data")
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

# --- KHỞI TẠO MODEL ---
print(f"Loading YOLO from: {YOLO_PATH}")
yolo_model = YOLO(YOLO_PATH)

print(f"Loading SFace from: {SFACE_PATH}")
face_recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")


def align_face(img, kpts):
    """Căn chỉnh mặt dựa trên 5 điểm landmarks"""
    dst = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]
    ], dtype=np.float32)
    src = np.array(kpts, dtype=np.float32)
    tform = cv2.estimateAffinePartial2D(src, dst)[0]
    if tform is None: return None
    return cv2.warpAffine(img, tform, (112, 112))


def build():
    # Sử dụng dictionary để gom nhóm vector theo ID
    person_vectors = {}

    if not os.path.exists(DATA_DIR):
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu tại {DATA_DIR}")
        return

    for person_id in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person_id)
        if not os.path.isdir(person_path): continue

        # Khởi tạo danh sách vector cho người này nếu chưa có
        if person_id not in person_vectors:
            person_vectors[person_id] = []

        print(f"Đang xử lý dữ liệu cho: {person_id}")
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            results = yolo_model(img, conf=0.6, verbose=False)[0]

            if results.boxes is not None and len(results.boxes) > 0:
                kpts = results.keypoints.data[0].cpu().numpy()[:, :2]
                box = results.boxes.xyxy[0].cpu().numpy().astype(int)

                face_crop = img[max(0, box[1]):box[3], max(0, box[0]):box[2]]
                lm_local = kpts - [max(0, box[0]), max(0, box[1])]

                aligned = align_face(face_crop, lm_local)
                if aligned is not None:
                    feat = face_recognizer.feature(aligned)
                    # Lưu tạm vector (đã flatten) vào danh sách của người này
                    person_vectors[person_id].append(feat.flatten())

    # --- BẮT ĐẦU TÍNH TRUNG BÌNH ---
    final_embeddings = []
    final_names = []

    for person_id, vecs in person_vectors.items():
        if len(vecs) > 0:
            # Chuyển danh sách vector thành numpy array (N, 128)
            vecs_array = np.array(vecs).astype('float32')

            # Tính trung bình cộng theo hàng (mean) -> kết quả là (128,)
            mean_vec = np.mean(vecs_array, axis=0)

            # Chuẩn hóa L2 cho vector trung bình này (bắt buộc cho Cosine Similarity)
            faiss.normalize_L2(mean_vec.reshape(1, -1))

            final_embeddings.append(mean_vec)
            final_names.append(person_id)
            print(f"-> Đã tổng hợp {len(vecs)} ảnh của [{person_id}] thành 1 vector duy nhất.")

    if final_embeddings:
        final_embeddings = np.array(final_embeddings).astype('float32')

        # Khởi tạo FAISS Index
        index = faiss.IndexFlatIP(128)
        index.add(final_embeddings)

        # Lưu vào thư mục faiss_db
        faiss.write_index(index, DB_INDEX_FILE)
        with open(DB_NAMES_FILE, "wb") as f:
            pickle.dump(final_names, f)
        print(f"\nThành công! Database hiện có {len(final_names)} người dùng.")
        print(f"Lưu tại: {DB_DIR}")
    else:
        print("Không có dữ liệu khuôn mặt hợp lệ.")

if __name__ == "__main__":
    build()