import os, cv2, numpy as np
import time
from dotenv import load_dotenv

from queue import Queue
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from mysql.connector import pooling,Error
from ultralytics import YOLO

from .face_align import align_face
from db_code.faissdb import FaceDatabase

import mysql

load_dotenv()

try:
    # Khởi tạo Connection Pool
    db_pool = pooling.MySQLConnectionPool(
        pool_name="my_pool",
        pool_size=5, # Số lượng kết nối tối đa được giữ sẵn
        pool_reset_session=True, # Tự động làm sạch dữ liệu cũ khi lấy kết nối ra
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    print("Khởi tạo Connection Pool thành công!")
except Error as e:
    print(f"Lỗi khi tạo Pool: {e}")

def get_connection():
    # Lấy một kết nối có sẵn từ pool (Cực nhanh)
    return db_pool.get_connection()

class CameraThread(QThread):
    notification = pyqtSignal(str)
    frame_ready = pyqtSignal(np.ndarray)
    step_signal = pyqtSignal(int)
    delete_done_signal = pyqtSignal(bool, str)

    def __init__(self, db: FaceDatabase, mode, in_mode):
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

        self.embedder = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
        self.delete_queue = Queue()

        self.model = YOLO(YOLO_PATH)
        self.size_process = 416
        self.conf_threshold = 0.5
        self.THRESHOLD = 0.6

        self.db = db

        self.in_mode = in_mode
        self.mode = mode
        self.current_frame = None
        self.is_waiting_next = False
        self.unknown_counter = 0
        self.user_id_to_delete = None  # Biến tạm lưu ID cần xóa

        self.detect_interval = 13
        self.running = True
        self.frame_id = 0
        self.last_results = []
        self.last_id = '0'

        self.prev_time = time.time()
        self.fps = 0.0
        self.frame_count = 0
        self.fps_update_interval = 0.5  # giây

    def stop(self):
        self.running = False
        self.condition.wakeAll()
        if self.cap.isOpened():
            self.cap.release()

    def run(self):
        self.load_and_process_db_queue()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:break
            if self.mode == "REAL_TIME" :
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
                    self.draw_fps(frame)
                except:
                    print("An error when drawing frame")
                self.frame_ready.emit(frame)
            elif self.mode == "EACH":
                self.each_process()
            if not self.running:break
        self.cap.release()

    def each_process(self):
        try:
            # --- BƯỚC 0: PREVIEW & CAPTURE ---
            self.step_signal.emit(0)
            self.is_waiting_next = True

            while self.is_waiting_next and self.running:
                # Cập nhật ảnh liên tục từ camera
                ret, frame = self.cap.read()
                if not ret:break
                if frame is not None:
                    self.current_frame = frame
                    self.frame_ready.emit(frame)
                self.msleep(30)  # Delay nhẹ để không quá tải CPU (khoảng 30fps)

            # Sau khi thoát vòng lặp while trên (do nhấn Next),
            # self.current_frame sẽ giữ nguyên tấm ảnh cuối cùng được chụp.
            if not self.running: return

            # --- STEP 1: Detect ---
            if not self.wait_for_user(1):return
            face_box = self.detect_face(frame)

            # --- STEP 2: Embedding ---
            if not self.wait_for_user(2): return
            vector = self.get_embedding(face_box)

            # --- STEP 3: Query Database ---
            if not self.wait_for_user(3):return
            self.last_results = self.query_db(vector)
            for x1, y1, x2, y2, user_id, score in self.last_results:
                self.notification.emit(f"user_id: {user_id}")

            # --- STEP 4: Draw Result ---
            if not self.wait_for_user(4):return
            if self.running and self.current_frame is not None:
                final_img = self.draw_result(self.current_frame)
                self.frame_ready.emit(final_img)
            if not self.wait_for_user(4): return
        except Exception as e:
            print("Lỗi hệ thống: {str(e)}")

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
                self.unknown_counter = 0
                if user_id != self.last_id:
                    self.last_id = user_id
                    self.query_mysql_db(self.last_id)
            else:
                self.unknown_counter += 1
            results.append((x1, y1, x2, y2, user_id, score))
        return results

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

    def next_step(self):
        if self.is_waiting_next:
            self.is_waiting_next = False
        self.mutex.lock()
        self.condition.wakeAll()
        self.mutex.unlock()

    def wait_for_user(self, current_step):
        if not self.running:
            return False
        self.step_signal.emit(current_step)  # Thông báo UI đổi màu ô vuông
        self.mutex.lock()
        self.condition.wait(self.mutex)  # Dừng Thread tại đây cho đến khi wakeAll()
        self.mutex.unlock()
        return self.running

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
            if user_id != "Unknown":
                if user_id != self.last_id:
                    self.unknown_counter = 0
                    self.last_id = user_id
                    try:
                        self.query_mysql_db(self.last_id)
                    except mysql.connector.Error as err:
                        print("Lỗi khi chạy truy xuất mysql database" + str(err))
            else:
                self.unknown_counter = +1
        return results

    def query_mysql_db(self, user_id):
        if self.in_mode:
            self.insert_box_user(user_id)
        else:
            self.delete_box_user(user_id)

    def draw_result(self, image):
        if image is None or len(self.last_results) == 0:
            return image
        annotated_image = image.copy()
        try:
            for x1, y1, x2, y2, user_id, score in self.last_results:
                # Ép kiểu tọa độ về int để OpenCV không lỗi
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cv2.rectangle(annotated_image, p1, p2, (0, 255, 0), 2)
                label = f"{user_id}: {score:.2f}"
                cv2.putText(annotated_image, label, (p1[0], p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return annotated_image
        except Exception as e:
            print(f"Lỗi vẽ hình: {e}")
            return image

    '''MYSQL DATABASE QUERY FORM HERE'''

    def load_and_process_db_queue(self):
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
              # Dùng dictionary để dễ truy cập theo tên cột
            query = "SELECT id, box_id FROM delete_queue WHERE status = 'In' ORDER BY id ASC"
            cursor.execute(query)
            rows = cursor.fetchall()
            if not rows:
                self.notification.emit("Không có dữ liệu cũ trong hàng đợi DB.")
                return
            count = 0
            for row in rows:
                db_id = row['id']
                box_id = row['box_id']
                self.delete_queue.put(box_id)
                count += 1
            conn.commit()
            self.notification.emit(f"Đã nạp lại {count} yêu cầu từ Database.")
        except mysql.connector.Error as err:
            self.notification.emit(f"Lỗi DB khi nạp hàng đợi: {err}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

    def insert_box_user(self, user_id):
        connection = get_connection()
        cursor = connection.cursor()
        try:
            query_data = "SELECT * FROM boxes_users WHERE user_id = %s"
            cursor.execute(query_data, (user_id,))
            data = cursor.fetchone()
        except mysql.connector.Error as e:
            print("Xuất hiện lỗi khi truy vấn dữ liệu" + str(e))
            data = None
        try:
            if data is None:
                target_box_id  = self.delete_queue.get()
                if target_box_id:
                    update_queue = "UPDATE delete_queue SET status = %s WHERE box_id = %s"
                    cursor.execute(update_queue, ('Not_in', target_box_id))
                    connection.commit()
                    insert_new_user = "INSERT INTO boxes_users (user_id, box_number) VALUES (%s, %s)"
                    cursor.execute(insert_new_user, (user_id,target_box_id))
                    connection.commit()
                    self.notification.emit(f"{user_id} check in thành công")
                    self.notification.emit(f"{user_id}: tủ {target_box_id}")
        except mysql.connector.Error as err:
            print("Lỗi khi thêm người mới " + str(err))
        finally:
            cursor.close()
            connection.close()

    def delete_box_user(self, user_id):
        connection = get_connection()
        cursor = connection.cursor()
        try:
            query_data = "SELECT box_number FROM boxes_users WHERE user_id = %s"
            cursor.execute(query_data, (user_id,))
            result_query = cursor.fetchone()
            if result_query:
                box_target = result_query[0]
                update_queue = "UPDATE delete_queue SET status = %s WHERE box_id = %s"
                cursor.execute(update_queue, ('In', box_target))
                connection.commit()
                self.delete_queue.put(box_target)
                query_delete = "DELETE FROM boxes_users WHERE user_id = %s"
                cursor.execute(query_delete, (user_id,))
                connection.commit()
                self.notification.emit(f"{user_id} check out thành công")
                self.notification.emit(f"{user_id}: tủ {box_target}")
        except mysql.connector.Error as err:
            print(err)
        finally:
            cursor.close()
            connection.close()
