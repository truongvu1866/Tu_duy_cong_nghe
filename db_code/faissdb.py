import faiss
import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "faiss_db")
os.makedirs(DB_DIR, exist_ok=True)
DB_INDEX_FILE = os.path.join(DB_DIR, "face_db.index")
DB_NAMES_FILE = os.path.join(DB_DIR, "names.pkl")

class FaceDatabase:
    def __init__(self, dimension=128, database_path=DB_INDEX_FILE, names_path=DB_NAMES_FILE):
        self.dimension = dimension
        self.database_path = database_path
        self.names_path = names_path

        # 1. Tạo Index (Sử dụng Inner Product để tính Cosine Similarity sau khi chuẩn hóa)
        self.index = faiss.IndexFlatIP(self.dimension)

        # 2. Danh sách lưu tên tương ứng với ID trong Index
        self.names = []

    def add_face(self, vector, name):
        """Thêm một khuôn mặt mới vào DB"""
        # Chuẩn hóa vector về độ dài 1 (Unit vector) để Inner Product = Cosine Similarity
        vector = vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(vector)

        # Thêm vào index
        self.index.add(vector)
        self.names.append(name)
        print(f"Đã thêm {name} vào database.")

    def search(self, vector, threshold=0.363):
        """Tìm kiếm khuôn mặt gần nhất"""
        if self.index.ntotal == 0:
            return "Unknown", 0

        # Chuẩn hóa vector đầu vào
        vector = vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(vector)

        # Tìm kiếm k=1 (người giống nhất)
        # D: Khoảng cách (độ tương đồng), I: Index (ID)
        D, I = self.index.search(vector, 1)

        score = D[0][0]
        idx = I[0][0]

        if idx != -1 and score >= threshold:
            return self.names[idx], score
        return "Unknown", score

    def save(self):
        """Lưu database ra file"""
        faiss.write_index(self.index, self.database_path)
        with open(self.names_path, "wb") as f:
            pickle.dump(self.names, f)
        print("Đã lưu database.")

    def load(self):
        """Load database từ file"""
        try:
            self.index = faiss.read_index(self.database_path)
            with open(self.names_path, "rb") as f:
                self.names = pickle.load(f)
            return f"Đã load {len(self.names)} khuôn mặt."
        except:
            return f"Chưa có database, khởi tạo mới."