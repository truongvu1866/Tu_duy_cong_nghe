import os
import cv2
import numpy as np
from process.mainprocess import Processer


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def embed_and_save(
    root_dir: str,
    output_txt: str,
    processer: Processer
):
    with open(output_txt, "w", encoding="utf-8") as f_out:

        for student_id in os.listdir(root_dir):
            user_folder = os.path.join(root_dir, student_id)
            if not os.path.isdir(user_folder):
                continue

            image_files = [
                f for f in os.listdir(user_folder)
                if f.lower().endswith(".jpg")
            ]

            if not image_files:
                print(f"[WARN] {student_id} không có ảnh")
                continue

            for img_name in image_files:
                img_path = os.path.join(user_folder, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"[WARN] Không đọc được {img_path}")
                    continue

                embedding = processer.process(img)
                if embedding is None:
                    print(f"[WARN] Không detect được mặt: {img_path}")
                    continue

                embedding = embedding.astype(np.float32).reshape(-1)
                embedding = normalize(embedding)

                emb_str = ",".join(f"{x:.6f}" for x in embedding)

                line = f"{student_id}|{img_name}|{emb_str}\n"
                f_out.write(line)

                print(f"[OK] {student_id} - {img_name}")


    print(f"\n[DONE] Đã lưu embedding vào {output_txt}")

    def aggregate_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
        """
        embeddings: list các embedding shape (512,) hoặc (1,512)
        return: vector đại diện shape (512,)
        """
        if len(embeddings) == 0:
            raise ValueError("Không có embedding để gộp")

        emb = np.vstack([e.reshape(-1) for e in embeddings]).astype(np.float32)

        mean_emb = np.mean(emb, axis=0)

        norm = np.linalg.norm(mean_emb)
        if norm == 0:
            raise ValueError("Mean embedding có norm = 0")

        return mean_emb / norm
if __name__ == "__main__":
    processer = Processer()  # model nhận diện mặt của bạn
    embed_and_save(
        root_dir="data",
        output_txt="result.txt",
        processer=processer
    )
