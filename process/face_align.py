import cv2
import numpy as np


def align_face(image, landmarks):
    """
    Căn chỉnh khuôn mặt dựa trên 5 điểm landmarks (Mắt trái, phải, mũi, miệng trái, phải).
    Output: Ảnh 112x112 chuẩn cho SFace.
    """
    # Tọa độ chuẩn của 5 điểm landmarks trên ảnh 112x112 (Standard ArcFace/SFace crop)
    REFERENCE_LANDMARKS = np.array([
        [38.2946, 51.6963],  # Mắt trái
        [73.5318, 51.5014],  # Mắt phải
        [56.0252, 71.7366],  # Mũi
        [41.5493, 92.3655],  # Khóe miệng trái
        [70.7299, 92.2041]  # Khóe miệng phải
    ], dtype=np.float32)

    if landmarks is None or len(landmarks) != 5:
        return None

    # Tính toán ma trận biến đổi (Affine Transform)
    # landmarks đầu vào phải là np.array shape (5, 2)
    st = cv2.estimateAffinePartial2D(np.array(landmarks, dtype=np.float32), REFERENCE_LANDMARKS)[0]

    # Thực hiện cắt và xoay ảnh
    aligned_face = cv2.warpAffine(image, st, (112, 112))
    return aligned_face
