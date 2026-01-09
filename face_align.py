import cv2
import numpy as np

ARC_SIZE = 112

ARC_LANDMARK = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face(image, landmarks, output_size=ARC_SIZE):
    if landmarks is None or len(landmarks) != 5:
        return None

    src = np.array(landmarks, dtype=np.float32)
    dst = ARC_LANDMARK.copy()

    if output_size != ARC_SIZE:
        dst *= output_size / ARC_SIZE

    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None

    aligned = cv2.warpAffine(
        image, M, (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderValue=0
    )
    return aligned
