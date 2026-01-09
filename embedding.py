import cv2
import numpy as np
import onnxruntime as ort

class FaceEmbedding:
    def __init__(self, model_path, num_threads=4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
        face = np.transpose(face, (2, 0, 1))
        return np.expand_dims(face, 0)

    def get_embedding(self, face):
        inp = self.preprocess(face)
        emb = self.session.run(None, {self.input_name: inp})[0][0]
        emb = emb / np.linalg.norm(emb)
        return emb
