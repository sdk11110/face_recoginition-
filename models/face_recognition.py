import cv2
import numpy as np
import dlib
import threading
from datetime import datetime
import logging
from db_config import get_db_connection, logger

class FaceRecognizer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FaceRecognizer, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
            self.face_rec_model = dlib.face_recognition_model_v1('data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
            self._initialized = True

    def detect_faces(self, frame):
        return self.detector(frame, 1)

    def get_face_encoding(self, frame, face):
        shape = self.shape_predictor(frame, face)
        face_descriptor = self.face_rec_model.compute_face_descriptor(frame, shape)
        return np.array(face_descriptor)

    def recognize_face(self, frame):
        try:
            faces = self.detect_faces(frame)
            if not faces:
                return None

            face = faces[0]
            face_encoding = self.get_face_encoding(frame, face)

            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT u.name, k.face_encoding 
                FROM known_faces k
                JOIN users u ON k.user_id = u.id
                WHERE k.face_encoding IS NOT NULL
            """)
            known_faces = cursor.fetchall()

            if not known_faces:
                return None

            best_match = None
            best_distance = float('inf')

            for name, encoding_blob in known_faces:
                known_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
                distance = np.linalg.norm(face_encoding - known_encoding)

                if distance < best_distance:
                    best_distance = distance
                    best_match = name

            cursor.close()
            conn.close()

            if best_distance <= 0.4:  # 使用0.4作为阈值
                return best_match
            return None

        except Exception as e:
            logger.error(f"人脸识别失败: {str(e)}")
            return None

# 创建全局实例
face_recognizer = FaceRecognizer() 