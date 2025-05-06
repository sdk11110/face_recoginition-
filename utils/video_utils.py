import cv2
import time
import logging

def generate_frames(face_recognizer=None):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logging.error("无法打开摄像头")
        return
        
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            # 添加水平翻转
            frame = cv2.flip(frame, 1)  # 1表示水平翻转
            
            if face_recognizer:
                # 检测人脸
                faces = face_recognizer.detect_faces(frame)
                
                # 在检测到的人脸周围画框
                for face in faces:
                    left = face.left()
                    top = face.top()
                    right = face.right()
                    bottom = face.bottom()
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
            # 将图像编码为JPEG格式
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)  # 控制帧率
            
    except Exception as e:
        logging.error(f"视频流错误: {str(e)}")
    finally:
        camera.release() 