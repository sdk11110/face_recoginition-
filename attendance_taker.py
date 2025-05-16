import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import datetime
import mysql.connector
import torch

# MySQL配置
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',  # 确保这是你的MySQL密码
    'database': 'attendance_db'
}

# Dlib  / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

def get_db_connection():
    return mysql.connector.connect(**db_config)

logger = logging.getLogger(__name__)

class Face_Recognizer:
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device('cpu')
        self.detector = dlib.get_frontal_face_detector()
        
        # 加载模型文件
        model_path = 'data/data_dlib/shape_predictor_68_face_landmarks.dat'
        face_rec_model_path = 'data/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
        
        self.shape_predictor = dlib.shape_predictor(model_path)
        self.face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
        
        # 初始化已知人脸列表
        self.known_face_encodings = []
        self.known_face_names = []
        
        logger.info(f"人脸识别器初始化完成，使用设备: {self.device}")
        
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        #  Save the features of faces in the database
        self.face_features_known_list = []
        # / Save the name of faces in the database
        self.face_name_known_list = []

        #  List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        #  cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        #  Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        #  Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    def add_known_face(self, name, face_encoding):
        """添加已知人脸"""
        # 将人脸编码转换为tensor并移动到GPU（如果可用）
        if isinstance(face_encoding, np.ndarray):
            face_encoding = torch.from_numpy(face_encoding).to(self.device)
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)

    def clear_known_faces(self):
        """清空已知人脸数据"""
        self.known_face_encodings = []
        self.known_face_names = []

    def recognize_face(self, image):
        """识别单个人脸"""
        try:
            # 转换图像到RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            # 检测人脸
            faces = self.detector(image, 1)
            
            if len(faces) == 0:
                return None
                
            # 获取第一个人脸的特征
            shape = self.shape_predictor(image, faces[0])
            face_encoding = np.array(self.face_rec_model.compute_face_descriptor(image, shape))
            
            # 转换为tensor并移动到GPU
            face_encoding = torch.from_numpy(face_encoding).to(self.device)
            
            if not self.known_face_encodings:
                return None
                
            # 计算与所有已知人脸的距离
            distances = []
            for known_encoding in self.known_face_encodings:
                if isinstance(known_encoding, np.ndarray):
                    known_encoding = torch.from_numpy(known_encoding).to(self.device)
                dist = torch.norm(face_encoding - known_encoding).item()
                distances.append(dist)
                
            # 找到最小距离的索引
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            
            # 如果最小距离小于阈值，返回对应的名字
            if min_dist < 0.4:  # 可以调整这个阈值
                return self.known_face_names[min_dist_idx]
            
            return None
            
        except Exception as e:
            logger.error(f"人脸识别过程出错: {str(e)}")
            return None

    def recognize_faces(self, image):
        """识别图像中的所有人脸"""
        try:
            # 转换图像到RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            # 检测人脸
            faces = self.detector(image, 1)
            face_locations = []
            face_names = []
            face_distances = []
            
            # 处理每个检测到的人脸
            for face in faces:
                # 获取人脸特征
                shape = self.shape_predictor(image, face)
                face_encoding = np.array(self.face_rec_model.compute_face_descriptor(image, shape))
                face_encoding = torch.from_numpy(face_encoding).to(self.device)
                
                # 计算与所有已知人脸的距离
                if self.known_face_encodings:
                    distances = []
                    for known_encoding in self.known_face_encodings:
                        if isinstance(known_encoding, np.ndarray):
                            known_encoding = torch.from_numpy(known_encoding).to(self.device)
                        dist = torch.norm(face_encoding - known_encoding).item()
                        distances.append(dist)
                        
                    min_dist_idx = np.argmin(distances)
                    min_dist = distances[min_dist_idx]
                    
                    if min_dist < 0.4:  # 阈值
                        name = self.known_face_names[min_dist_idx]
                    else:
                        name = "未知"
                else:
                    name = "未知"
                    min_dist = 1.0
                
                face_locations.append((face.top(), face.right(), face.bottom(), face.left()))
                face_names.append(name)
                face_distances.append(min_dist)
                
            return face_locations, face_names, face_distances
            
        except Exception as e:
            logger.error(f"批量人脸识别过程出错: {str(e)}")
            return [], [], []

    #  "features_all.csv"  / Get known faces from database
    def get_face_database(self):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name, face_encoding FROM known_faces")
            known_faces = cursor.fetchall()
            
            for name, encoding in known_faces:
                self.face_name_known_list.append(name)
                # 确保使用正确的数据类型读取二进制数据
                face_encoding = np.frombuffer(encoding, dtype=np.float64)
                
                # 验证数据完整性
                if len(face_encoding) != 128:
                    logging.error(f"人脸特征数据长度错误: {name}")
                    continue
                    
                # 检查数据是否有效
                if np.all(face_encoding == 0) or np.any(np.isnan(face_encoding)):
                    logging.error(f"人脸特征数据无效: {name}")
                    continue
                    
                # 确保数据类型正确
                face_encoding = face_encoding.astype(np.float64)
                self.face_features_known_list.append(face_encoding)
                logging.info(f"成功加载人脸数据: {name}")
            
            logging.info("从数据库加载人脸数据：%d 个", len(self.face_features_known_list))
            
            # 如果没有加载到任何人脸数据，返回失败
            if not self.face_features_known_list:
                logging.error("没有从数据库加载到任何有效的人脸数据")
                return 0
                
            return 1
        except Exception as e:
            logging.error(f"从数据库加载人脸数据时出错: {str(e)}")
            return 0
        finally:
            cursor.close()
            conn.close()

    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            #  For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    #  cv2 window / putText on cv2 window
    def draw_note(self, img_rd):
        #  / Add some info on windows
        cv2.putText(img_rd, "Face Recognizer with Deep Learning", (20, 40), self.font, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "ESC: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font,
                                 0.8, (255, 190, 0),
                                 1,
                                 cv2.LINE_AA)

    # insert data in database

    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # 检查今日是否已打卡
            cursor.execute(
                "SELECT COUNT(*) FROM attendance WHERE name = %s AND DATE(check_in_time) = %s",
                (name, current_date)
            )
            if cursor.fetchone()[0] > 0:
                print(f"{name} 今日已打卡")
                return
            
            # 插入打卡记录
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                "INSERT INTO attendance (name, check_in_time) VALUES (%s, %s)",
                (name, current_time)
            )
            conn.commit()
            print(f"{name} 打卡成功，时间：{current_time}")
            
        except Exception as e:
            print(f"打卡失败：{str(e)}")
        finally:
            cursor.close()
            conn.close()

    #  Face detection and recognition wit OT from input video stream
    def process(self, stream):
        # 1.  Get faces known from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                # 添加水平翻转
                img_rd = cv2.flip(img_rd, 1)  # 1表示水平翻转
                
                # 检查ESC键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键的ASCII码是27
                    logging.info("用户按下ESC键，程序退出")
                    break
                    
                # 2.  Detect faces for frame X
                faces = detector(img_rd, 0)

                # 3.  Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4.  Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5.  update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1  if cnt not changes
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1:   No face cnt changes in this frame!!!")

                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1

                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            img_rd = cv2.rectangle(img_rd,
                                                   tuple([d.left(), d.top()]),
                                                   tuple([d.right(), d.bottom()]),
                                                   (255, 255, 255), 2)

                    #  Multi-faces in current frame, use centroid-tracker to track
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        # 6.2 Write names under ROI
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                             self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                             cv2.LINE_AA)
                    self.draw_note(img_rd)

                # 6.2  If cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    logging.debug("scene 2: / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    # 6.2.1  Face cnt decreases: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  / No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                    # 6.2.2 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.debug("  scene 2.2  Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        # 6.2.2.1 Traversal all the faces in the database
                        for k in range(len(faces)):
                            logging.debug("  For face %d in current frame:", k + 1)
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []

                            # 6.2.2.2  Positions of faces captured
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 6.2.2.3
                            # For every faces detected, compare the faces in the database
                            for i in range(len(self.face_features_known_list)):
                                #
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i])
                                    logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    #  person_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # 6.2.2.4 / Find the one with minimum e distance
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                logging.debug("  Face recognition result: %s",
                                              self.face_name_known_list[similar_person_num])

                                # Insert attendance record
                                nam = self.face_name_known_list[similar_person_num]

                                print(type(self.face_name_known_list[similar_person_num]))
                                print(nam)
                                self.attendance(nam)
                            else:
                                logging.debug("  Face recognition result: Unknown person")

                        # 7.  / Add note on cv2 window
                        self.draw_note(img_rd)

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

                logging.debug("Frame ends\n\n")

    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0)  # Get video stream from camera
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()

def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
