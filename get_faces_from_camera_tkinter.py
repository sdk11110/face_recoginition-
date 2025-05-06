import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk
from tkinter import ttk

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()


class Face_Register:
    def __init__(self):

        self.current_frame_faces_cnt = 0  # cnt for counting faces in current frame
        self.existing_faces_cnt = 0  # cnt for counting saved faces
        self.ss_cnt = 0  # cnt for screen shots

        # Tkinter GUI
        self.win = tk.Tk()
        self.win.title("Face Register (按ESC退出)")

        # PLease modify window size here if needed
        self.win.geometry("1000x700")

        # GUI left part
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        # GUI right part
        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces_cnt))
        self.label_fps_info = tk.Label(self.frame_right_info, text="")
        self.input_name = tk.Entry(self.frame_right_info)
        self.input_password = tk.Entry(self.frame_right_info, show="*")  # 添加密码输入框
        self.input_name_char = ""
        self.input_password_char = ""  # 添加密码变量
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        # 使用相对路径
        self.path_photos_from_camera = "data/data_faces_from_camera"
        self.current_face_dir = ""
        self.font = cv2.FONT_ITALIC

        # Current frame and face ROI position
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.cap = cv2.VideoCapture(0)  # Get video stream from camera

        # self.cap = cv2.VideoCapture("test.mp4")   # Input local video

        # 确保基础目录存在
        os.makedirs(self.path_photos_from_camera, exist_ok=True)

        # 添加下拉列表变量
        self.folder_var = tk.StringVar()

    #  Delete old face folders
    def GUI_clear_data(self):
        try:
            # 1. 清除本地文件夹
            if os.path.exists(self.path_photos_from_camera):
                folders_rd = os.listdir(self.path_photos_from_camera)
                for folder in folders_rd:
                    folder_path = os.path.join(self.path_photos_from_camera, folder)
                    if os.path.isdir(folder_path):
                        shutil.rmtree(folder_path)
                        logging.info(f"已删除文件夹: {folder_path}")

            # 2. 删除features_all.csv文件
            if os.path.isfile("data/features_all.csv"):
                os.remove("data/features_all.csv")
                logging.info("已删除features_all.csv文件")

            # 3. 尝试清除数据库中的数据
            try:
                from db_config import get_db_connection
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    
                    # 清除人脸数据
                    cursor.execute("DELETE FROM known_faces")
                    logging.info("已清除数据库中的人脸数据")
                    
                    # 清除考勤记录
                    cursor.execute("DELETE FROM attendance")
                    logging.info("已清除数据库中的考勤记录")
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
                else:
                    logging.warning("无法连接到数据库，仅清除了本地文件")
            except ImportError:
                logging.warning("未找到数据库配置文件，仅清除了本地文件")
            except Exception as db_err:
                logging.warning(f"数据库操作出错: {db_err}，仅清除了本地文件")

            self.label_cnt_face_in_database['text'] = "0"
            self.existing_faces_cnt = 0
            self.log_all["text"] = "所有本地文件已清除!"

        except Exception as e:
            logging.error(f"清除数据时出错: {str(e)}")
            self.log_all["text"] = "清除数据时出错，请检查日志!"

    def GUI_get_input_name(self):
        # 每次创建新文件夹时重新检查现有的最大编号
        self.check_existing_faces_cnt()
        self.input_name_char = self.input_name.get().strip()
        self.input_password_char = self.input_password.get().strip()
        
        if not self.input_name_char:
            self.log_all["text"] = "请输入姓名!"
            return
            
        if not self.input_password_char:
            self.log_all["text"] = "请输入密码!"
            return
            
        if len(self.input_password_char) < 6:
            self.log_all["text"] = "密码长度至少6位!"
            return
            
        # 检查用户名是否已存在
        try:
            from db_config import get_db_connection
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users WHERE username = %s", (self.input_name_char,))
                if cursor.fetchone()[0] > 0:
                    self.log_all["text"] = "该用户名已存在!"
                    cursor.close()
                    conn.close()
                    return
                cursor.close()
                conn.close()
        except Exception as e:
            print(f"检查用户名时出错: {str(e)}")
            
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)

    def GUI_info(self):
        tk.Label(self.frame_right_info,
                 text="人脸注册",
                 font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=20)

        # FPS显示
        tk.Label(self.frame_right_info, text="FPS: ").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info, text="数据库中的人脸数量: ").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info,
                 text="当前画面中的人脸数: ").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=2, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # 步骤1：清除数据选项
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="步骤1: 清除人脸数据").grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        
        # 添加文件夹选择下拉列表
        self.folder_var = tk.StringVar()
        self.folder_list = ttk.Combobox(self.frame_right_info, textvariable=self.folder_var, width=30)
        self.folder_list.grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.update_folder_list()
        
        # 添加刷新和删除按钮
        tk.Button(self.frame_right_info,
                  text='刷新列表',
                  command=self.update_folder_list).grid(row=6, column=2, sticky=tk.W, padx=5, pady=2)
        
        tk.Button(self.frame_right_info,
                  text='删除选中',
                  command=self.delete_selected_folder).grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
        tk.Button(self.frame_right_info,
                  text='删除全部',
                  command=self.GUI_clear_data).grid(row=7, column=1, sticky=tk.W, padx=5, pady=2)

        # 步骤2：输入用户信息
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="步骤2: 输入用户信息").grid(row=8, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Label(self.frame_right_info, text="用户名: ").grid(row=9, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_name.grid(row=9, column=1, sticky=tk.W, padx=0, pady=2)
        
        tk.Label(self.frame_right_info, text="密码: ").grid(row=10, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_password.grid(row=10, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Button(self.frame_right_info,
                  text='确认',
                  command=self.GUI_get_input_name).grid(row=10, column=2, padx=5)

        # 步骤3：保存人脸图像
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="步骤3: 采集人脸图像").grid(row=11, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Button(self.frame_right_info,
                  text='保存当前人脸',
                  command=self.save_current_face).grid(row=12, column=0, columnspan=3, sticky=tk.W)

        # 显示日志信息
        self.log_all.grid(row=13, column=0, columnspan=20, sticky=tk.W, padx=5, pady=20)

        self.frame_right_info.pack()

    # Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    # Start from person_x+1
    def check_existing_faces_cnt(self):
        if os.path.exists(self.path_photos_from_camera):
            # 获取所有文件夹
            person_list = [f for f in os.listdir(self.path_photos_from_camera) 
                         if os.path.isdir(os.path.join(self.path_photos_from_camera, f))]
            
            # 从文件夹名称中提取所有已使用的编号
            used_numbers = set()
            for person in person_list:
                try:
                    if person.startswith('person_'):
                        # 提取编号部分（例如从 'person_05_name' 提取 '05'）
                        number_str = person.split('_')[1]
                        number = int(number_str)
                        used_numbers.add(number)
                except (IndexError, ValueError):
                    continue
            
            # 从1开始查找第一个未使用的编号
            self.existing_faces_cnt = 1
            while self.existing_faces_cnt in used_numbers:
                self.existing_faces_cnt += 1
        else:
            # 如果文件夹不存在，从1开始
            self.existing_faces_cnt = 1
        
        logging.info(f"下一个可用编号: {self.existing_faces_cnt:02d}")

    # Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        #  Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

        self.label_fps_info["text"] = str(self.fps.__round__(2))

    def create_face_folder(self):
        if not self.input_name_char:
            self.log_all["text"] = "请输入姓名!"
            return
            
        # 直接使用用户名作为文件夹名
        folder_name = f"person_{self.input_name_char.strip()}"
        self.current_face_dir = os.path.join(self.path_photos_from_camera, folder_name)

        # 如果文件夹已存在，提示错误
        if os.path.exists(self.current_face_dir):
            self.log_all["text"] = "该用户名已存在!"
            self.face_folder_created_flag = False
            return

        # 创建新目录
        os.makedirs(self.current_face_dir, exist_ok=True)
        self.log_all["text"] = f"\"{self.current_face_dir}/\" created!"
        logging.info("\n%-40s %s", "Create folders:", self.current_face_dir)

        self.ss_cnt = 0  # Clear the cnt of screen shots
        self.face_folder_created_flag = True  # Face folder already created

    def save_current_face(self):
        if not self.face_folder_created_flag:
            self.log_all["text"] = "请先输入用户信息并点击确认!"
            return

        if self.current_frame_faces_cnt != 1:
            self.log_all["text"] = f"检测到 {self.current_frame_faces_cnt} 张人脸，请确保画面中只有一张人脸!"
            return

        if self.out_of_range_flag:
            self.log_all["text"] = "人脸位置超出范围，请调整位置!"
            return

        try:
            self.ss_cnt += 1
            
            # 保存人脸图像
            self.face_ROI_image = np.zeros((int(self.face_ROI_height * 2), self.face_ROI_width * 2, 3),
                                           np.uint8)
            for ii in range(self.face_ROI_height * 2):
                for jj in range(self.face_ROI_width * 2):
                    self.face_ROI_image[ii][jj] = self.current_frame[self.face_ROI_height_start - self.hh + ii][
                        self.face_ROI_width_start - self.ww + jj]

            if not os.path.exists(self.current_face_dir):
                os.makedirs(self.current_face_dir)
                
            path_save_img = os.path.join(self.current_face_dir, f"img_face_{self.ss_cnt}.jpg")
            cv2.imwrite(path_save_img, self.face_ROI_image)

            # 保存用户信息到数据库
            if self.ss_cnt == 1:  # 只在第一次保存时添加用户信息
                try:
                    from db_config import get_db_connection
                    conn = get_db_connection()
                    if conn:
                        cursor = conn.cursor()
                        
                        # 插入新用户
                        cursor.execute("""
                            INSERT INTO users (username, password, name, role)
                            VALUES (%s, %s, %s, %s)
                        """, (self.input_name_char, self.input_password_char, self.input_name_char, 'user'))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        self.log_all["text"] = f"已保存第 {self.ss_cnt} 张人脸图像，并创建用户账号"
                except Exception as e:
                    self.log_all["text"] = f"保存用户信息失败: {str(e)}"
                    print(f"保存用户信息时出错: {str(e)}")
            else:
                self.log_all["text"] = f"已保存第 {self.ss_cnt} 张人脸图像"

        except Exception as e:
            self.log_all["text"] = f"保存图像失败: {str(e)}"
            print(f"保存图像时出错: {str(e)}")

    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                # 添加水平翻转，使其像镜子一样
                frame = cv2.flip(frame, 1)  # 1表示水平翻转
                frame = cv2.resize(frame, (640, 480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error: No video input!!!")

    #  Main process of face detection and saving
    def process(self):
        ret, self.current_frame = self.get_frame()
        faces = detector(self.current_frame, 0)
        # Get frame
        if ret:
            self.update_fps()
            self.label_face_cnt["text"] = str(len(faces))
            #  Face detected
            if len(faces) != 0:
                #   Show the ROI of faces
                for k, d in enumerate(faces):
                    self.face_ROI_width_start = d.left()
                    self.face_ROI_height_start = d.top()
                    #  Compute the size of rectangle box
                    self.face_ROI_height = (d.bottom() - d.top())
                    self.face_ROI_width = (d.right() - d.left())
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)

                    # If the size of ROI > 480x640
                    if (d.right() + self.ww) > 640 or (d.bottom() + self.hh > 480) or (d.left() - self.ww < 0) or (
                            d.top() - self.hh < 0):
                        self.label_warning["text"] = "OUT OF RANGE"
                        self.label_warning['fg'] = 'red'
                        self.out_of_range_flag = True
                        color_rectangle = (255, 0, 0)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = ""
                        color_rectangle = (255, 255, 255)
                    self.current_frame = cv2.rectangle(self.current_frame,
                                                       tuple([d.left() - self.ww, d.top() - self.hh]),
                                                       tuple([d.right() + self.ww, d.bottom() + self.hh]),
                                                       color_rectangle, 2)
            self.current_frame_faces_cnt = len(faces)

            # Convert PIL.Image.Image to PIL.Image.PhotoImage
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)

        # Refresh frame
        self.win.after(20, self.process)

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.GUI_info()
        self.process()
        self.win.mainloop()

    def update_folder_list(self):
        # 更新文件夹下拉列表
        if os.path.exists(self.path_photos_from_camera):
            folders = [f for f in os.listdir(self.path_photos_from_camera) 
                      if os.path.isdir(os.path.join(self.path_photos_from_camera, f))]
            self.folder_list['values'] = folders
            if folders:
                self.folder_list.set(folders[0])
        else:
            self.folder_list['values'] = []
            self.folder_list.set('')

    def delete_selected_folder(self):
        selected_folder = self.folder_var.get()
        if not selected_folder:
            self.log_all["text"] = "请先选择要删除的文件夹!"
            return

        try:
            # 删除选中的文件夹
            folder_path = os.path.join(self.path_photos_from_camera, selected_folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                logging.info(f"已删除文件夹: {folder_path}")

            # 从数据库中删除对应的人脸数据
            if '_' in selected_folder:
                parts = selected_folder.split('_')
                if len(parts) >= 3:
                    person_name = '_'.join(parts[2:])
                    try:
                        from db_config import get_db_connection
                        conn = get_db_connection()
                        if conn:
                            cursor = conn.cursor()
                            
                            # 删除人脸数据
                            cursor.execute("DELETE FROM known_faces WHERE name = %s", (person_name,))
                            # 删除考勤记录
                            cursor.execute("DELETE FROM attendance WHERE name = %s", (person_name,))
                            
                            conn.commit()
                            cursor.close()
                            conn.close()
                            logging.info(f"已删除数据库中 {person_name} 的数据")
                        else:
                            logging.warning("无法连接到数据库，仅删除了本地文件")
                    except ImportError:
                        logging.warning("未找到数据库配置文件，仅删除了本地文件")
                    except Exception as db_err:
                        logging.warning(f"数据库操作出错: {db_err}，仅删除了本地文件")

            self.update_folder_list()  # 更新文件夹列表
            self.log_all["text"] = f"已删除 {selected_folder} 的所有数据!"

        except Exception as e:
            logging.error(f"删除数据时出错: {str(e)}")
            self.log_all["text"] = "删除数据时出错，请检查日志!"


def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()


if __name__ == '__main__':
    main()
