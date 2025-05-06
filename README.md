# 基于人脸识别的考勤系统

本项目是一个使用Python、OpenCV和Flask开发的人脸识别考勤系统。系统通过摄像头捕获人脸图像，与数据库中存储的人脸特征进行比对，自动完成考勤记录。

## 技术栈

- **后端**：Python、Flask
- **数据库**：MySQL
- **人脸识别**：OpenCV、dlib、face_recognition
- **前端**：HTML、CSS、JavaScript、Bootstrap

## 功能特点

- 人脸采集与注册
- 实时人脸识别考勤
- 考勤记录管理与统计
- 用户管理（管理员/普通用户）
- 数据可视化展示
- Web界面操作

## 安装步骤

1. 克隆项目到本地：
   ```
   git clone https://github.com/Arijit1080/Face-Recognition-Based-Attendance-System.git
   ```

2. 安装所需依赖包：
   ```
   pip install -r requirements.txt
   ```

3. 下载dlib模型文件并放置到正确位置：
   从 https://drive.google.com/drive/folders/12It2jeNQOxwStBxtagL1vvIJokoz-DL4?usp=sharing 下载模型文件，并将data文件夹放置在项目根目录下。

4. 配置MySQL数据库：
   - 创建数据库：`attendance_db`
   - 导入数据库结构：`mysql -u username -p attendance_db < database.sql`
   - 修改`db_config.py`中的数据库连接参数

## 使用方法

1. 采集人脸数据集：
   ```
   python get_faces_from_camera_tkinter.py
   ```
   通过此步骤拍摄并保存用户人脸图像。

2. 处理人脸特征：
   ```
   python face_features_processor.py
   ```
   提取人脸特征向量并存储到数据库。

3. 启动考勤功能：
   ```
   python attendance_taker.py
   ```
   开启摄像头进行实时人脸识别考勤。

4. 启动Web管理界面：
   ```
   python app.py
   ```
   通过浏览器访问`http://localhost:5000`进入系统管理界面。

## 系统架构

系统采用MVC架构设计：
- **Model层**：处理数据和业务逻辑，包括数据库操作和人脸识别算法
- **View层**：负责用户界面展示，包括HTML模板和静态资源
- **Controller层**：处理用户请求，连接Model和View层

## 贡献指南

欢迎贡献代码或提出建议！您可以：
- 提交Pull Request
- 创建Issue报告bug或提出新功能建议
- 完善文档


# face_recoginition-
