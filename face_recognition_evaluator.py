import os
import cv2
import dlib
import numpy as np
import matplotlib
# 设置matplotlib使用Agg后端，避免中文字体问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 尝试设置中文字体
try:
    # 尝试使用系统自带的字体
    font_paths = [
        'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
        'C:/Windows/Fonts/simsun.ttc',  # 宋体
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Linux
        '/System/Library/Fonts/PingFang.ttc'  # macOS
    ]
    
    font_found = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            font_prop = FontProperties(fname=font_path)
            plt.rcParams['font.family'] = 'sans-serif'
            if "msyh" in font_path:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            elif "simsun" in font_path:
                plt.rcParams['font.sans-serif'] = ['SimSun']
            elif "DroidSans" in font_path:
                plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback']
            elif "PingFang" in font_path:
                plt.rcParams['font.sans-serif'] = ['PingFang SC']
            plt.rcParams['axes.unicode_minus'] = False
            font_found = True
            break
            
    if not font_found:
        # 如果找不到中文字体，则使用英文标签
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
except Exception as e:
    # 如果设置字体出错，使用默认字体
    print(f"设置字体时出错: {str(e)}")
    
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import face_recognition
import logging
import mysql.connector
try:
    from db_config import get_db_connection, connection_pool, logger
except ImportError:
    from db_config import get_db_connection, logger
    connection_pool = None
from tqdm import tqdm
import json
import time
from datetime import datetime

# 导入项目的人脸识别处理器
from face_recognition_processor import face_recognition_test_with_db

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognitionEvaluator:
    def __init__(self, test_dataset_path=None):
        """
        初始化人脸识别评估器
        
        参数:
            test_dataset_path: 测试数据集路径，若为None则使用默认路径
        """
        # 测试数据集路径
        self.test_dataset_path = test_dataset_path or "data/test_faces"
        
        # 标记是否创建了模拟数据
        self.mock_data_created = False
        self.create_mock_data_for_testing = False
        
        # 确保路径存在，如果不存在则尝试寻找其他可能的路径
        if not os.path.exists(self.test_dataset_path):
            possible_paths = [
                "data/test_faces",
                "./data/test_faces",
                "../data/test_faces",
                "../../data/test_faces"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.test_dataset_path = path
                    logger.info(f"使用找到的路径: {path}")
                    break
        
        logger.info(f"使用测试数据集路径: {self.test_dataset_path}")
        
        # 加载人脸检测和识别模型
        self.detector = dlib.get_frontal_face_detector()
        
        # 确保dlib模型文件存在
        shape_predictor_path = 'data/data_dlib/data_dlib/shape_predictor_68_face_landmarks.dat'
        face_rec_model_path = 'data/data_dlib/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
        
        # 如果模型文件路径不存在，尝试查找可能的路径
        if not os.path.exists(shape_predictor_path):
            possible_paths = [
                'data/data_dlib/shape_predictor_68_face_landmarks.dat',
                './data/data_dlib/shape_predictor_68_face_landmarks.dat',
                'shape_predictor_68_face_landmarks.dat',
                './shape_predictor_68_face_landmarks.dat'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    shape_predictor_path = path
                    logger.info(f"使用找到的shape_predictor路径: {path}")
                    break
                    
        if not os.path.exists(face_rec_model_path):
            possible_paths = [
                'data/data_dlib/dlib_face_recognition_resnet_model_v1.dat',
                './data/data_dlib/dlib_face_recognition_resnet_model_v1.dat',
                'dlib_face_recognition_resnet_model_v1.dat',
                './dlib_face_recognition_resnet_model_v1.dat'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    face_rec_model_path = path
                    logger.info(f"使用找到的face_rec_model路径: {path}")
                    break
        
        try:
            self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
            self.face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
            logger.info("成功加载dlib模型")
        except Exception as e:
            logger.error(f"加载dlib模型失败: {str(e)}")
            raise
        
        # 初始化结果存储
        self.thresholds = np.linspace(0.1, 1.0, 20)  # 测试的距离阈值范围
        self.results = {
            'thresholds': self.thresholds,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'frr': [],  # 拒识率
            'far': [],  # 误识率
            'test_time': None
        }
        
        # 加载已知人脸特征
        self.known_faces = self._load_known_faces()
        
    def _load_known_faces(self):
        """从数据库加载已知人脸特征"""
        conn = None
        cursor = None
        try:
            # 直接调用而不使用with语句
            conn = connection_pool.get_connection() if 'connection_pool' in globals() else mysql.connector.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                user=os.getenv('DB_USER', 'root'),
                password=os.getenv('DB_PASSWORD', '123456'),
                database=os.getenv('DB_NAME', 'attendance_db')
            )
            
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT u.id, u.name, kf.face_encoding 
                FROM known_faces kf
                JOIN users u ON kf.user_id = u.id
                WHERE kf.is_active = TRUE
            """)
            
            known_faces = cursor.fetchall()
            
            if known_faces:
                logger.info(f"成功从数据库加载 {len(known_faces)} 个已知人脸特征")
                return known_faces
            else:
                logger.warning("数据库中没有已知人脸特征，将使用测试文件夹中的一部分图片作为已知人脸")
                return self._create_mock_known_faces()
                
        except Exception as e:
            logger.error(f"从数据库加载已知人脸特征时出错: {str(e)}")
            logger.info("将使用测试文件夹中的一部分图片作为已知人脸")
            return self._create_mock_known_faces()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
            
    def _create_mock_known_faces(self):
        """
        创建模拟的known_faces数据，使用测试集的前5张图片作为已知人脸
        """
        mock_known_faces = []
        sdk_path = os.path.join(self.test_dataset_path, "sdk")
        
        if not os.path.exists(sdk_path):
            logger.error(f"无法创建模拟数据，路径不存在: {sdk_path}")
            return []
            
        # 获取所有图片文件
        image_files = [f for f in os.listdir(sdk_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 按照人名（文件名前缀，如1.jpg的前缀是1）进行分组
        person_groups = {}
        for img_file in image_files:
            # 从文件名中提取人名/ID
            person_name = os.path.splitext(img_file)[0]
            # 确保相同名字的图片分到同一组
            if person_name not in person_groups:
                person_groups[person_name] = []
            person_groups[person_name].append(img_file)
        
        logger.info(f"发现 {len(person_groups)} 个不同的人（标签）")
        
        # 选择前5个不同的人（或更少）作为已知人脸
        selected_persons = list(person_groups.keys())[:min(5, len(person_groups))]
        
        for person_idx, person_name in enumerate(selected_persons):
            # 为每个人选择第一张图片
            img_file = person_groups[person_name][0]
            img_path = os.path.join(sdk_path, img_file)
            
            # 提取人脸特征
            feature = self._extract_face_feature(img_path)
            if feature is not None:
                # 将numpy数组转换为bytes以模拟数据库中的存储方式
                encoding_bytes = feature.tobytes()
                mock_known_faces.append((person_idx + 1, person_name, encoding_bytes))
                logger.info(f"已添加模拟已知人脸: {person_name} (来自 {img_file})")
                
                # 把这张图片标记为已使用，以便在测试集中排除
                person_groups[person_name][0] = None
                
        logger.info(f"成功创建 {len(mock_known_faces)} 个模拟已知人脸特征")
        
        # 保存训练/测试图片的分离情况，以便后续处理
        self.training_images = set()
        for person in selected_persons:
            files = person_groups[person]
            if files and files[0] is None:  # 第一张图片已用于训练
                img_path = os.path.join(sdk_path, person_groups[person][0]) if person_groups[person][0] else None
                if img_path:
                    self.training_images.add(img_path)
        
        # 标记已创建模拟数据
        self.mock_data_created = True
        
        return mock_known_faces

    def _prepare_test_dataset(self):
        """准备测试数据集，遍历 data/test_faces 下所有子文件夹"""
        test_images = []
        test_labels = []
        base_dir = self.test_dataset_path
        for person_dir in os.listdir(base_dir):
            person_path = os.path.join(base_dir, person_dir)
            if not os.path.isdir(person_path) or person_dir == 'sdk_backup':  # 跳过备份目录
                continue
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(person_path, img_file))
                    test_labels.append(person_dir)  # 文件夹名作为标签
        logger.info(f"成功加载 {len(test_images)} 张测试图片，来自 {len(set(test_labels))} 个不同的标签")
        return test_images, test_labels
        
    def _extract_face_feature(self, image_path):
        """提取图像中的人脸特征"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"无法读取图像: {image_path}")
                return None
                
            # 转换为RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            faces = self.detector(rgb_img, 1)
            if not faces:
                logger.warning(f"未检测到人脸: {image_path}")
                return None
                
            # 获取第一个人脸的特征
            shape = self.shape_predictor(rgb_img, faces[0])
            face_descriptor = self.face_rec_model.compute_face_descriptor(rgb_img, shape)
            
            return np.array(face_descriptor)
            
        except Exception as e:
            logger.error(f"提取人脸特征时出错 {image_path}: {str(e)}")
            return None
            
    def _compute_metrics(self, y_true, y_pred, threshold):
        """计算识别性能指标"""
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # 计算性能指标
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算拒识率(FRR)和误识率(FAR)
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0  # 拒识率
        far = fp / (tn + fp) if (tn + fp) > 0 else 0  # 误识率
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'frr': frr,
            'far': far
        }
        
    def evaluate(self):
        """评估人脸识别系统性能"""
        start_time = time.time()
        
        # 准备测试数据
        test_images, test_labels = self._prepare_test_dataset()
        if not test_images:
            # 如果没有测试图片，但需要创建模拟数据用于测试
            if self.create_mock_data_for_testing:
                logger.info("使用测试集创建模拟已知人脸数据")
                self.known_faces = self._create_mock_known_faces()
                # 重新加载测试数据
                test_images, test_labels = self._prepare_test_dataset()
                if not test_images:
                    logger.error("没有找到测试图片，评估终止")
                    return
            else:
                logger.error("没有找到测试图片，评估终止")
                return
            
        # 使用face_recognition_processor中的函数进行评估
        logger.info("使用项目中的人脸识别处理器进行评估...")
        
        # 初始化结果变量
        y_true_all = []  # 真实标签 (1=应该识别为已知人, 0=应该拒识/陌生人)
        y_pred_all = []  # 预测结果 (1=识别为已知人, 0=拒识/未识别)
        y_scores_all = []  # 相似度分数
        
        # 获取known_faces格式兼容的结构，确保与face_recognition_test_with_db参数匹配
        # face_recognition_test_with_db期望的格式: [(user_id, name, encoding), ...]
        known_faces_list = self.known_faces  # 数据库加载的格式恰好符合要求
        
        # 获取数据库中的已知标签
        known_labels = set([name for _, name, _ in self.known_faces])
        logger.info(f"数据库中的已知人脸标签: {known_labels}")
        
        logger.info(f"开始处理 {len(test_images)} 张测试图片...")
        
        # 对每张测试图片进行识别
        for img_path, true_label in tqdm(zip(test_images, test_labels), total=len(test_images)):
            # 调用项目中的识别函数
            result = face_recognition_test_with_db(img_path, known_faces_list)
            
            # 处理识别结果
            if result['status'] == 'success':
                predicted_label = result['name']
                confidence = result['confidence'] / 100.0  # 转换为0-1范围
            else:
                predicted_label = None
                confidence = result.get('confidence', 0.0) / 100.0 if 'confidence' in result else 0.0  # 未识别出结果但可能有置信度
            
            # 判断匹配情况
            # 对于数据库中有的人（如sdk/szq等）：预测为本人算TP，预测为None算FN，预测为其他人算FP
            # 对于陌生人（stranger）：预测为None算TN，预测为数据库中任何人都算FP
            if true_label == 'stranger':
                # 对于stranger，预测为None才是正确的
                is_match = (predicted_label is None)
                # 真实值为0（陌生人），预测值根据是否识别为已知人决定
                y_true = 0  # 陌生人
                y_pred = 0 if predicted_label is None else 1  # 未识别为0，识别为1
            else:
                # 对于数据库中应有的人，预测标签与真实标签一致才是正确的
                is_match = (true_label == predicted_label)
                # 真实值为1（已知人），预测值根据是否识别为对应人决定
                y_true = 1  # 已知人
                y_pred = 1 if predicted_label == true_label else 0  # 识别正确为1，错误为0
                
            # 记录日志以便调试
            logger.info(f"图片: {img_path}, 真实标签: {true_label}, 预测标签: {predicted_label}, 匹配: {is_match}")
            
            # 存储真实值和预测值用于后续计算
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
            y_scores_all.append(confidence)
        
        # 如果没有进行任何有效的识别比较，终止评估
        if not y_true_all:
            logger.error("没有有效的识别比较，评估终止")
            return
            
        # 输出准确率，帮助调试
        correct_count = sum(1 for t, p in zip(y_true_all, y_pred_all) if t == p)
        total_count = len(y_true_all)
        logger.info(f"总体准确率: {correct_count}/{total_count}, {correct_count/total_count:.2%}")
        
        # 统计准确率明细，分别统计stranger和已知人的准确率
        stranger_count = sum(1 for t in y_true_all if t == 0)
        known_count = sum(1 for t in y_true_all if t == 1)
        stranger_correct = sum(1 for t, p in zip(y_true_all, y_pred_all) if t == 0 and p == 0)
        known_correct = sum(1 for t, p in zip(y_true_all, y_pred_all) if t == 1 and p == 1)
        
        if stranger_count > 0:
            logger.info(f"陌生人准确率: {stranger_correct}/{stranger_count}, {stranger_correct/stranger_count:.2%}")
        if known_count > 0:
            logger.info(f"已知人准确率: {known_correct}/{known_count}, {known_correct/known_count:.2%}")
        
        # 计算不同阈值下的性能指标
        for threshold in self.thresholds:
            # 基于阈值生成预测结果
            threshold_y_pred = []
            for i, score in enumerate(y_scores_all):
                # 对于已知人标签
                if y_true_all[i] == 1:
                    # 如果置信度大于等于阈值，则预测为已知人
                    threshold_y_pred.append(1 if score >= threshold else 0)
                else:
                    # 对于陌生人标签，如果置信度超过阈值，则误识别为已知人
                    # 否则正确拒识
                    threshold_y_pred.append(1 if score >= threshold else 0)
            
            # 计算性能指标
            metrics = self._compute_metrics(y_true_all, threshold_y_pred, threshold)
            
            # 更新结果
            self.results['accuracy'].append(metrics['accuracy'])
            self.results['precision'].append(metrics['precision'])
            self.results['recall'].append(metrics['recall'])
            self.results['f1_score'].append(metrics['f1_score'])
            self.results['frr'].append(metrics['frr'])
            self.results['far'].append(metrics['far'])
            
        self.results['test_time'] = time.time() - start_time
        
        # 找到最佳F1-Score对应的阈值
        best_idx = np.argmax(self.results['f1_score'])
        best_threshold = self.thresholds[best_idx]
        
        logger.info(f"评估完成! 耗时: {self.results['test_time']:.2f}秒")
        logger.info(f"最佳阈值: {best_threshold:.2f}, F1-Score: {self.results['f1_score'][best_idx]:.4f}")
        
        return self.results
        
    def plot_roc_curve(self, save_path=None):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))
        plt.plot(self.results['far'], 1 - np.array(self.results['frr']), 'b-', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        
        # 计算ROC曲线下面积
        roc_auc = auc(self.results['far'], 1 - np.array(self.results['frr']))
        
        plt.xlabel('False Acceptance Rate (FAR)', fontsize=14)
        plt.ylabel('1 - False Rejection Rate (1-FRR)', fontsize=14)
        plt.title(f'Face Recognition System ROC Curve (AUC = {roc_auc:.4f})', fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"ROC曲线已保存至: {save_path}")
            
        return plt.gcf()
        
    def plot_metrics(self, save_path=None):
        """绘制性能指标随阈值变化的曲线"""
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.thresholds, self.results['accuracy'], 'b-', linewidth=2)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy vs Threshold', fontsize=14)
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.thresholds, self.results['precision'], 'r-', linewidth=2)
        plt.plot(self.thresholds, self.results['recall'], 'g-', linewidth=2)
        plt.plot(self.thresholds, self.results['f1_score'], 'b-', linewidth=2)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Precision, Recall and F1-Score', fontsize=14)
        plt.legend(['Precision', 'Recall', 'F1-Score'])
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.thresholds, self.results['frr'], 'r-', linewidth=2)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('False Rejection Rate (FRR)', fontsize=12)
        plt.title('FRR vs Threshold', fontsize=14)
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.thresholds, self.results['far'], 'g-', linewidth=2)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('False Acceptance Rate (FAR)', fontsize=12)
        plt.title('FAR vs Threshold', fontsize=14)
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"性能指标图已保存至: {save_path}")
            
        return plt.gcf()
        
    def generate_report(self, output_dir="reports"):
        """生成评估报告，并创建递增编号的子文件夹"""
        # 确保根输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建递增编号的子文件夹
        existing_folders = []
        for folder_name in os.listdir(output_dir):
            folder_path = os.path.join(output_dir, folder_name)
            if os.path.isdir(folder_path) and folder_name.isdigit():
                existing_folders.append(int(folder_name))
                
        next_folder_num = 1
        if existing_folders:
            next_folder_num = max(existing_folders) + 1
            
        # 创建新的子文件夹
        report_dir = os.path.join(output_dir, str(next_folder_num))
        os.makedirs(report_dir, exist_ok=True)
        logger.info(f"创建报告子目录：{report_dir}")
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存性能指标图
        metrics_plot_path = os.path.join(report_dir, f"metrics_plot_{timestamp}.png")
        self.plot_metrics(metrics_plot_path)
        
        # 保存ROC曲线
        roc_plot_path = os.path.join(report_dir, f"roc_curve_{timestamp}.png")
        self.plot_roc_curve(roc_plot_path)
        
        # 创建性能指标表格
        metrics_df = pd.DataFrame({
            'Threshold': self.thresholds,
            'Accuracy': self.results['accuracy'],
            'Precision': self.results['precision'],
            'Recall': self.results['recall'],
            'F1-Score': self.results['f1_score'],
            'FRR': self.results['frr'],
            'FAR': self.results['far']
        })
        
        # 保存性能指标表格
        metrics_csv_path = os.path.join(report_dir, f"metrics_table_{timestamp}.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        
        # 找到最佳F1-Score对应的阈值
        best_idx = np.argmax(self.results['f1_score'])
        best_threshold = self.thresholds[best_idx]
        
        # 生成总结信息
        summary = {
            'timestamp': timestamp,
            'test_dataset': self.test_dataset_path,
            'known_faces_count': len(self.known_faces),
            'test_time': self.results['test_time'],
            'best_threshold': float(best_threshold),
            'best_metrics': {
                'accuracy': self.results['accuracy'][best_idx],
                'precision': self.results['precision'][best_idx],
                'recall': self.results['recall'][best_idx],
                'f1_score': self.results['f1_score'][best_idx],
                'frr': self.results['frr'][best_idx],
                'far': self.results['far'][best_idx]
            }
        }
        
        # 保存总结信息
        summary_path = os.path.join(report_dir, f"summary_{timestamp}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        
        logger.info(f"评估报告已生成，保存在目录: {report_dir}")
        return {
            'summary': summary,
            'metrics_table': metrics_csv_path,
            'metrics_plot': metrics_plot_path,
            'roc_curve': roc_plot_path,
            'report_dir': report_dir
        }

def main():
    print("启动人脸识别系统评估...")
    print("使用data/test_faces目录下的所有子文件夹进行测试")
    
    try:
        # 创建评估器实例
        evaluator = FaceRecognitionEvaluator()
        
        # 运行评估
        print("正在评估人脸识别系统性能，请稍候...")
        results = evaluator.evaluate()
        
        if results:
            # 生成报告
            print("评估完成，正在生成报告...")
            report = evaluator.generate_report()
            
            print("\n===== 评估结果摘要 =====")
            print(f"测试图片数量: {len(results.get('accuracy', []))}")
            print(f"最佳阈值: {report['summary']['best_threshold']:.2f}")
            print(f"最佳F1-Score: {report['summary']['best_metrics']['f1_score']:.4f}")
            print(f"准确率: {report['summary']['best_metrics']['accuracy']:.4f}")
            print(f"拒识率(FRR): {report['summary']['best_metrics']['frr']:.4f}")
            print(f"误识率(FAR): {report['summary']['best_metrics']['far']:.4f}")
            print(f"处理时间: {report['summary']['test_time']:.2f}秒")
            print(f"报告已保存至: {report['report_dir']}")
        else:
            print("评估未能完成，请检查日志获取详细信息")
    except Exception as e:
        print(f"评估过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 