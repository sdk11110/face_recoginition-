from face_recognition_evaluator import FaceRecognitionEvaluator

def run_evaluation():
    """
    运行人脸识别系统评估
    """
    print("开始人脸识别系统评估...")
    
    # 创建评估器实例，指定测试数据集路径
    # 如果路径不同，请修改为您的实际路径
    evaluator = FaceRecognitionEvaluator(test_dataset_path="data/test_faces")
    
    # 运行评估
    print("正在评估人脸识别系统性能，请稍候...")
    results = evaluator.evaluate()
    
    if results:
        # 生成评估报告
        print("评估完成，正在生成报告...")
        report = evaluator.generate_report()
        
        print("\n===== 评估结果摘要 =====")
        print(f"最佳阈值: {report['summary']['best_threshold']:.2f}")
        print(f"最佳F1-Score: {report['summary']['best_metrics']['f1_score']:.4f}")
        print(f"准确率: {report['summary']['best_metrics']['accuracy']:.4f}")
        print(f"拒识率(FRR): {report['summary']['best_metrics']['frr']:.4f}")
        print(f"误识率(FAR): {report['summary']['best_metrics']['far']:.4f}")
        print(f"处理时间: {report['summary']['test_time']:.2f}秒")
        print(f"报告已保存至: {report['report_dir']}")
    else:
        print("评估未能完成，请检查日志获取详细信息")

if __name__ == "__main__":
    run_evaluation() 