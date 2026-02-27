# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
YOLOv8 目标检测完整示例

本脚本演示了使用 Ultralytics YOLOv8 进行目标检测的完整流程，包括：
1. 模型加载与初始化
2. 单张图像预测
3. 批量图像预测
4. 视频检测
5. 实时摄像头检测
6. 模型训练
7. 模型验证
8. 模型导出

使用方法：
    python object_detection_tutorial.py --mode predict --source path/to/image.jpg
    python object_detection_tutorial.py --mode train --data coco8.yaml
    python object_detection_tutorial.py --mode video --source path/to/video.mp4
    python object_detection_tutorial.py --mode webcam --source 0
"""

import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS, ROOT, SETTINGS


class ObjectDetector:
    """
    YOLOv8 目标检测器类
    
    该类封装了 YOLOv8 模型的各种功能，包括预测、训练、验证和导出。
    
    属性:
        model (YOLO): YOLOv8 模型实例
        device (str): 运行设备 ('cpu', 'cuda', 'mps' 等)
        
    方法:
        predict_image: 对单张图像进行目标检测
        predict_batch: 对多张图像进行批量检测
        predict_video: 对视频文件进行检测
        predict_webcam: 使用摄像头进行实时检测
        train: 训练模型
        validate: 验证模型性能
        export: 导出模型到其他格式
    """
    
    def __init__(self, model_path="yolov8n.pt", device=""):
        """
        初始化目标检测器
        
        参数:
            model_path (str): 模型权重文件路径，默认使用 YOLOv8n
            device (str): 运行设备，空字符串表示自动选择
        """
        print(f"正在加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
    def predict_image(self, source, conf=0.25, iou=0.7, save=True, show=False):
        """
        对单张或多张图像进行目标检测
        
        参数:
            source (str): 图像路径或图像文件夹路径
            conf (float): 置信度阈值
            iou (float): NMS IOU 阈值
            save (bool): 是否保存结果
            show (bool): 是否显示结果
            
        返回:
            results: 检测结果列表
        """
        print(f"\n开始图像检测: {source}")
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            show=show,
            device=self.device
        )
        
        for i, result in enumerate(results):
            print(f"\n图像 {i+1} 检测结果:")
            if result.boxes is not None:
                print(f"  检测到 {len(result.boxes)} 个目标")
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        cls_name = result.names[cls_id]
                        print(f"  - {cls_name}: {conf_score:.2f}")
            else:
                print(f"  未检测到目标（boxes 为 None）")
                    
        return results
    
    def predict_video(self, source, conf=0.25, iou=0.7, save=True, show=False):
        """
        对视频文件进行目标检测
        
        参数:
            source (str): 视频文件路径
            conf (float): 置信度阈值
            iou (float): NMS IOU 阈值
            save (bool): 是否保存结果视频
            show (bool): 是否实时显示结果
        """
        print(f"\n开始视频检测: {source}")
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            show=show,
            stream=True,
            device=self.device
        )
        
        frame_count = 0
        for result in results:
            frame_count += 1
            if frame_count % 30 == 0:
                num_boxes = len(result.boxes) if result.boxes is not None else 0
                print(f"已处理 {frame_count} 帧, 当前帧检测到 {num_boxes} 个目标")
                
        print(f"\n视频检测完成，共处理 {frame_count} 帧")
    
    def predict_webcam(self, source=0, conf=0.25, iou=0.7):
        """
        使用摄像头进行实时目标检测
        
        参数:
            source (int): 摄像头编号，默认 0 表示默认摄像头
            conf (float): 置信度阈值
            iou (float): NMS IOU 阈值
        """
        print(f"\n启动摄像头检测 (按 'q' 键退出)")
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            show=True,
            stream=True,
            device=self.device
        )
        
        for result in results:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        print("\n摄像头检测结束")
    
    def train(self, data, epochs=100, imgsz=640, batch=16, project=None, name="exp"):
        """
        训练 YOLOv8 模型
        
        参数:
            data (str): 数据集配置文件路径 (YAML)
            epochs (int): 训练轮数
            imgsz (int): 输入图像尺寸
            batch (int): 批次大小
            project (str): 项目保存路径
            name (str): 实验名称
            
        返回:
            results: 训练结果
        """
        print(f"\n开始训练模型")
        print(f"  数据集: {data}")
        print(f"  训练轮数: {epochs}")
        print(f"  图像尺寸: {imgsz}")
        print(f"  批次大小: {batch}")
        
        results = self.model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=project,
            name=name
        )
        
        print(f"\n训练完成!")
        return results
    
    def validate(self, data=None, split="val", imgsz=640):
        """
        验证模型性能
        
        参数:
            data (str): 数据集配置文件路径，None 表示使用训练时的数据集
            split (str): 数据集划分 ('val', 'test')
            imgsz (int): 输入图像尺寸
            
        返回:
            metrics: 验证指标
        """
        print(f"\n开始验证模型")
        metrics = self.model.val(
            data=data,
            split=split,
            imgsz=imgsz,
            device=self.device
        )
        
        print(f"\n验证结果:")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP75: {metrics.box.map75:.4f}")
        
        return metrics
    
    def export(self, format="onnx", imgsz=640, half=False, simplify=True):
        """
        导出模型到其他格式
        
        参数:
            format (str): 导出格式 ('onnx', 'torchscript', 'openvino', 'engine', 'coreml', 'tflite' 等)
            imgsz (int): 输入图像尺寸
            half (bool): 是否使用 FP16 半精度
            simplify (bool): 是否简化 ONNX 模型
            
        返回:
            export_path: 导出文件路径
        """
        print(f"\n开始导出模型到 {format.upper()} 格式")
        export_path = self.model.export(
            format=format,
            imgsz=imgsz,
            half=half,
            simplify=simplify
        )
        
        print(f"\n导出完成: {export_path}")
        return export_path


def main():
    """主函数：解析命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(description="YOLOv8 目标检测完整示例")
    
    parser.add_argument("--mode", type=str, default="predict", 
                       choices=["predict", "train", "validate", "export", "video", "webcam"],
                       help="运行模式")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                       help="模型权重文件路径")
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"),
                       help="输入源 (图像路径、视频路径、摄像头编号)")
    parser.add_argument("--data", type=str, default="coco8.yaml",
                       help="数据集配置文件路径 (训练/验证时使用)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7,
                       help="NMS IOU 阈值")
    parser.add_argument("--device", type=str, default="",
                       help="运行设备 (cpu, cuda, mps 等)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--format", type=str, default="onnx",
                       help="导出格式 (onnx, torchscript, openvino, engine, coreml, tflite 等)")
    parser.add_argument("--save", action="store_true", default=True,
                       help="是否保存结果")
    parser.add_argument("--show", action="store_true",
                       help="是否显示结果")
    
    args = parser.parse_args()
    
    detector = ObjectDetector(model_path=args.model, device=args.device)
    
    if args.mode == "predict":
        detector.predict_image(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            show=args.show
        )
        
    elif args.mode == "video":
        detector.predict_video(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            show=args.show
        )
        
    elif args.mode == "webcam":
        source = int(args.source) if args.source.isdigit() else args.source
        detector.predict_webcam(
            source=source,
            conf=args.conf,
            iou=args.iou
        )
        
    elif args.mode == "train":
        detector.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch
        )
        
    elif args.mode == "validate":
        detector.validate(
            data=args.data,
            imgsz=args.imgsz
        )
        
    elif args.mode == "export":
        detector.export(
            format=args.format,
            imgsz=args.imgsz
        )
        
    print("\n任务完成!")


if __name__ == "__main__":
    main()