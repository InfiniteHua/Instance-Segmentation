from ultralytics import YOLO
import os

if __name__ == "__main__":
    # ===========================
    # 配置区
    # ===========================
    # 数据集路径：YOLO 格式
    # 数据集结构示例：
    # dataset/
    #   images/
    #       train/
    #       val/
    #   labels/
    #       train/
    #       val/
    dataset_path = "flowers/data.yaml"  # 修改为你的数据集路径

    # 模型选择: yolov8n-seg (轻量), yolov8s-seg, yolov8m-seg 等
    model_name = "yolov8n-seg"

    # 训练参数
    epochs = 120
    batch_size = 4
    img_size = 1280
    learning_rate = 0.001

    # 保存权重路径
    save_dir = "runs/flower_seg"

    # ===========================
    # 创建模型
    # ===========================
    model = YOLO(model_name)

    # ===========================
    # 训练
    # ===========================
    # ultralytics 的 train 方法可以直接传入 YOLO 数据集路径
    model.train(
        data=dataset_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate,
        project=save_dir,
        name="flower_segmentation",
        exist_ok=True  # 如果已有相同名称的目录则覆盖
    )

    # ===========================
    # 保存最终模型
    # ===========================
    final_model_path = os.path.join(save_dir, "flower_segmentation", "weights", "best.pt")
    print(f"训练完成，权重保存在: {final_model_path}")
