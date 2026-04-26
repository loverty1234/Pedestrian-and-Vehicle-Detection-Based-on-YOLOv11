from ultralytics import YOLO

# 加载训练好的权重
model = YOLO('runs/detect/runs/detect/yolo11_vp/weights/best.pt')

# 对图片进行推理（将 test.jpg 替换为你自己的图片）
results = model.predict(
    source='img.png',
    conf=0.25,  # 置信度阈值，低于此值的检测框会被过滤
    iou=0.45,  # NMS 去重阈值
    save=True,  # 保存标注结果图
    device='cpu',
)

# 打印检测到的目标
for r in results:
    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        name = model.names[cls]
        print(f"检测到：{name}，置信度：{conf:.2f}")