from ultralytics import YOLO

# 加 载 YOLOv11 nano 预 训 练 权 重 （ 首 次 运 行 自 动 下 载 ， 约 5MB ）
model = YOLO('yolo11n.pt')

# 注 意 ： data 填 写 data.yaml 的 完 整 路 径
# 如 果 train.py 和 数 据 集 文 件 夹 放 在 同 一 目 录 ， 可 以 用 下 面 的 相 对 路 径 ：
results = model.train(
data='data.yaml',
epochs =10, # CPU 训 练 建 议 10 轮 ， 约 30 -60 分 钟
imgsz =412, # 降 低 分 辨 率 以 加 快 CPU 训 练 速 度
batch =4, # CPU 训 练 batch 设 小 一 点
device='cpu', # 无 GPU 则 使 用 CPU
workers =2,
project ='runs/detect',
name='yolo11_vp',
val=True ,
)

print(" 训练完成!权重保存于 :", results .save_dir)