from ultralytics import YOLO
import torch
import os
import cv2
import numpy as np
import albumentations as al
# model = YOLO('yolov8n.pt')
os.system("pwd")
image = "data/images/train/1748333053264.jpg"
transform = al.Compose([
    al.RandomRotate90(p=0.5),
    al.HorizontalFlip(p=0.5),
    al.VerticalFlip(p=0.2),
    
    # 颜色增强
    al.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    al.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.7),
    al.RandomGamma(gamma_limit=(80, 120), p=0.5),
    
    # 改进的色彩增强组合

    # 几何变换
  al.Affine(
        translate_percent=0.1,
        scale=(0.85, 1.15),  # 对应 scale_limit=0.15
        rotate=(-25, 25),    # 对应 rotate_limit=25
        p=0.8
    ),
    # 噪声和模糊
    al.OneOf([
        al.GaussianBlur(blur_limit=(3, 7),p=0.3),
        al.GaussNoise(var_limit=(10.0, 50.0), p=1),
        al.ISONoise(color_shift=(0.01, 0.05)),
        al.MultiplicativeNoise(multiplier=(0.9, 1.1))
    ], p=1),
    # 裁剪和缩放
    al.CoarseDropout(
        max_holes=8,
        max_height=32,
        max_width=32,     # 添加     # 添加
        p=1
    ),
])
image = cv2.imread(image)
image = transform(image=image)['image']
image = cv2.resize(image, (640, 640))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (640, 640))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite("1.jpg",image)
image = image.astype(np.float32) / 255.0  
image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0)  

# 修改后正确的推理和结果解析
# result = model(image)[0] 
# print(type(result)) # 取第一个输出（Results对象）
# boxes = result.boxes.xyxy.cpu().numpy()  # 转为numpy数组
# print(type(boxes))
# print("检测框坐标：\n", boxes)
