import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import Conv2d as Conv
from pathlib import Path
import torch.nn.functional as F
import math
# The preparation of train
os.chdir('/workspace/Objective-Detection-Competition/src')
device = torch.device('cpu')
model = YOLO('yolov8n.pt').to(device)
model.model.nc = 0
# Freeze 80 percentage bakebone paraments
total_params = len(list(model.model.parameters()))
freeze_num = int(total_params * 0.8)  # 计算80%冻结点
for idx, param in enumerate(model.model.parameters()):
    if idx < freeze_num:  # 冻结前80%参数
        param.requires_grad_(False)    
model = model.to(device).half()  
# Define a custom detect layer
class Detect(nn.Module):
    
        
        
# Dataset
class TrainDataset(Dataset):
    def __init__(self, data_dir, mod='train', transform=None, nc=1):
        self.data_dir = data_dir
        self.mod = mod
        self.transform = transform
        self.nc = nc
        self.images = []
        self.labels = []
        self.load_data()
        self.classes = ['mushroom']
        self.class_map = {class_name: i for i, class_name in enumerate(self.classes)}
        self.image_size = (640, 640)

    def load_data(self):
        image_dir = Path(self.data_dir) / 'images' / self.mod
        label_dir = Path(self.data_dir) / 'labels' / self.mod
        
        # 获取所有图像文件路径
        self.image_paths = [f for f in image_dir.glob('*.jpg')]
        # 对应标签文件路径
        self.label_paths = [label_dir / os.path.basename(f).replace('.jpg', '.txt') for f in self.image_paths]

        for icd, file in enumerate(self.image_paths):
            label_path = self.label_paths[icd]
            if label_path.exists():
                self.images.append(file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    label = []
                    for line in lines:
                        line_contain = line.strip().split(',')
                        if len(line_contain) >= 5:
                            class_id, x, y, w, h = map(float, line_contain)
                            label.append([class_id, x, y, w, h])
                        else:
                            class_id = 1
                            x, y, w, h = map(float, line_contain)
                            label.append([class_id, x, y, w, h])
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_ids = []
        bboxes = []
        for box in label:
            class_id, x, y, w, h = box
            bboxes.append([x,y,w,h])
            class_ids.append(class_id)
        image = image.astype(np.float32) / 255.0
        bboxes = np.array(bboxes) /640.0
        class_ids = np.array(class_ids)
        if self.transform:
            transform = self.transform(image=image, bboxes=bboxes, class_ids=class_ids )
            image = transform[0]
            bboxes = transform[1]
            class_ids = transform[2]

        class_ids = torch.tensor(class_ids)

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        return image, {'bboxes':bboxes,'class_ids':class_ids}


    
# Transform 
def augment_image(image,bboxes,class_ids):

    augmented = transform(image=image, bboxes=bboxes, class_ids=class_ids)
    return augmented['image'], augmented['bboxes'], augmented['class_ids']

transform = A.Compose([
    A.Resize(height=640, width=640, p=1.0),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    
    # 颜色增强
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.7),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    
    # 改进的色彩增强组合
    A.OneOf([
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.5),
    ], p=0.7),
    # 几何变换
    A.Affine(
        translate_percent=0.1,
        scale=(0.85, 1.15),  # 对应 scale_limit=0.15
        rotate=(-25, 25),    # 对应 rotate_limit=25
        p=0.8
    ),
    # 噪声和模糊
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7),p=0.3),
        A.GaussNoise(p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05)),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1))
    ], p=0.3),
    # 裁剪和缩放
    A.CoarseDropout(p=0.3),
    A.RandomResizedCrop(size=(640,640), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
    ToTensorV2()
],
    bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=['class_ids']),
)
val_transform = A.Compose([
    A.Resize(height=640, width=640, p=1.0),
    ToTensorV2()
])
# Train model
def collate_fn(batch):

    images, labels = zip(*batch)
    bboxes = [labels['bboxes'] for labels in labels]
    class_id = [labels['class_ids'] for labels in labels]

    # 统一图像维度 [B, C, H, W]
    images = torch.stack(images)
    
    # 计算最大边界框数量
    max_boxes = max(len(b) for b in bboxes)
    
    # 填充边界框和标签
    padded_bboxes = []
    padded_class_id = []
    valid_masks = []  # 有效边界框掩码
    
    for bbox, class_id in zip(bboxes, class_id):
        num_boxes = len(bbox)
        
        # 边界框填充
        if num_boxes < max_boxes:
            pad_bbox = torch.cat([
                bbox,
                torch.zeros((max_boxes - num_boxes, 4), dtype=torch.float32)
            ], dim=0)
        else:
            pad_bbox = bbox
        
        # 标签填充
        pad_class_id = torch.cat([
            class_id,
            torch.zeros((max_boxes - num_boxes), dtype=torch.float32)
        ], dim=0)
        
        # 有效掩码（1表示真实框，0表示填充）
        valid_mask = torch.cat([
            torch.ones(num_boxes),
            torch.zeros(max_boxes - num_boxes)
        ], dim=0)
        
        padded_bboxes.append(pad_bbox)
        padded_class_id.append(pad_class_id)
        valid_masks.append(valid_mask)
    return {
        'images': images,
        'bboxes': torch.stack(padded_bboxes),
        'class_id': torch.stack(padded_class_id),
        'masks': torch.stack(valid_masks).bool()
    }
def train_model(model, train_loader, optimizer, criterion, device, num_epochs=300, val = True):
    # model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            images = batch['images']
            bboxes = batch['bboxes']
            class_id = batch['class_id']
            masks = batch['masks']
            one_bboxes = bboxes[masks]
            one_class_id = class_id[masks]
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, one_bboxes, one_class_id,bboxes,class_id)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        if val:
            val_loss = validate_model(model, val_loader, criterion)
            print(f"Validation Loss: {val_loss:.4f}")
    with torch.no_grad():
        for images, _ in train_loader:
            _ = model(images.to(device))
    torch.quantization.convert(model.model, inplace=True)
    return model
def validate_model(model, val_loader, criterion, device):  # 添加device参数
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['images']
            bboxes = batch['bboxes']
            class_id = batch['class_id']
            masks = batch['masks'].to(device)
            one_bboxes = bboxes[masks]
            one_labels = class_id[masks]
            outputs = model(images)
            loss = criterion(outputs, one_bboxes, one_labels,bboxes,class_id)
            val_loss += loss.item() * images.size(0) 
    val_loss /= len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss
def loss_fn(outputs, one_bboxes, one_labels, bboxes, labels):
    # 获取正确的输出格式
    _, z = outputs  # 根据Detect层forward返回值调整
    
    # 重构损失计算
    cls_loss = nn.BCEWithLogitsLoss()(z[0], labels)
    reg_loss = nn.MSELoss()(z[1], one_bboxes)
    return cls_loss + reg_loss

def calculate_ciou(box1, box2):
    box1_xywh = box1.xywh
    box1_xyxy = box1.xyxy
    box2_xywh = box2[:]
    box2cx = box2[:,0]
    box2cy = box2[:,1]
    box2w = box2[:,2]
    box2h = box2[:,3]
    box2x = box2cx - box2w / 2
    box2y = box2cy - box2h / 2
    box2_xyxy = torch.stack([box2x, box2y, box2x + box2w, box2y + box2h], dim=1)
    inter_x1 = torch.max(box1_xyxy[:, 0], box2_xyxy[:, 0])
    inter_y1 = torch.max(box1_xyxy[:, 1], box2_xyxy[:, 1])
    inter_x2 = torch.min(box1_xyxy[:, 2], box2_xyxy[:, 2])
    inter_y2 = torch.min(box1_xyxy[:, 3], box2_xyxy[:, 3])
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    box1_area = (box1_xyxy[:, 2] - box1_xyxy[:, 0]) * (box1_xyxy[:, 3] - box1_xyxy[:, 1])
    box2_area = (box2_xyxy[:, 2] - box2_xyxy[:, 0]) * (box2_xyxy[:, 3] - box2_xyxy[:, 1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    cw = torch.max(box1_xywh[:, 2], box2_xywh[:, 2])
    ch = torch.max(box1_xywh[:, 3], box2_xywh[:, 3])
    w2 = box2_xywh[:, 2]
    h2 = box2_xywh[:, 3]
    arctan = torch.atan(w2 / h2) if h2 != 0 else 0
    v = (4 / (math.pi ** 2)) * (arctan - box2_xywh[:, 0]) ** 2
    alpha = v / (1 - iou + v)
    ciou = iou - (1 - alpha * v)
    return ciou


train_image_paths = 'data'
mod_train = 'train'
train_dataset = TrainDataset(train_image_paths, mod=mod_train, transform=augment_image)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_image_paths = 'data/images/val'
val_dataset = TrainDataset(val_image_paths, 'val', transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = loss_fn
trained_model = train_model(model, train_loader, optimizer, criterion, device, val = True)
validated_model = validate_model(model, val_loader, criterion)
if validated_model<0.5:
    torch.save(trained_model.state_dict(), 'yolov8n.pt')
