from functools import partial
import os
import torch
import cv2
import albumentations as A
import numpy as np
from torch import nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import Dataset
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

device = torch.device('cpu')
model = YOLO('yolov8n.pt').to(device)
model.model.nc = 0
model.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # CPU/NPU适配
torch.quantization.prepare(model.model, inplace=True)

# 修改后
total_params = len(list(model.model.parameters()))
freeze_num = int(total_params * 0.8)  # 计算80%冻结点

for idx, param in enumerate(model.model.parameters()):
    if idx < freeze_num:  # 冻结前80%参数
        param.requires_grad_(False)    
model = model.to(device).half()  
class Detect(nn.Module):
    def __init__(self, nc=80, ch=(256, 512, 1024)):
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数
        self.reg_max = 16  # DFL参数
        
        # 分类分支
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, x, 3), Conv(x, x, 3), 
            nn.Conv2d(x, nc, 1)) for x in ch)
        
        # 回归分支
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, x, 3), Conv(x, x, 3), 
            nn.Conv2d(x, 4 * self.reg_max, 1)) for x in ch)
model.model = model.model.fuse()  # 卷积层融合
model.model = model.model.half()  # FP16量化

val_transform = A.Compose([
    A.Resize(height=640, width=640, p=1.0),
    ToTensorV2()
])

# 数据集
class TrainDataset(Dataset):
    def __init__(self, root,anotations,transform=None):
        self.image_paths = root
        self.transform = transform
        self.image_files = []
        self.annotations = []
        # 寻找文件
        for file in os.listdir(root):
            self.image_files.append(os.path.join(root, file))
        
        # 确保找到图像文件
        if not self.image_files:
            raise RuntimeError(f"No images found in directory: {root}")
        
        # 加载标注文件
        if anotations is not None:
            self.annotations = self.load_annotations()
    # 加载标注文件
    def load_annotations(self):
        annotations = {}
        subset_name = os.path.basename(self.image_paths) # 获取子集名称 (train/val)
        
        for img_file in self.image_files:
            # 文件名称
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            
            # 更健壮的路径构建
            base_dir = Path(self.image_paths).parent.parent
            label_file = base_dir / 'labels' / subset_name / f"{base_name}.txt"
            # 读取标注
            labels = []
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f.readlines():
                        part = line.strip().split(',')
                        for num in range(len(part)):
                            part[num] = float(part[num])
                        labels.append(part)
            annotations[img_file] = labels
        
        return annotations
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        image_names = self.image_files[idx]
        image = cv2.imread(image_names)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0  
        label = self.annotations[self.image_files[idx]]
        for num in range(len(label)):
            boxes = label[num]
            if len(boxes) == 0:  
                continue
            for i in range(len(boxes)):
                label[num][i] = boxes[i] / 640 
        image, bboxes = self.transform(image, label)
        # bboxes = torch.from_numpy(np.array(bboxes)).unsqueeze(0)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        return image, bboxes
def augment_image(image, bboxes):
    augmented = transform(image=image, bboxes=bboxes)
    return augmented['image'], augmented['bboxes']
# def custom_collate(batch):
#     images, targets = zip(*batch)
    
#     # 统一图像尺寸
#     images = torch.stack([img for img in images])
    
#     # 处理标签维度
#     max_boxes = max(len(t["boxes"]) for t in targets)
#     padded_targets = []
#     for t in targets:
#         boxes = t["boxes"]
#         # 填充到最大边界框数量
#         padded_boxes = F.pad(boxes, (0, 0, 0, max_boxes - len(boxes)), value=-1)
#         # 创建有效性掩码
#         valid = torch.cat([
#             torch.ones(len(boxes), dtype=torch.bool),
#             torch.zeros(max_boxes - len(boxes), dtype=torch.bool)
#         ])
#         padded_targets.append({
#             'boxes': padded_boxes,
#             'valid': valid
#         })
    
#     return images, padded_targets

# 更新DataLoader配置
transform = A.Compose([
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
    A.Resize(height=640, width=640, p=1.0),
    ToTensorV2()
],
    bbox_params=A.BboxParams(format='yolo', min_visibility=0.4),
)
val_transform = A.Compose([
    A.Resize(height=640, width=640, p=1.0),
    ToTensorV2()
])
def collate_fn(batch):
    images, bboxes = zip(*batch)
    images = torch.stack(images)
    max_boxes = max(len(t) for t in bboxes)
    padded_targets = []
    for t in bboxes:
        boxes = t        # 填充到最大边界框数量
        padded_boxes = F.pad(boxes, (0, 0, 0, max_boxes - len(boxes)), value=-1)
        valid = torch.cat([
            torch.ones(len(boxes), dtype=torch.bool),
            torch.zeros(max_boxes - len(boxes), dtype=torch.bool)
        ])
        padded_targets.append({
            'boxes': padded_boxes,
            'valid': valid
        })
    bboxes = torch.stack([t['boxes'] for t in padded_targets])
    return images, bboxes
    # bboxes = torch.stack(bboxes)  
    # return images, bboxes
def train_model(model, train_loader, optimizer, criterion, device, num_epochs=300, val = True):
    # model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, bboxes in train_loader:
            boxes =[]
            for t in bboxes:
                valid_boxes = t['boxes'][t['valid']]
                boxes.append(valid_boxes)
            # for i in range(len(bboxes)):
            #     image = images[i]
            #     box = bboxes[i]
                # valid_boxes = b['boxes'][box['valid']] 
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, boxes)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        if val:
            val_loss = validate_model(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}")
    with torch.no_grad():
        for images, _ in train_loader:
            _ = model(images.to(device))
    torch.quantization.convert(model.model, inplace=True)
    return model
def validate_model(model, val_loader, criterion):
    # model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, bboxes in val_loader:
            outputs = model(images)
            loss = criterion(outputs, bboxes)
            val_loss += loss.item() * images.size(0) 
    val_loss /= len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}")
    return val_loss
def loss_fn(outputs, targets):
    # 定位损失 CIou
    boxes = []
    for ele in outputs:
        boxe = ele.boxes
        boxes.append(boxe)

    ciou = calculate_ciou(boxes, targets[..., :4])
    reg_loss = 1.0 - ciou

    pred_conf = torch.sigmoid(outputs[..., 4])
    gt_conf = (targets[..., 4] > 0).float()
    conf_loss = F.binary_cross_entropy_with_logits(
        outputs[..., 4], 
        gt_conf,
        reduction='none'
    ) * ((1 - pred_conf) ** 2 * gt_conf + pred_conf ** 2 * (1 - gt_conf))
    
    # 平衡权重
    return (reg_loss * 3.0 + conf_loss * 0.5).mean()

    # 置信度损失

def calculate_ciou(box1, box2):
    # box1 = box1[0]
    # box1 = torch.tensor(box1[:4])
    # box1 = torch.clone(box1).detach().unsqueeze(1)
    box1 = box1.unsqueeze(1)
    box2 = box2.unsqueeze(0)
    # 坐标转换函数优化
    def _cwh_to_xyxy(box):
        xyxy = torch.empty_like(box)
        xyxy[..., 0] = box[..., 0] - box[..., 2] / 2  # x1
        xyxy[..., 1] = box[..., 1] - box[..., 3] / 2  # y1
        xyxy[..., 2] = box[..., 0] + box[..., 2] / 2  # x2
        xyxy[..., 3] = box[..., 1] + box[..., 3] / 2  # y2
        return xyxy
    
    # 保持原始cwh格式用于宽高比计算
    box1_cwh = box1.clone()
    box2_cwh = box2.clone()
    
    # 转换到xyxy格式
    box1 = _cwh_to_xyxy(box1)
    box2 = _cwh_to_xyxy(box2)
    
    # 交并比计算优化
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    # 数值稳定性增强
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = area1 + area2 - inter_area + 1e-6
    
    # CIoU完整计算
    center_distance = torch.sum((box1_cwh[..., :2] - box2_cwh[..., :2]) ** 2, dim=-1)
    
    enclose_x1 = torch.min(box1[..., 0], box2[..., 0])
    enclose_y1 = torch.min(box1[..., 1], box2[..., 1])
    enclose_x2 = torch.max(box1[..., 2], box2[..., 2])
    enclose_y2 = torch.max(box1[..., 3], box2[..., 3])
    c_squared = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(box1_cwh[..., 2]/box1_cwh[..., 3].clamp(min=1e-6)) - 
        torch.atan(box2_cwh[..., 2]/box2_cwh[..., 3].clamp(min=1e-6)), 2)
    
    alpha = v / (1 - (inter_area/union_area) + v + 1e-6)
    ciou = (inter_area/union_area) - (center_distance/(c_squared + 1e-6)) - (alpha * v)
    
    return ciou.mean()

train_image_paths = 'data/images/train'
annotations = 'data/labels/train'
train_dataset = TrainDataset(train_image_paths, annotations, transform=augment_image)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_image_paths = 'data/images/val'
val_annotations = 'data/labels/val'
val_dataset = TrainDataset(val_image_paths, val_annotations, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = loss_fn
trained_model = train_model(model, train_loader, optimizer, criterion, device, val = True)
torch.save(trained_model.state_dict(), 'yolov8n.pt')
import subprocess
def export_for_edge():
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    # 量化感知导出
    torch.onnx.export(
        model,
        dummy_input,
        "yolov8n_int8.onnx",
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        quantization_dir='calibration_data', 
    )
# 原问题 “which page is it form” 与代码修改无关，由于没有明确的修改逻辑，保持原代码不变
  # 确保导入 subprocess 模块
    subprocess.run([
        'mo',
        '--input_model', 'yolov8n_int8.onnx',
        '--data_type', 'INT8',
        '--mean_values', '[123.675,116.28,103.53]',
        '--scale_values', '[58.395,57.12,57.375]',
        '--output_dir', './openvino_int8'
    ], check=True)
validate_model(model, val_loader, criterion)
# export_for_edge()
