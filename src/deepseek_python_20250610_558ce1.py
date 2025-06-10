import os
import math  # Added for CIoU calculation
import torch
import cv2
import albumentations as A
from torch import nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import Dataset
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model with proper class count
model = YOLO('yolov8n.pt').to(device)
NUM_CLASSES = 1  # Set to your actual class count
model.model.nc = NUM_CLASSES

# Quantization setup (updated parameters)
model.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model.model, inplace=True)

# Parameter freezing
total_params = sum(p.numel() for p in model.model.parameters())
freeze_num = int(total_params * 0.8)
for idx, (name, param) in enumerate(model.model.named_parameters()):
    if idx < freeze_num:
        param.requires_grad = False

# Model modifications
model.model = model.model.fuse().to(device)

# Fixed transforms
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.7),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.OneOf([
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.5),
    ], p=0.7),
    A.Affine(  # Replaced ShiftScaleRotate
        translate_percent=0.1,
        scale=(0.85, 1.15),
        rotate=(-25, 25),
        cval=0,
        p=0.8
    ),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0)),  # Fixed parameter
        A.ISONoise(color_shift=(0.01, 0.05)),  # Removed p
        A.MultiplicativeNoise(multiplier=(0.9, 1.1))  # Removed p
    ], p=0.3),
    A.CoarseDropout(  # Fixed parameters
        max_holes=8,
        max_height=32,
        max_width=32,
        min_holes=1,
        min_height=8,
        min_width=8,
        p=0.3
    ),
    # A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),  # Fixed parameter
    A.Resize(height=640, width=640, p=1.0),
    ToTensorV2()
],
    bbox_params=A.BboxParams(format='yolo', min_visibility=0.4),
)

val_transform = A.Compose([
    A.Resize(height=640, width=640, p=1.0),
    ToTensorV2()
])

class TrainDataset(Dataset):
    def __init__(self, root, annotations, transform=None):
        self.image_paths = root
        self.transform = transform
        self.image_files = []
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for file in os.listdir(root):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                self.image_files.append(os.path.join(root, file))
        
        if not self.image_files:
            raise RuntimeError(f"No images found in directory: {root}")
        
        self.annotations = self.load_annotations(annotations)  # Pass annotations directory

    def load_annotations(self, annotations_dir):
        annotations = {}
        for img_file in self.image_files:
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            label_file = Path(annotations_dir) / f"{base_name}.txt"
            labels = []
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        labels.append([
                            x_center, 
                            y_center, 
                            width, 
                            height,
                            class_id  # Include class ID
                        ])
            annotations[img_file] = labels
        return annotations

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.annotations[img_path]
        bboxes = []
        for bbox in label:
            # Keep coordinates in absolute values
            bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes)
            image = transformed['image']
            bboxes = transformed['bboxes']
            print(type(bboxes))
        # Convert to tensor
        image = image.float() / 255.0
        return image, torch.tensor(bboxes, dtype=torch.float32)  # Keep absolute coordinates

def collate_fn(batch):
    images, bboxes = zip(*batch)
    images = torch.stack(images)
    bboxes = [bbox for bbox in bboxes if len(bbox) > 0]  # Filter empty bboxes
    return images, bboxes

def train_model(model, train_loader, optimizer, criterion, device, num_epochs=300, val=True):
    # model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, bboxes in train_loader:  # Fixed iteration
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, bboxes)  # Pass entire batch
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        if val:
            val_loss = validate_model(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}")
    
    # Convert quantization
    torch.quantization.convert(model.model, inplace=True)
    return model

def validate_model(model, val_loader, criterion, device):
    # model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, bboxes in val_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, bboxes)
            val_loss += loss.item() * images.size(0)
    val_loss /= len(val_loader.dataset)
    return val_loss

def loss_fn(outputs, targets):
    # Simplified loss - use built-in YOLO loss instead
    # Placeholder implementation
    reg_loss = F.mse_loss(outputs[0].boxes, targets[..., :4])
    conf_loss = F.binary_cross_entropy_with_logits(
        outputs[0].boxes, 
        (targets[..., 4] > 0).float()
    )
    return reg_loss + conf_loss

# Initialize datasets
train_image_paths = 'data/images/train'
train_annotations = 'data/labels/train'
train_dataset = TrainDataset(train_image_paths, train_annotations, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

val_image_paths = 'data/images/val'
val_annotations = 'data/labels/val'
val_dataset = TrainDataset(val_image_paths, val_annotations, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trained_model = train_model(model, train_loader, optimizer, loss_fn, device, val=True)
torch.save(trained_model.state_dict(), 'yolov8n_finetuned.pt')

# Export function
def export_for_edge(model):
    # model.eval()
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        "yolov8n_int8.onnx",
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )

export_for_edge(model)