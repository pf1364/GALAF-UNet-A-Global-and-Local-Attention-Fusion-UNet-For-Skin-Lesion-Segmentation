import os
import sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from tqdm import tqdm
!pip install MedPy

try:
    from medpy import metric
    medpy_available = True
except ImportError:
    print("Warning: MedPy library not found...")
    medpy_available = False
 
sys.path.append('/kaggle/input/unet')
from src import UNet

 
# 1. 评估设置
class TestConfig:
    num_classes = 2
    # 2017的mean和std
    # mean = (0.7060, 0.5888, 0.5449)
    # std = (0.0941, 0.1108, 0.1248)
    # 2018的mean和std
    mean = (0.7082, 0.5799, 0.5343)
    std = (0.0990, 0.1136, 0.1277)
    input_size_h = 256
    input_size_w = 256
    model_path = '/kaggle/input/models/unet_best_model.pth'
    test_data_path = '/kaggle/input/isic2018-split/test' 
    results_dir = '/kaggle/working/test_results'
    save_interval = 1000
    batch_size = 8
    workers = 4
 
# 2. 可视化图片保存函数 (no changes here)
def save_visualization_image(image_tensor, gt_mask, pred_mask, save_path, config):
    mean = torch.tensor(config.mean).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor(config.std).view(3, 1, 1).to(image_tensor.device)
    image_tensor = image_tensor * std + mean
    image_tensor = torch.clamp(image_tensor, 0, 1)
    image_pil = F.to_pil_image(image_tensor)
    gt_mask_pil = Image.fromarray((gt_mask * 255).astype(np.uint8), mode='L').convert('RGB')
    pred_mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L').convert('RGB')
    combined_image = Image.new('RGB', (image_pil.width * 3, image_pil.height))
    combined_image.paste(image_pil, (0, 0))
    combined_image.paste(gt_mask_pil, (image_pil.width, 0))
    combined_image.paste(pred_mask_pil, (image_pil.width * 2, 0))
    combined_image.save(save_path)
 
# 3. 数据预处理和数据集类
class JointTransform:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, img, mask):
        for t in self.transforms: img, mask = t(img, mask)
        return img, mask
 
class JointResize:
    def __init__(self, size): self.size = size
    def __call__(self, img, mask):
        img = F.resize(img, self.size, interpolation=T.InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST)
        return img, mask
 
class ToTensor:
    def __call__(self, img, mask):
        img = F.to_tensor(img)
        mask = torch.as_tensor(np.array(mask) // 255, dtype=torch.int64)
        return img, mask
 
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def forward(self, img, mask):
        return F.normalize(img, self.mean, self.std), mask
    __call__ = forward
 
def get_test_transforms(config):
    transform_list = [
        JointResize([config.input_size_h, config.input_size_w]),
        ToTensor(),
        Normalize(mean=config.mean, std=config.std)
    ]
    return JointTransform(transform_list)


class ISICDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms):
        self.transforms = transforms
        image_dir = os.path.join(root_dir, 'images')
        mask_dir = os.path.join(root_dir, 'ground_truth')
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.mask_paths = []
        for img_path in self.image_paths:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_name = f"{img_name}_segmentation.png"
            self.mask_paths.append(os.path.join(mask_dir, mask_name))
 
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        # 直接应用完整的 transform pipeline
        img, mask = self.transforms(img, mask)
        return img, mask
 
    def __len__(self):
        return len(self.image_paths)
 
 
def evaluate_with_all_metrics(model, data_loader, device, config):
    model.eval()
    per_image_records = []
    
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)
        print(f"Created directory for saving results: {config.results_dir}")
 
    dice_scores, miou_scores, acc_scores= [], [], []
    precision_scores, recall_scores, specificity_scores = [], [], []
    hd95_scores, asd_scores = [], []
 
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for batch_idx, (image_batch, target_batch) in progress_bar:
            image_batch, target_batch = image_batch.to(device), target_batch.to(device)
            output = model(image_batch)['out']
            pred_batch = torch.argmax(output, dim=1)
 
            if batch_idx % config.save_interval == 0:
                image_to_save = image_batch[0].cpu()
                gt_to_save = target_batch[0].cpu().numpy().astype(np.uint8)
                pred_to_save = pred_batch[0].cpu().numpy().astype(np.uint8)
                save_path = os.path.join(config.results_dir, f"batch_{batch_idx}_visualization.png")
                save_visualization_image(image_to_save, gt_to_save, pred_to_save, save_path, config)
 
            for i in range(pred_batch.shape[0]):
                pred_single = pred_batch[i].cpu().numpy().astype(np.uint8)
                target_single = target_batch[i].cpu().numpy().astype(np.uint8)
                if np.sum(target_single) == 0 and np.sum(pred_single) == 0:
                    dice, miou, acc, precision, recall, specificity = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
                elif np.sum(target_single) == 0 and np.sum(pred_single) > 0:
                    # dice, iou, precision, recall, specificity = 0.0, 0.0, 0.0, 0.0, 0.0
                    TP = 0
                    FP = np.sum(pred_single)
                    TN = np.sum((pred_single == 0) & (target_single == 0)) 
                    FN = 0
                    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
                    
                    iou_foreground = 0.0 # 因为 TP=0
                    iou_background = TN / (TN + FP + FN + 1e-8) # TN / (TN + FP)
                    miou = (iou_foreground + iou_background) / 2.0
                    dice, precision, recall, specificity = 0.0, 0.0, 0.0, (TN / (TN + FP + 1e-8))
                else:
                    TP = np.sum((pred_single == 1) & (target_single == 1))
                    TN = np.sum((pred_single == 0) & (target_single == 0))
                    FP = np.sum((pred_single == 1) & (target_single == 0))
                    FN = np.sum((pred_single == 0) & (target_single == 1))
                    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)

                    # 计算前景 IoU
                    iou_foreground = TP / (TP + FP + FN + 1e-8)
                    # 计算背景 IoU (TP_background = TN)
                    iou_background = TN / (TN + FN + FP + 1e-8)
                    # 计算 mIoU
                    miou = (iou_foreground + iou_background) / 2.0

                    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
                    
                    # iou = TP / (TP + FP + FN + 1e-8)
                    precision = TP / (TP + FP + 1e-8)
                    recall = TP / (TP + FN + 1e-8)
                    specificity = TN / (TN + FP + 1e-8)
                
                dice_scores.append(dice)
                miou_scores.append(miou)
                acc_scores.append(acc)
                precision_scores.append(precision)
                recall_scores.append(recall)
                specificity_scores.append(specificity)
                if medpy_available:
                    if np.sum(pred_single) > 0 and np.sum(target_single) > 0:
                        try:
                            hd95_scores.append(metric.binary.hd95(pred_single, target_single))
                            asd_scores.append(metric.binary.asd(pred_single, target_single))
                        except RuntimeError: pass
                    elif np.sum(pred_single) == 0 and np.sum(target_single) == 0:
                        hd95_scores.append(0.0)
                        asd_scores.append(0.0)
    
    metrics_summary = {"Dice": np.mean(dice_scores),
                       "mIoU": np.mean(miou_scores), 
                       "Accuracy": np.mean(acc_scores),
                       "Precision": np.mean(precision_scores), 
                       "Recall": np.mean(recall_scores), 
                       "Specificity": np.mean(specificity_scores), 
                       "HD95": np.mean(hd95_scores) if hd95_scores else -1.0, 
                       "ASD": np.mean(asd_scores) if asd_scores else -1.0
                      }
    return metrics_summary
 
 
def main():
    config = TestConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 模型加载部分不变
    print(f"Loading model from: {config.model_path}")
    model = UNet(in_channels=3, num_classes=config.num_classes, base_c=64)
    # model = UNetPP(in_channels=3, num_classes=config.num_classes, base_c=64)
    # model = AttentionUNet(in_channels=3, num_classes=config.num_classes, base_c=32)
        
    checkpoint = torch.load(config.model_path, map_location=device)
    # 这个加载逻辑可以兼容字典和直接的state_dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    print("Model loaded successfully.")
 
    test_transforms = get_test_transforms(config)
    test_dataset = ISICDataset(root_dir=config.test_data_path, transforms=test_transforms)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True
    )
    
    metrics_results = evaluate_with_all_metrics(model, test_loader, device, config)
    
    # 打印结果
    print(f"\nVisualizations saved in: {config.results_dir}")
    print("\n" + "="*30)
    print("      Evaluation Results")
    print("="*30)
    for name, value in metrics_results.items():
        if ("HD95" in name or "ASD" in name) and value == -1.0:
            print(f"{name:<12}: Not calculated")
        else:
            print(f"{name:<12}: {value:.4f}")
    print("="*30)


if __name__ == "__main__":
    main()