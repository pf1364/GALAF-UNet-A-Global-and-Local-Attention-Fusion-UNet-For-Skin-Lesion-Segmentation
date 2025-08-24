import sys
import time
import os
import datetime
 
# --------------torch等相关------------------#
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
 
# --- 导入自定义模型和工具 ---
sys.path.append('/kaggle/input/unet')             # 填入正确的路径 
from train_utils import dice_loss, build_target
from src import UNet
 
# ---参考：https://space.bilibili.com/18161609?spm_id_from=333.1387.follow.user_card.click---
def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss
    # 检查是否存在辅助输出  ，注释：暂时不可用，若使用到辅助输出可参考EGE-UNet的github
    if 'aux' not in losses:
        return losses['out']
    return losses['out'] + 0.5 * losses['aux']
 
 
# 1. Config配置
class Config:
    # --- 模型和损失函数配置 ---
    num_classes = 2
    use_dice_loss = True # 是否使用dice损失(默认使用交叉熵损失+dice损失)
 
    # --- 数据集和路径配置 ---
    data_path = '/kaggle/input/isic2017-split/train/'
    val_data_path = '/kaggle/input/isic2017-split/test/' 

    # 训练集的mean和std        -2017  
    mean = (0.7060, 0.5888, 0.5449)
    std = (0.0941, 0.1108, 0.1248)
    # 训练集的mean和std        -2018  
    # mean = (0.7082, 0.5799, 0.5343)
    # std = (0.0990, 0.1136, 0.1277)
 
    # ---超参数---
    epochs = 2
    batch_size = 16
    input_size_h = 256
    input_size_w = 256
 
    # ---分布式训练配置---
    distributed = True
    dist_backend = 'nccl'
    dist_url = 'env://'
    sync_bn = True
    workers = 4
    
    # ---训练过程控制---
    output_dir = '/kaggle/working/train_output'
    resume = '' # 若采用分段训练，这里请填入对应的pth文件
    seed = 42
    amp = True
    
    # ---优化器选择---  默认使用AdamW，初始学习率0.001
    opt = 'AdamW'
    if opt == 'AdamW':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-4 
    elif opt == 'SGD':
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.05
    
    # ---学习率调度器选择---  默认使用余弦退火学习率更新策略，参考MALUnet的github
    sch = 'CosineAnnealingLR'
    if sch == 'CosineAnnealingLR':
        # T_max 将在 get_scheduler 中根据 epoch 和 steps 动态计算
        eta_min = 1e-5
    elif sch == 'StepLR':
        step_size = epochs // 5
        gamma = 0.5
 
# 2. 数据预处理           参考：b站博主“霹雳吧啦Wz”  以及 MALUnet
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
 
class JointRandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, img, mask):
        if torch.rand(1) < self.p: return F.hflip(img), F.hflip(mask)
        return img, mask
 
class JointRandomVerticalFlip(T.RandomVerticalFlip):
    def forward(self, img, mask):
        if torch.rand(1) < self.p: return F.vflip(img), F.vflip(mask)
        return img, mask
 
class JointRandomRotation(T.RandomRotation):
    def forward(self, img, mask):
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, self.interpolation, self.expand, self.center, self.fill), \
               F.rotate(mask, angle, T.InterpolationMode.NEAREST, self.expand, self.center, 0)
 
class ToTensor:
    def __call__(self, img, mask):
        img = F.to_tensor(img)
        mask = torch.as_tensor(np.array(mask) // 255, dtype=torch.int64)
        return img, mask
 
class Normalize(T.Normalize):
    def forward(self, img, mask):
        return super().forward(img), mask
 
def get_transforms(config, train=True):
    transform_list = [JointResize([config.input_size_h, config.input_size_w])]
    if train:
        transform_list.extend([
            JointRandomHorizontalFlip(p=0.5),
            JointRandomVerticalFlip(p=0.5),
            JointRandomRotation(degrees=[0, 360]),
        ])
    transform_list.extend([ToTensor(),Normalize(mean=config.mean, std=config.std)])
    return JointTransform(transform_list)
 
# 3. 数据集类
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
        img, mask = self.transforms(img, mask)
        return img, mask
 
    def __len__(self):
        return len(self.image_paths)
 
# 4. 优化器和调度器
def get_optimizer(config, model):
    params = [p for p in model.parameters() if p.requires_grad]
    if config.opt == 'AdamW':
        return torch.optim.AdamW(params, lr=config.lr, betas=config.betas, weight_decay=config.weight_decay, eps=config.eps)
    elif config.opt == 'SGD':
        return torch.optim.SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {config.opt} not implemented")
 
def get_scheduler(config, optimizer, steps_per_epoch):
    if config.sch == 'CosineAnnealingLR':
        T_max = config.epochs * steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=config.eta_min)
    elif config.sch == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    else:
        print(f"Warning: Scheduler '{config.sch}' not implemented. Using no scheduler.")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
 
# 5. DDP
def init_ddp(config):
    if not config.distributed:
        config.rank, config.world_size, config.local_rank = 0, 1, 0
        return
    config.local_rank = int(os.environ["LOCAL_RANK"])
    config.rank = int(os.environ["RANK"])
    config.world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(config.local_rank)
    dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url)
    dist.barrier()
 
def cleanup_ddp():
    if dist.is_initialized(): dist.destroy_process_group()
 
def is_main_process(config): return config.rank == 0
 
def save_on_main_process(data, path, config):
    if is_main_process(config): torch.save(data, path)
 
# 6. 训练和验证循环
def train_one_epoch(model, optimizer, data_loader, device, epoch, config, scaler, scheduler):
    model.train()
    total_loss = 0
    if is_main_process(config):
        data_loader = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")
 
    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(images)
            # mask 值现在是 {0, 1}，255为忽略，参考自b站up主：霹雳吧啦Wz
            loss = criterion(output, targets, num_classes=config.num_classes, dice=config.use_dice_loss, ignore_index=255)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if config.sch in ['CosineAnnealingLR']: scheduler.step()
        total_loss += loss.item()
 
    if config.sch in ['StepLR']: scheduler.step()
    
    avg_loss_tensor = torch.tensor([total_loss / len(data_loader)], device=device)
    if config.distributed: dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
    return avg_loss_tensor.item()
 
@torch.no_grad()
def evaluate(model, data_loader, device, config):
    model.eval()
    # 用于收集所有GPU上的单张图片dice分数
    dice_scores_list = []
    total_loss = 0.0
    num_samples = 0
    for images, targets in data_loader:
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets, num_classes=config.num_classes, dice=config.use_dice_loss, ignore_index=255)
        total_loss += loss.item() * images.size(0)

        num_samples += images.size(0) # num_samples 在DDP下是当前GPU处理的样本数
        preds = torch.argmax(outputs['out'], dim=1)

        # 遍历批次中的每一张图像
        for i in range(preds.shape[0]):
            pred_single = preds[i].cpu().numpy()
            target_single = targets[i].cpu().numpy()
            if np.sum(target_single) == 0 and np.sum(pred_single) == 0:
                dice = 1.0
            elif np.sum(target_single) == 0 and np.sum(pred_single) > 0:
                dice = 0.0
            else:
                # 计算 TP, FP, FN
                TP = np.sum((pred_single == 1) & (target_single == 1))
                FP = np.sum((pred_single == 1) & (target_single == 0))
                FN = np.sum((pred_single == 0) & (target_single == 1))
                dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
            
            dice_scores_list.append(dice)

    if config.distributed:
        # 在DDP模式下，每个GPU都有一份 dice_scores_list，需要汇总
        # 1. 汇总所有GPU的 total_loss 和 num_samples
        loss_tensor = torch.tensor([total_loss], device=device)
        num_tensor = torch.tensor([num_samples], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / num_tensor.item()
        # 2. 汇总所有GPU的 dice_scores_list
        # 创建一个空列表来接收所有GPU的数据
        all_dice_scores = [None] * config.world_size
        dist.all_gather_object(all_dice_scores, dice_scores_list)
 
        # 只有主进程需要计算最终的平均值
        if is_main_process(config):
            final_dice_list = [item for sublist in all_dice_scores for item in sublist]
            avg_dice = np.mean(final_dice_list)
        else:
            # 其他进程不需要返回值，设为0
            avg_dice = 0.0
    else:
        avg_loss = total_loss / num_samples
        avg_dice = np.mean(dice_scores_list)

    return avg_loss, avg_dice
 
# 7. 主函数
def main(config):
    results_file = os.path.join(config.output_dir,
                                "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    init_ddp(config)
    
    if is_main_process(config):
        print(f"Starting DDP training with {config.world_size} GPUs.")
        os.makedirs(config.output_dir, exist_ok=True)
    
    torch.manual_seed(config.seed + config.rank)
    np.random.seed(config.seed + config.rank)
    device = torch.device(config.local_rank if config.distributed else "cuda")
    
    train_dataset = ISICDataset(root_dir=config.data_path, transforms=get_transforms(config, train=True))
    val_dataset = ISICDataset(root_dir=config.val_data_path, transforms=get_transforms(config, train=False))
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if config.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if config.distributed else None
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=config.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler, shuffle=False, num_workers=config.workers, pin_memory=True)

    # -----------------------模型----------------------------
    model = UNet(in_channels=3, num_classes=config.num_classes, base_c=64).to(device)
    
    if config.distributed:
        if config.sync_bn: model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[config.local_rank])
    model_without_ddp = model.module if config.distributed else model
 
    optimizer = get_optimizer(config, model_without_ddp)
    scheduler = get_scheduler(config, optimizer, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler() if config.amp else None
 
    start_epoch, best_dice = 0, 0.0
    best_loss = 999.0
    if config.resume:
        if os.path.isfile(config.resume):
            if is_main_process(config): print(f"=> Loading checkpoint '{config.resume}'")
            map_location = f'cuda:{config.local_rank}' if config.distributed else "cuda"
            checkpoint = torch.load(config.resume, map_location=map_location, weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('best_dice', 0.0)
            best_loss = checkpoint.get('best_loss', 999.0)
            if config.amp and 'scaler' in checkpoint: scaler.load_state_dict(checkpoint['scaler'])
            if is_main_process(config): print(f"=> Loaded checkpoint '{config.resume}' (epoch {checkpoint['epoch']})")
        else:
            if is_main_process(config): print(f"=> No checkpoint found at '{config.resume}'")
    
    if config.distributed: dist.barrier()
 
    start_time = time.time()
    for epoch in range(start_epoch, config.epochs):
        if config.distributed: train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, config, scaler, scheduler)
        val_loss, val_dice = evaluate(model, val_loader, device, config)
        
        if is_main_process(config):
            # print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            with open(results_file, "a") as f:
                f.write(f"Epoch {epoch}\n")
                f.write(f"Train Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}\n")
                f.write(f"Val Loss: {val_loss:.4f}, Val Dice coefficient: {val_dice:.4f}\n\n")
                
            if val_dice > best_dice:
                best_loss = val_loss
                best_dice = val_dice
                best_model_weights = model_without_ddp.state_dict()
                save_on_main_process(best_model_weights, os.path.join(config.output_dir, 'best_unetmodel.pth'), config)
                print(f"** New best model saved with Dice: {best_dice:.6f} **")
            
            save_file = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'best_dice':best_dice
            }
            if config.amp: save_file["scaler"] = scaler.state_dict()
    
            save_on_main_process(save_file, os.path.join(config.output_dir, 'latest.pth'), config)
 
    cleanup_ddp()
    total_time = time.time() - start_time
    if is_main_process(config):
        print(f"Training finished in {datetime.timedelta(seconds=int(total_time))}")
 
if __name__ == "__main__":
    config = Config()
    main(config)