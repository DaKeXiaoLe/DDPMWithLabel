import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torchviz import make_dot
from Diffusion.Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler


def setup_multigpu_training():
    """设置多GPU训练环境"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if world_size > 1:
        print(f"数据并行训练: 使用 {world_size} 个GPU")
    else:
        print("单GPU训练")
    
    return device, world_size


def train(modelConfig: Dict):
    # 设置多GPU训练
    device, world_size = setup_multigpu_training()
    
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    # 数据加载器
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True)

    # model setup - 模型设置部分
    # 创建UNet模型，用于噪声预测
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"])
    
    # 多GPU设置 - 使用DataParallel
    if world_size > 1:
        # 数据并行
        net_model = torch.nn.DataParallel(net_model)
        print(f"模型已包装为DataParallel，使用 {world_size} 个GPU")
    
    net_model = net_model.to(device)
    
    # 如果提供了预训练权重，则加载权重
    if modelConfig["training_load_weight"] is not None:
        checkpoint = torch.load(os.path.join(modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device)
        # 处理DataParallel包装的模型权重
        if isinstance(net_model, torch.nn.DataParallel):
            net_model.module.load_state_dict(checkpoint)
        else:
            net_model.load_state_dict(checkpoint)
    
    # 使用AdamW优化器，结合权重衰减防止过拟合
    optimizer = torch.optim.AdamW(
                        net_model.parameters(),  # 优化模型的所有参数
                        lr=modelConfig["lr"],    # 学习率
                        weight_decay=1e-4)       # 权重衰减系数
    
    # 余弦退火学习率调度器，在训练过程中逐渐降低学习率
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer=optimizer,
                            T_max=modelConfig["epoch"],  # 余弦周期的长度（总训练轮数）
                            eta_min=0,                   # 最小学习率
                            last_epoch=-1)               # 从初始学习率开始
    
    # 渐进式预热调度器，在训练开始时逐渐增加学习率
    warmUpScheduler = GradualWarmupScheduler(
                            optimizer=optimizer,
                            multiplier=modelConfig["multiplier"],  # 学习率倍增因子
                            warm_epoch=modelConfig["epoch"] // 10, # 预热轮数（总轮数的10%）
                            after_scheduler=cosineScheduler)       # 预热后使用余弦退火调度器
    
    # 创建高斯扩散训练器，用于训练扩散模型的前向过程
    trainer = GaussianDiffusionTrainer(
                    net_model,              # 噪声预测模型
                    modelConfig["beta_1"],  # 起始噪声调度参数
                    modelConfig["beta_T"],  # 终止噪声调度参数
                    modelConfig["T"]).to(device)  # 总时间步数

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device)
                loss = trainer(x_0, labels).sum() / float(modelConfig["epoch"])
                loss.backward()
                # 将模型所有参数的梯度范数限制在一个指定的阈值范围内。
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(),
                    modelConfig["grad_clip"])
                optimizer.step()
                # 实时更新训练进度条的显示信息
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": f"{e}/{modelConfig['epoch']}",
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "labels shape: ": labels.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        # 更新学习率调度器
        warmUpScheduler.step()
        
        # 保存模型权重（处理DataParallel包装）
        if isinstance(net_model, torch.nn.DataParallel):
            state_dict = net_model.module.state_dict()
        else:
            state_dict = net_model.state_dict()
        
        # 确保保存目录存在
        os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
            
        # torch.save(state_dict, os.path.join(
        #     modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
                    modelConfig["save_weight_dir"],
                    modelConfig["test_load_weight"]),
                    map_location=device)
        
        # 处理可能的多GPU模型权重
        if any(key.startswith('module.') for key in ckpt.keys()):
            # 如果权重来自DataParallel模型，需要移除'module.'前缀
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(ckpt)
            
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
                        model,
                        modelConfig["beta_1"],
                        modelConfig["beta_T"],
                        modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        if isinstance(modelConfig["img_size"], int):
            size = [modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]]
        elif isinstance(modelConfig["img_size"], tuple):
            size = [modelConfig["batch_size"], 3, modelConfig["img_size"][0], modelConfig["img_size"][1]]
        else:
            raise ValueError("img_size must be either int or tuple")
        
        noisyImage = torch.randn(size=size, device=device)
        # 将标准正态分布的噪声图像转换为可视化的图像格式
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        
        # 确保采样目录存在
        os.makedirs(modelConfig["sampled_dir"], exist_ok=True)
        
        save_image(saveNoisy, os.path.join(
                                    modelConfig["sampled_dir"],
                                    modelConfig["sampledNoisyImgName"]),
                                    nrow=modelConfig["nrow"])
        
        # 条件生成：为每个样本生成对应的标签
        if "num_classes" in modelConfig:
            labels = torch.randint(modelConfig["num_classes"], (modelConfig["batch_size"],), device=device)
            # CIFAR10类别名称
            classes = ('plane', 'car', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck')
            
            # 保存标签到txt文件（包含数字标签和类别名称）
            labels_file = os.path.join(modelConfig["sampled_dir"], "labels.txt")
            with open(labels_file, 'w') as f:
                for i, label in enumerate(labels.cpu().numpy()):
                    class_name = classes[label] if label < len(classes) else f"Unknown_{label}"
                    f.write(f"Sample {i}: {label} ({class_name})\n")
            print(f"标签已保存到: {labels_file}")

            sampledImgs = sampler(noisyImage, labels)
        else:
            print("无条件生成")
            # 无条件生成
            sampledImgs = sampler(noisyImage)
            
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
                                    modelConfig["sampled_dir"],
                                    modelConfig["sampledImgName"]),
                                    nrow=modelConfig["nrow"])
        