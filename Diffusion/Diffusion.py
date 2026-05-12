
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import numpy as np


def extract(v, t, x_shape):
    """
    从给定的张量v中提取指定时间步t对应的系数，并重塑为适合广播的形状
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    
    Args:
        v: 系数张量，形状为[T, ...]，包含所有时间步的系数
        t: 时间步张量，形状为[batch_size]，包含每个样本的时间步索引
        x_shape: 目标张量的形状，用于确定广播所需的维度
    
    Returns:
        重塑后的系数张量，形状为[batch_size, 1, 1, ...]
    """
    device = t.device  # 获取时间步张量所在的设备
    # 使用torch.gather从v中收集对应时间步t的系数
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # 重塑为[batch_size, 1, 1, ...]的形状以便广播
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    """
    高斯扩散训练器，用于训练扩散模型的前向加噪过程
    Gaussian Diffusion Trainer for training the forward diffusion process
    """
    def __init__(self, model, beta_1, beta_T, T):
        """
        初始化扩散训练器
        Initialize the diffusion trainer
        
        Args:
            model: 噪声预测模型，输入为加噪图像和时间步，输出为预测的噪声
            beta_1: 起始噪声调度参数
            beta_T: 终止噪声调度参数
            T: 总时间步数
        """
        super().__init__()

        self.model = model  # 噪声预测模型
        self.T = T  # 总时间步数

        # 注册噪声调度参数β，从beta_1线性增加到beta_T,T等分。
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas  # α = 1 - β
        alphas_bar = torch.cumprod(alphas, dim=0)  # ᾱ_t = ∏_{i=1}^t α_i，累积乘积

        # 计算扩散过程 q(x_t | x_{t-1}) 和其他相关参数
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))  # √(ᾱ_t)
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))  # √(1-ᾱ_t)

    def forward(self, x_0, labels=None):
        """
        前向传播：执行扩散过程并计算损失
        Algorithm 1: Forward diffusion process and loss computation
        
        Args:
            x_0: 原始图像张量，形状为[batch_size, channels, height, width]
            labels: 类别标签张量，形状为[batch_size]，用于条件生成
            
        Returns:
            loss: 噪声预测损失
        """
        # 为每个样本随机采样一个时间步，范围[0, T-1]，x_0.shape[0]为bantch_size
        # print(torch.randint(1000, size=(3, ))) # tensor([888, 751, 475])
        # 对batch图片分别进行不同时间长度的模糊处理。
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        # 生成与x_0形状相同的标准高斯噪声
        noise = torch.randn_like(x_0)   # ε ~ N(0, I)
        # 根据扩散公式计算加噪后的图像：x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε,
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +  #  √(ᾱ_t) * x_0
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)  # √(1-ᾱ_t) * ε
        '''
        extract:从预计算数组中根据时间步提取对应值，并调整形状以匹配目标张量
        Args:
            arr: 预计算的一维张量，包含所有时间步的参数值 [T]
            timesteps: 时间步索引，形状为 [batch_size]
            target_shape: 目标形状，通常是 x_0 的形状 [batch_size, channels, height, width]
        Returns:
            调整形状后的张量，便于广播运算
        '''
        
        # 计算模型预测噪声与真实噪声之间的均方误差损失
        if labels is not None:
            # 条件生成：使用标签信息
            predicted_noise = self.model(x_t, t, labels)
        else:
            # 无条件生成：使用默认标签（例如类别0）
            default_labels = torch.zeros(x_0.shape[0], dtype=torch.long, device=x_0.device)
            predicted_noise = self.model(x_t, t, default_labels)
            
        loss = F.mse_loss(predicted_noise, noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    """
    高斯扩散采样器，用于从噪声中生成图像的反向去噪过程
    Gaussian Diffusion Sampler for generating images from noise through reverse denoising process
    """
    def __init__(self, model, beta_1, beta_T, T):
        """
        初始化扩散采样器
        Initialize the diffusion sampler
        
        Args:
            model: 噪声预测模型，与训练器使用相同的模型
            beta_1: 起始噪声调度参数
            beta_T: 终止噪声调度参数
            T: 总时间步数
        """
        super().__init__()

        self.model = model  # 噪声预测模型
        self.T = T  # 总时间步数

        # 注册噪声调度参数β，从beta_1线性增加到beta_T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas  # α = 1 - β
        alphas_bar = torch.cumprod(alphas, dim=0)  # α_bar = ∏α，累积乘积
        # α_bar_prev = [1, α_bar_0, α_bar_1, ..., α_bar_{T-1}]
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # 计算反向过程的系数
        # coeff1 = √(1/α_t)，用于计算均值
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        # coeff2 = coeff1 * (1 - α_t) / √(1 - α_bar_t)，用于计算均值
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        # 计算后验分布的方差：β_t * (1 - α_bar_{t-1}) / (1 - α_bar_t)
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        """
        根据预测的噪声ε计算x_{t-1}的均值
        Compute the mean of x_{t-1} given the predicted noise ε
        
        Args:
            x_t: 当前时间步的加噪图像
            t: 当前时间步
            eps: 模型预测的噪声
            
        Returns:
            x_{t-1}的均值
        """
        assert x_t.shape == eps.shape  # 确保x_t和eps形状相同
        # 根据DDPM的反向过程公式：μ_θ(x_t, t) = (1/√α_t) * (x_t - (1-α_t)/√(1-α_bar_t) * ε_θ)
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -  # coeff1 * x_t
            extract(self.coeff2, t, x_t.shape) * eps    # coeff2 * eps
        )

    def p_mean_variance(self, x_t, t, labels=None):
        """
        计算反向过程中x_{t-1}的均值和方差
        Compute the mean and variance for x_{t-1} in the reverse process
        
        Args:
            x_t: 当前时间步的加噪图像
            t: 当前时间步
            labels: 类别标签张量，形状为[batch_size]，用于条件生成
            
        Returns:
            xt_prev_mean: x_{t-1}的均值
            var: x_{t-1}的方差
        """
        # below: only log_variance is used in the KL computations
        # 构建方差张量：第一个时间步使用posterior_var[1]，其他时间步使用betas[1:]
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)  # 提取对应时间步的方差

        # 使用模型预测噪声，支持条件生成
        if labels is not None:
            eps = self.model(x_t, t, labels)  # 条件生成：使用标签信息
        else:
            # 无条件生成：使用默认标签（例如类别0）
            default_labels = torch.zeros(x_t.shape[0], dtype=torch.long, device=x_t.device)
            eps = self.model(x_t, t, default_labels)  # 无条件生成
            
        # 根据预测的噪声计算x_{t-1}的均值
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T, labels=None):
        """
        前向传播：执行反向去噪过程生成图像
        Algorithm 2: Reverse denoising process for image generation
        
        Args:
            x_T: 初始噪声张量，形状为[batch_size, channels, height, width]
            labels: 类别标签张量，形状为[batch_size]，用于条件生成
            
        Returns:
            x_0: 生成的图像，值被裁剪到[-1, 1]范围内
        """
        x_t = x_T  # 从纯噪声开始
        # 从T-1到0反向遍历所有时间步
        for time_step in reversed(range(self.T)):
            # 显示进度条：当前时间步/总时间步
            print(f"\r去噪进度: {self.T - time_step}/{self.T}", end="", flush=True)
            # 创建与批次大小相同的时间步张量，所有元素都为当前时间步
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            # 计算x_{t-1}的均值和方差，支持条件生成
            mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            # 当t=0时不添加噪声，其他时间步添加随机噪声
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)  # 生成随机噪声
            else:
                noise = 0  # 最后一个时间步不添加噪声
            # 根据反向过程公式：x_{t-1} = μ_θ(x_t, t) + σ_t * z
            x_t = mean + torch.sqrt(var) * noise
            # 检查张量中是否有NaN值
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

            # 保存每一个t的图片
            # save_image(torch.clip(x_t, -1, 1) * 0.5 + 0.5,
            #             os.path.join(
            #                         "./SampledImgs/",
            #                         f"sampledImg_{int(self.T - time_step)}.png"),
            #                         nrow=8)

        x_0 = x_t  # 最终得到去噪后的图像x_0
        return torch.clip(x_0, -1, 1)   # 将图像值裁剪到[-1, 1]范围内


