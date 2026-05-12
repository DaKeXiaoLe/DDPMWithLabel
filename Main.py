from Diffusion.Train import train, eval
import os

def main(model_config = None):
    """
    扩散模型的主函数，用于配置和启动训练或评估过程
    Main function for diffusion model, used to configure and start training or evaluation process
    
    Args:
        model_config: 可选的模型配置字典，如果提供则覆盖默认配置
    """
    # 默认模型配置参数
    modelConfig = {
        "state": "eval", # 运行状态："train" 训练模式 或 "eval" 评估模式
        "epoch": 100,     # 训练的总轮数，建议至少10的倍数轮以获得良好结果
        "batch_size": 16, # 批次大小
        # "batch_size": 10, # eval批次可以小一点
        "T": 1000,        # 扩散过程的总时间步数
        "channel": 128,   # 模型的基础通道数
        "channel_mult": [1, 2, 3, 4],  # 通道倍增因子，用于不同分辨率层
        "attn": [2],      # 使用注意力机制的层索引
        "num_res_blocks": 2,  # 每个分辨率层的残差块数量
        "dropout": 0.15,  # Dropout比率，用于防止过拟合
        # 优化器
        "lr": 1e-4,       # 学习率
        # 渐进式预热调度器
        "multiplier": 2., # 学习率倍增因子
        # 高斯扩散训练器
        "beta_1": 1e-4,   # 起始噪声调度参数β，控制噪声添加的起始强度
        "beta_T": 0.02,   # 结束噪声调度参数β，控制噪声添加的最终强度eval

        "img_size": 32,   # 输入图像的尺寸（正方形）
        "grad_clip": 1.,  # 梯度裁剪阈值，防止梯度爆炸
        "device": "cuda", # 训练设备，使用GPU加速（自动使用所有可用GPU）
        
        "training_load_weight": None,  # 训练时加载的预训练权重路径，None表示从头训练
        "save_weight_dir": "./save_weight/",  # 模型权重保存目录
        "test_load_weight": "ckpt_499_.pt",  # 评估时加载的模型权重文件名
        "sampled_dir": "./SampledImgs/",  # 生成图像的保存目录
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",  # 带噪声图像的保存文件名
        "sampledImgName": "SampledNoGuidenceImgs.png",  # 生成图像的保存文件名
        "nrow": 4,  # 图像网格中每行显示的图像数量
        "num_classes": 10  # CIFAR10的类别数量，启用条件生成
        }
    
    # 如果提供了自定义配置，则覆盖默认配置
    if model_config is not None:
        modelConfig = model_config
    
    # 根据配置状态选择训练或评估模式
    if modelConfig["state"] == "train":
        train(modelConfig)  # 启动训练过程
    else:
        eval(modelConfig)   # 启动评估过程


if __name__ == '__main__':
    """
    程序入口点，当直接运行此脚本时执行main函数
    Program entry point, executes main function when script is run directly
    """
    main()