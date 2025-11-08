import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def plot_training_curves(train_losses, val_losses, save_path=None, show=True):
    """
    绘制训练和验证损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 图像保存路径，None表示不保存
        show: 是否显示图像
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_attention_weights(attention_weights, src_tokens, tgt_tokens, layer=0, head=0, save_path=None, show=True):
    """
    绘制注意力权重热力图
    
    Args:
        attention_weights: 注意力权重 [batch_size, n_heads, tgt_len, src_len]
        src_tokens: 源语言标记列表
        tgt_tokens: 目标语言标记列表
        layer: 要可视化的层索引
        head: 要可视化的头索引
        save_path: 图像保存路径，None表示不保存
        show: 是否显示图像
    """
    # 提取指定层和头的注意力权重
    attn = attention_weights[0, head].cpu().numpy()
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attn, cmap='viridis')
    
    # 设置坐标轴标签
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90)
    ax.set_yticklabels(tgt_tokens)
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # 设置标题
    ax.set_title(f"Attention weights (Layer {layer+1}, Head {head+1})")
    
    # 调整布局
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Attention weights plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def save_training_stats(train_losses, val_losses, save_path):
    """
    保存训练统计数据到文本文件
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    with open(save_path, 'w') as f:
        f.write("Epoch,Train Loss,Validation Loss\n")
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{i+1},{train_loss:.6f},{val_loss:.6f}\n")
    
    print(f"Training statistics saved to {save_path}")

def create_results_directory(base_dir="results"):
    """
    创建结果目录
    
    Args:
        base_dir: 基础目录名
        
    Returns:
        result_dir: 创建的结果目录路径
    """
    # 确保基础目录存在
    os.makedirs(base_dir, exist_ok=True)
    
    # 查找当前最大的实验编号
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("exp_")]
    if existing_dirs:
        max_exp_num = max([int(d.split("_")[1]) for d in existing_dirs])
        new_exp_num = max_exp_num + 1
    else:
        new_exp_num = 1
    
    # 创建新的实验目录
    result_dir = os.path.join(base_dir, f"exp_{new_exp_num}")
    os.makedirs(result_dir, exist_ok=True)
    
    return result_dir