"""
可视化Patch的时空结构

这个脚本帮助理解DiT中patch的概念
"""

import numpy as np

# 配置参数 (来自 wan_t2v_1_3B.py)
frame_num = 81          # 输入帧数
vae_stride_t = 4        # VAE时间压缩率
vae_stride_h = 8        # VAE高度压缩率
vae_stride_w = 8        # VAE宽度压缩率
patch_t = 1             # Patch时间大小
patch_h = 2             # Patch空间高度
patch_w = 2             # Patch空间宽度
size_h = 480            # 输入高度
size_w = 832            # 输入宽度

# 计算
print("=" * 60)
print("Patch结构分析")
print("=" * 60)

# VAE编码后的latent尺寸
t_lat = (frame_num - 1) // vae_stride_t + 1  # (81-1)/4 + 1 = 21
h_lat = size_h // vae_stride_h                # 480/8 = 60
w_lat = size_w // vae_stride_w                # 832/8 = 104

print(f"\n1. VAE编码后 (Latent空间):")
print(f"   时间: {t_lat}帧 (原{frame_num}帧 / {vae_stride_t})")
print(f"   高度: {h_lat} (原{size_h} / {vae_stride_h})")
print(f"   宽度: {w_lat} (原{size_w} / {vae_stride_w})")
print(f"   形状: [{t_lat}, {h_lat}, {w_lat}]")

# Patch划分
n_patches_t = t_lat // patch_t  # 21/1 = 21
n_patches_h = h_lat // patch_h  # 60/2 = 30
n_patches_w = w_lat // patch_w  # 104/2 = 52

print(f"\n2. Patch划分:")
print(f"   Patch大小: [{patch_t}帧, {patch_h}高, {patch_w}宽]")
print(f"   时间方向: {n_patches_t}个patches ({t_lat}/{patch_t})")
print(f"   高度方向: {n_patches_h}个patches ({h_lat}/{patch_h})")
print(f"   宽度方向: {n_patches_w}个patches ({w_lat}/{patch_w})")

# 总token数
total_tokens = n_patches_t * n_patches_h * n_patches_w
print(f"\n3. 总Token数:")
print(f"   {n_patches_t} × {n_patches_h} × {n_patches_w} = {total_tokens}")

# 每个patch对应的原始视频区域
original_frames = patch_t * vae_stride_t  # 1×4 = 4帧
original_h = patch_h * vae_stride_h        # 2×8 = 16像素
original_w = patch_w * vae_stride_w        # 2×8 = 16像素

print(f"\n4. 每个Patch对应的原始视频区域:")
print(f"   时间: {original_frames}帧 ({patch_t}×{vae_stride_t})")
print(f"   高度: {original_h}像素 ({patch_h}×{vae_stride_h})")
print(f"   宽度: {original_w}像素 ({patch_w}×{vae_stride_w})")
print(f"   体积: {original_frames}×{original_h}×{original_w} = {original_frames*original_h*original_w} 像素")

# 时空感受野示例
print(f"\n5. 举例说明:")
print(f"   Token #0:  视频第0-3帧, 画面左上角 16×16 区域")
print(f"   Token #52: 视频第0-3帧, 画面右上角 16×16 区域 (第2个空间列)")
print(f"   Token #{n_patches_h*n_patches_w}: 视频第4-7帧, 画面左上角 (第2个时间步)")

print(f"\n6. SAE处理:")
print(f"   输入: [32760, 1536] - 32760个token, 每个1536维")
print(f"   SAE编码: [32760, 6144] - 每个token变成6144维稀疏向量")
print(f"   Top-K稀疏: 每行只有64个非零值")

print("=" * 60)
