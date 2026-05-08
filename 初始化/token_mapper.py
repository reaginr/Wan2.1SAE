"""
Wan2.1 Token时空映射工具
用于SAE训练和概念提取中的Token采样与定位

使用方法:
    from 初始化.token_mapper import WanTokenMapper

    mapper = WanTokenMapper()
    t, h, w = mapper.idx_to_spatial(10000)
    tokens = mapper.get_frame_tokens(10)
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


class WanTokenMapper:
    """
    Wan2.1 Token时空映射器

    功能:
    1. Token索引 ↔ 时空位置 (t, h, w) 双向映射
    2. 采样特定帧/区域的所有Token
    3. 均匀采样整个时空体积
    """

    def __init__(self,
                 frame_num: int = 81,
                 size_h: int = 480,
                 size_w: int = 832,
                 vae_stride: Tuple[int, int, int] = (4, 8, 8),
                 patch_size: Tuple[int, int, int] = (1, 2, 2)):
        """
        初始化Token映射器

        参数:
            frame_num: 视频帧数 (默认81)
            size_h: 视频高度 (默认480)
            size_w: 视频宽度 (默认832)
            vae_stride: VAE下采样步长 (默认4, 8, 8)
            patch_size: DiT Patch大小 (默认1, 2, 2)
        """
        self.frame_num = frame_num
        self.size_h = size_h
        self.size_w = size_w
        self.vae_stride = vae_stride
        self.patch_size = patch_size

        # 计算Token维度
        self.T = (frame_num - 1) // vae_stride[0] + 1  # 时间token数
        self.H = (size_h // vae_stride[1]) // patch_size[1]  # 高度token数
        self.W = (size_w // vae_stride[2]) // patch_size[2]  # 宽度token数
        self.total_tokens = self.T * self.H * self.W

        # 像素映射
        self.pixel_per_token_h = patch_size[1] * vae_stride[1]
        self.pixel_per_token_w = patch_size[2] * vae_stride[2]

    def idx_to_spatial(self, token_idx: int) -> Tuple[int, int, int]:
        """
        Token索引 → 时空位置

        参数:
            token_idx: Token索引 (0 ~ total_tokens-1)

        返回:
            (t, h, w): 时间、高度、宽度索引
        """
        t = token_idx // (self.H * self.W)
        h = (token_idx % (self.H * self.W)) // self.W
        w = token_idx % self.W
        return t, h, w

    def spatial_to_idx(self, t: int, h: int, w: int) -> int:
        """
        时空位置 → Token索引

        参数:
            t: 时间索引 (0 ~ T-1)
            h: 高度索引 (0 ~ H-1)
            w: 宽度索引 (0 ~ W-1)

        返回:
            token_idx: Token索引
        """
        return t * (self.H * self.W) + h * self.W + w

    def get_frame_tokens(self, t: int) -> List[int]:
        """
        获取第t帧的所有Token索引

        参数:
            t: 帧索引 (0 ~ T-1)

        返回:
            tokens: 该帧所有Token的索引列表
        """
        start = t * self.H * self.W
        return list(range(start, start + self.H * self.W))

    def get_frame_token_range(self, t: int) -> Tuple[int, int]:
        """获取第t帧的Token索引范围"""
        start = t * self.H * self.W
        return start, start + self.H * self.W

    def get_spatial_region(self, t: int, h_range: Tuple[int, int],
                          w_range: Tuple[int, int]) -> List[int]:
        """
        获取特定帧的空间区域Token

        参数:
            t: 帧索引
            h_range: (h_start, h_end) 高度范围
            w_range: (w_start, w_end) 宽度范围

        返回:
            tokens: 该区域所有Token索引
        """
        h_start, h_end = h_range
        w_start, w_end = w_range
        tokens = []
        for h in range(h_start, h_end):
            for w in range(w_start, w_end):
                tokens.append(self.spatial_to_idx(t, h, w))
        return tokens

    def get_center_region(self, t: int, margin: int = 10) -> List[int]:
        """
        获取帧中心区域的Token

        参数:
            t: 帧索引
            margin: 距离中心的边界

        返回:
            tokens: 中心区域Token索引
        """
        h_center = self.H // 2
        w_center = self.W // 2

        h_range = (max(0, h_center - margin), min(self.H, h_center + margin))
        w_range = (max(0, w_center - margin), min(self.W, w_center + margin))

        return self.get_spatial_region(t, h_range, w_range)

    def uniform_sample(self, stride_t: int = 1, stride_h: int = 3,
                       stride_w: int = 4) -> List[int]:
        """
        在整个时空体积上均匀采样Token

        参数:
            stride_t: 时间采样步长
            stride_h: 高度采样步长
            stride_w: 宽度采样步长

        返回:
            tokens: 采样得到的Token索引列表
        """
        tokens = []
        for t in range(0, self.T, stride_t):
            for h in range(0, self.H, stride_h):
                for w in range(0, self.W, stride_w):
                    tokens.append(self.spatial_to_idx(t, h, w))
        return tokens

    def uniform_sample_frames(self, num_frames: int = 5) -> Tuple[List[int], List[int]]:
        """
        均匀采样多个完整帧

        参数:
            num_frames: 采样帧数

        返回:
            frame_indices: 采样的帧索引
            tokens: 所有采样Token
        """
        step = max(1, self.T // num_frames)
        frame_indices = list(range(0, self.T, step))[:num_frames]

        all_tokens = []
        for t in frame_indices:
            all_tokens.extend(self.get_frame_tokens(t))

        return frame_indices, all_tokens

    def token_to_pixels(self, token_idx: int) -> Tuple[int, int, int, int, int]:
        """
        Token索引 → 原始视频像素范围

        返回:
            (t_frame, h_start, h_end, w_start, w_end): 帧索引和像素范围
        """
        t, h, w = self.idx_to_spatial(token_idx)

        # Latent坐标
        t_latent = t * self.patch_size[0]
        h_latent = h * self.patch_size[1]
        w_latent = w * self.patch_size[2]

        # 像素坐标
        t_frame = t_latent * self.vae_stride[0]
        h_start = h_latent * self.vae_stride[1]
        w_start = w_latent * self.vae_stride[2]

        h_end = min(h_start + self.pixel_per_token_h, self.size_h)
        w_end = min(w_start + self.pixel_per_token_w, self.size_w)

        return t_frame, h_start, h_end, w_start, w_end

    def get_info(self) -> dict:
        """获取Token映射信息"""
        return {
            "video_size": f"{self.frame_num}×{self.size_h}×{self.size_w}",
            "token_dims": f"T={self.T}, H={self.H}, W={self.W}",
            "total_tokens": self.total_tokens,
            "tokens_per_frame": self.H * self.W,
            "pixel_per_token": f"{self.pixel_per_token_h}×{self.pixel_per_token_w}",
        }

    def __repr__(self):
        info = self.get_info()
        return f"WanTokenMapper({info['video_size']}, tokens={info['total_tokens']})"


# 预设配置
DEFAULT_480P = WanTokenMapper(frame_num=81, size_h=480, size_w=832)
DEFAULT_720P = WanTokenMapper(frame_num=81, size_h=720, size_w=1280)


if __name__ == "__main__":
    # 测试
    mapper = WanTokenMapper()

    print("=" * 60)
    print("Wan2.1 Token映射器测试")
    print("=" * 60)

    print(f"\n{mapper}")
    print(f"详细信息: {mapper.get_info()}")

    # 测试映射
    print(f"\n【测试1】Token索引 → 时空位置")
    test_indices = [0, 100, 1000, 10000, 32759]
    for idx in test_indices:
        t, h, w = mapper.idx_to_spatial(idx)
        back_idx = mapper.spatial_to_idx(t, h, w)
        print(f"  Token {idx:5d} → (t={t:2d}, h={h:2d}, w={w:2d}) → {back_idx} {'✓' if idx == back_idx else '✗'}")

    # 测试帧采样
    print(f"\n【测试2】帧采样")
    frame_10_tokens = mapper.get_frame_tokens(10)
    print(f"  第10帧Token范围: [{frame_10_tokens[0]}, {frame_10_tokens[-1]}]")
    print(f"  第10帧Token数: {len(frame_10_tokens)}")

    # 测试均匀采样
    print(f"\n【测试3】均匀采样")
    sampled = mapper.uniform_sample(stride_t=2, stride_h=5, stride_w=5)
    print(f"  采样Token数: {len(sampled)}")
    print(f"  采样比例: {len(sampled)/mapper.total_tokens*100:.1f}%")

    # 测试像素映射
    print(f"\n【测试4】Token → 像素范围")
    for idx in [0, 100, 1000]:
        t_frame, h_start, h_end, w_start, w_end = mapper.token_to_pixels(idx)
        print(f"  Token {idx:5d} → 帧{t_frame}, 像素H[{h_start},{h_end}), W[{w_start},{w_end})")

    print("\n" + "=" * 60)
