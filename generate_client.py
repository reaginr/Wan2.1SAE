#!/usr/bin/env python3
"""
Wan2.1 T2V 生成服务客户端
简化版：直接调用服务生成视频
"""

import os
import sys
import time
import argparse
from datetime import datetime

# 临时方案：直接在客户端中复用模型
# 实际使用时应该通过API/队列等方式与服务通信

import torch
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import cache_video

# 全局模型（首次加载后复用）
_model = None
_cfg = None


def get_model(ckpt_dir):
    """获取或加载模型"""
    global _model, _cfg

    if _model is None:
        print("首次加载模型，请等待约20分钟...")
        _cfg = WAN_CONFIGS['t2v-1.3B']
        _model = wan.WanT2V(
            config=_cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_cpu=False,
            offload_model=True,
        )
        print("模型加载完成！")
    else:
        print("复用已加载的模型")

    return _model, _cfg


def generate(ckpt_dir, prompt, save_file=None, **kwargs):
    """生成视频"""
    model, cfg = get_model(ckpt_dir)

    # 默认参数
    params = {
        'size': (832, 480),
        'frame_num': 81,
        'shift': 8.0,
        'sampling_steps': 50,
        'guide_scale': 6.0,
        'seed': -1,
        'offload_model': True,
    }
    params.update(kwargs)

    # 自动生成文件名
    if save_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = prompt.replace(' ', '_')[:30]
        save_file = f"output/{safe_prompt}_{timestamp}.mp4"

    os.makedirs(os.path.dirname(save_file) if os.path.dirname(save_file) else '.', exist_ok=True)

    print(f"\n生成视频: {prompt[:50]}...")
    start = time.time()

    video = model.generate(
        prompt,
        size=params['size'],
        frame_num=params['frame_num'],
        shift=params['shift'],
        sampling_steps=params['sampling_steps'],
        guide_scale=params['guide_scale'],
        seed=params['seed'],
        offload_model=params['offload_model'],
    )

    cache_video(
        tensor=video[None],
        save_file=save_file,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )

    elapsed = time.time() - start
    print(f"完成！保存到: {save_file}")
    print(f"耗时: {elapsed/60:.1f}分钟")

    return save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='../Wan/Wan2.1-T2V-1.3B',
                        help='模型路径')
    parser.add_argument('--prompt', type=str, required=True,
                        help='提示词')
    parser.add_argument('--save_file', type=str,
                        help='保存路径')
    parser.add_argument('--size', type=str, default='832*480',
                        help='分辨率，如 832*480')
    parser.add_argument('--frame_num', type=int, default=81,
                        help='帧数')
    parser.add_argument('--seed', type=int, default=-1,
                        help='随机种子')

    args = parser.parse_args()

    # 解析size
    w, h = args.size.split('*')
    size = (int(w), int(h))

    # 生成
    generate(
        ckpt_dir=args.ckpt_dir,
        prompt=args.prompt,
        save_file=args.save_file,
        size=size,
        frame_num=args.frame_num,
        seed=args.seed,
    )


if __name__ == '__main__':
    # 示例：连续生成多个视频，模型只加载一次
    if len(sys.argv) == 1:
        print("使用方法:")
        print("  python generate_client.py --prompt 'your prompt'")
        print("  python generate_client.py --prompt 'prompt1' --save_file out1.mp4")
        print("\n或者直接在Python中调用:")
        print("  from generate_client import generate, get_model")
        print("  get_model('../Wan/Wan2.1-T2V-1.3B')  # 加载模型")
        print("  generate(None, 'prompt1')  # 生成第一个")
        print("  generate(None, 'prompt2')  # 生成第二个（复用模型）")
        sys.exit(0)

    main()
