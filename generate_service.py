#!/usr/bin/env python3
"""
Wan2.1 T2V 生成服务 - 模型常驻内存版
解决：每次运行generate.py都要重新加载模型的问题

使用方法：
1. 启动服务：python generate_service.py --ckpt_dir ../Wan/Wan2.1-T2V-1.3B
2. 发送请求：echo "your prompt" > /tmp/prompt.txt
3. 查看结果：视频将保存到 output/ 目录

或者使用提供的客户端脚本：python generate_client.py "your prompt"
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import cache_video

# 全局模型实例（常驻内存）
wan_t2v = None
cfg = None


def load_model(ckpt_dir):
    """加载模型到内存（只执行一次）"""
    global wan_t2v, cfg

    print("="*60)
    print("正在加载模型到内存（首次加载需要约20分钟）...")
    print("="*60)

    cfg = WAN_CONFIGS['t2v-1.3B']
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,  # T5放在GPU上更快
        offload_model=True,
    )

    print("="*60)
    print("模型加载完成！服务已就绪")
    print("="*60)
    return wan_t2v


def generate_video(prompt, save_file=None, **kwargs):
    """生成视频（复用已加载的模型）"""
    global wan_t2v, cfg

    if wan_t2v is None:
        raise RuntimeError("模型未加载，请先调用load_model()")

    print(f"\n{'='*60}")
    print(f"生成视频 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"提示词: {prompt}")

    # 默认参数
    default_params = {
        'size': (832, 480),
        'frame_num': 81,
        'shift': 8.0,
        'sampling_steps': 50,
        'guide_scale': 6.0,
        'seed': -1,
        'offload_model': True,
    }
    default_params.update(kwargs)

    # 自动生成文件名
    if save_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = prompt.replace(' ', '_')[:50]
        save_file = f"output/{safe_prompt}_{timestamp}.mp4"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # 生成视频
    start_time = time.time()

    video = wan_t2v.generate(
        prompt,
        size=default_params['size'],
        frame_num=default_params['frame_num'],
        shift=default_params['shift'],
        sampling_steps=default_params['sampling_steps'],
        guide_scale=default_params['guide_scale'],
        seed=default_params['seed'],
        offload_model=default_params['offload_model'],
    )

    # 保存视频
    cache_video(
        tensor=video[None],
        save_file=save_file,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )

    elapsed = time.time() - start_time
    print(f"\n视频已保存: {save_file}")
    print(f"生成耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")

    return save_file


def interactive_mode(ckpt_dir):
    """交互模式：持续接收用户输入"""
    load_model(ckpt_dir)

    print("\n" + "="*60)
    print("交互模式启动！")
    print("输入提示词生成视频，输入 'quit' 退出")
    print("="*60 + "\n")

    while True:
        try:
            prompt = input("\n提示词 > ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("退出服务")
                break

            if not prompt:
                continue

            generate_video(prompt)

        except KeyboardInterrupt:
            print("\n退出服务")
            break
        except Exception as e:
            print(f"错误: {e}")
            continue


def batch_mode(ckpt_dir, prompt_file, output_dir="output"):
    """批量模式：从文件读取提示词批量生成"""
    load_model(ckpt_dir)

    # 读取提示词文件
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"\n批量生成 {len(prompts)} 个视频...")

    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] 生成中...")
        save_file = os.path.join(output_dir, f"video_{i:03d}.mp4")
        try:
            generate_video(prompt, save_file=save_file)
        except Exception as e:
            print(f"生成失败: {e}")
            continue

    print(f"\n批量生成完成！视频保存在: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Wan2.1 T2V 生成服务')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='模型检查点目录')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'batch'],
                        help='运行模式')
    parser.add_argument('--prompt_file', type=str,
                        help='批量模式：提示词文件路径')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录')

    args = parser.parse_args()

    if args.mode == 'interactive':
        interactive_mode(args.ckpt_dir)
    elif args.mode == 'batch':
        if not args.prompt_file:
            print("错误: 批量模式需要 --prompt_file 参数")
            sys.exit(1)
        batch_mode(args.ckpt_dir, args.prompt_file, args.output_dir)


if __name__ == '__main__':
    main()
