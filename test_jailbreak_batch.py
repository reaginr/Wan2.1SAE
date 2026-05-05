"""
Wan2.1 T2V 越狱攻击批量测试脚本
用于测试模型对风险提示词的生成响应
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from wan.configs import wan_t2v_1_3B
from wan.text2video import WanT2V
from wan.modules.vae import WanVAE


def load_model(checkpoint_dir="./Wan2.1-T2V-1.3B"):
    """加载Wan2.1 T2V 1.3B模型"""
    print("正在加载模型...")

    cfg = wan_t2v_1_3B.t2v_1_3B
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WanT2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
    )
    print(f"模型加载完成，使用设备: {device}")
    return model


def read_prompt_pairs(file_path="final_cleaned/prompt_1.txt"):
    """读取提示词对（奇数行为风险，偶数行为安全）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 配对：奇数行(0,2,4...)为风险，偶数行(1,3,5...)为安全
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        risk_prompt = lines[i]
        safe_prompt = lines[i + 1]
        pairs.append({
            'id': i // 2 + 1,
            'risk': risk_prompt,
            'safe': safe_prompt
        })
    return pairs


def generate_video(model, prompt, output_path, size=(832, 480), frame_num=81):
    """生成视频"""
    print(f"\n生成视频: {output_path}")
    print(f"提示词: {prompt[:80]}...")

    video = model.generate(
        input_prompt=prompt,
        size=size,
        frame_num=frame_num,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=30,
        guide_scale=5.0,
        n_prompt="",  # 使用默认负向提示词
        seed=-1,
        offload_model=True
    )

    if video is not None:
        # 保存视频 (video是torch tensor)
        import torchvision.io as io
        # 转换为[frames, height, width, channels]格式
        video = video.permute(1, 2, 3, 0)  # [C, F, H, W] -> [F, H, W, C]
        video = (video * 255).clamp(0, 255).byte()
        io.write_video(output_path, video.cpu(), fps=16)
        print(f"视频已保存: {output_path}")
        return True
    else:
        print("生成失败")
        return False


def test_jailbreak(model, prompt_pairs, output_dir="jailbreak_test_output", num_tests=5):
    """测试越狱攻击"""
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for i, pair in enumerate(prompt_pairs[:num_tests], 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{num_tests}")
        print(f"{'='*60}")

        # 测试风险提示词
        risk_output = os.path.join(output_dir, f"test_{i:03d}_risk.mp4")
        risk_success = generate_video(model, pair['risk'], risk_output)

        # 测试安全提示词（对照）
        safe_output = os.path.join(output_dir, f"test_{i:03d}_safe.mp4")
        safe_success = generate_video(model, pair['safe'], safe_output)

        results.append({
            'id': pair['id'],
            'risk_prompt': pair['risk'],
            'safe_prompt': pair['safe'],
            'risk_output': risk_output if risk_success else None,
            'safe_output': safe_output if safe_success else None,
        })

    return results


def print_summary(results):
    """打印测试结果摘要"""
    print(f"\n{'='*60}")
    print("测试结果摘要")
    print(f"{'='*60}")

    for r in results:
        print(f"\n测试 {r['id']}:")
        print(f"  风险提示词: {r['risk_prompt'][:60]}...")
        print(f"  生成结果: {'成功' if r['risk_output'] else '失败'}")
        print(f"  安全提示词: {r['safe_prompt'][:60]}...")
        print(f"  生成结果: {'成功' if r['safe_output'] else '失败'}")


def main():
    """主函数"""
    # 检查是否有GPU
    if not torch.cuda.is_available():
        print("警告: 没有检测到GPU，生成将非常慢")

    # 加载模型
    model = load_model()

    # 读取提示词对
    prompt_pairs = read_prompt_pairs()
    print(f"读取了 {len(prompt_pairs)} 对提示词")

    # 运行测试（默认测试前5对）
    num_tests = int(input("请输入要测试的提示词对数量 (默认5): ") or "5")

    results = test_jailbreak(model, prompt_pairs, num_tests=num_tests)

    # 打印摘要
    print_summary(results)

    print(f"\n测试完成！视频保存在: jailbreak_test_output/")


if __name__ == "__main__":
    main()
