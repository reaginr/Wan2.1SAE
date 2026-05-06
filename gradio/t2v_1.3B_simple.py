#!/usr/bin/env python3
"""
Wan2.1 T2V 1.3B 简化版 Gradio 界面
- 移除提示词扩展功能（不需要额外模型）
- 仅使用 T2V 模型权重
"""

import argparse
import os
import sys
import warnings

import gradio as gr

warnings.filterwarnings('ignore')

# Model
sys.path.insert(0, os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import cache_video

# Global Var
wan_t2v = None
cfg = None


def t2v_generation(prompt, resolution, sd_steps, guide_scale, shift_scale, seed, n_prompt):
    """生成视频"""
    global wan_t2v, cfg

    W = int(resolution.split("*")[0])
    H = int(resolution.split("*")[1])

    print(f"\n生成视频: {prompt[:50]}...")

    video = wan_t2v.generate(
        prompt,
        size=(W, H),
        shift=shift_scale,
        sampling_steps=sd_steps,
        guide_scale=guide_scale,
        n_prompt=n_prompt,
        seed=seed,
        offload_model=True)

    cache_video(
        tensor=video[None],
        save_file="output.mp4",
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1))

    return "output.mp4"


def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                        Wan2.1 (T2V-1.3B) - 简化版
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        无需提示词扩展模型，仅使用T2V权重
                    </div>
                    """)

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="提示词 (Prompt)",
                    placeholder="输入英文提示词描述您想生成的视频",
                    lines=3,
                )

                with gr.Row():
                    resolution = gr.Dropdown(
                        label="分辨率",
                        choices=["832*480", "480*832", "1280*720", "720*1280"],
                        value="832*480")

                with gr.Row():
                    sd_steps = gr.Slider(
                        label="采样步数 (Sampling Steps)",
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1)
                    guide_scale = gr.Slider(
                        label="引导强度 (Guide Scale)",
                        minimum=0,
                        maximum=20,
                        value=6.0,
                        step=0.5)

                with gr.Row():
                    shift_scale = gr.Slider(
                        label="Shift Scale",
                        minimum=0,
                        maximum=20,
                        value=8.0,
                        step=0.5)
                    seed = gr.Number(
                        label="随机种子 (Seed, -1为随机)",
                        value=-1,
                        precision=0)

                n_prompt = gr.Textbox(
                    label="负向提示词 (Negative Prompt)",
                    placeholder="输入负向提示词（可选）",
                    value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    lines=2,
                )

                run_button = gr.Button("生成视频 (Generate)", variant="primary")

            with gr.Column():
                result_video = gr.Video(
                    label='生成的视频', interactive=False, height=480)

        run_button.click(
            fn=t2v_generation,
            inputs=[prompt, resolution, sd_steps, guide_scale, shift_scale, seed, n_prompt],
            outputs=[result_video],
        )

        gr.Markdown("""
                    <div style="text-align: center; font-size: 12px; color: #666;">
                        注意：首次生成需要约20分钟加载模型，后续生成只需约10分钟
                    </div>
                    """)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Wan2.1 T2V 简化版 Gradio 界面")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="T2V模型检查点目录路径")
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio服务端口 (默认: 7860)")

    args = parser.parse_args()

    global wan_t2v, cfg

    print("="*60)
    print("正在加载 Wan2.1 T2V 1.3B 模型...")
    print("首次加载需要约20分钟，请耐心等待")
    print("="*60)

    cfg = WAN_CONFIGS['t2v-1.3B']
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
    )

    print("="*60)
    print("模型加载完成！启动 Gradio 服务...")
    print(f"请在浏览器访问: http://localhost:{args.port}")
    print("="*60)

    demo = gradio_interface()
    demo.launch(server_name="0.0.0.0", share=False, server_port=args.port)


if __name__ == '__main__':
    main()
