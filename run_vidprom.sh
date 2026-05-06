#!/bin/bash
# 生成迷幻艺术风格视频 - 清理后的提示词

# 清理后的提示词（移除Midjourney参数）
PROMPT="Hyper real Psychedelic trip, DEEP DREAM aesthetic, Nam June Paik inspired, Found footage inspired, Hyper resolution, POV you are following a nature sprite into a portal of hyperbolic space. Hyper dimensional feminine. There are towering angelic entities giving off live electricity surrounding a magnetic blue flame. Visual focal point. Your mind intertwines with the collective psychedelic resonance. Hyper psychic beings show you realms of advanced novelty. They emit webs of lightning that illuminate the heavens. You carry out a SHAMANIC ritual with your tribe of twelve. Each being represents a station of the Zodiac. Hyper dimensional linguistic intelligence emerges out of the ether as the true source of cosmic progression. Mirrored lighting: Divine, Zeus, Mount Olympus, Tornado, Hurricane, Pharoah, kaleidoscope, Psylocybin, Mycelial Network, Fly agaric, Octagon, Time Web, Alien spacecraft, Neural network, Quantum Computer, Eclipse, Mirror, Quantum jumping, Cosmic cycle, Ancient past, Far future, Wormhole, high beauty, super symmetry, luminous sfumato, kodak tmax p3200, Surreal, mirage, amnesia"

# 负向提示词（从-neg参数提取）
NEG_PROMPT="dark, poorly drawn, glitch, blurry, 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

cd ~/Wan2.1SAE

nohup python generate.py \
    --task t2v-1.3B \
    --size 832*480 \
    --ckpt_dir ../Wan/Wan2.1-T2V-1.3B \
    --prompt "$PROMPT" \
    --save_file ./vidprom_prompt.mp4 \
    --offload_model True \
    --t5_cpu \
    --sample_shift 8 \
    --sample_guide_scale 6 \
    --frame_num 81 \
    --base_seed 42 \
    > run.log 2>&1 &

echo "=========================================="
echo "迷幻艺术视频生成任务已启动"
echo "=========================================="
echo "输出文件: vidprom_prompt.mp4"
echo "日志文件: run.log"
echo ""
echo "查看日志: tail -f run.log"
echo "=========================================="
