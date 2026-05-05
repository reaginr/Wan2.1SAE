"""
修复 Wan2.1 视频保存错误的补丁脚本
问题: cache_video 中 Float 无法转换为 Byte
解决: 在转换前确保 tensor 为 float32 类型
"""

import os
import sys

# 读取原文件
utils_file = "wan/utils/utils.py"

with open(utils_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 找到需要修改的行并修复
# 原代码: tensor = (tensor * 255).type(torch.uint8).cpu()
# 修复后: tensor = (tensor * 255).to(torch.float32).clamp(0, 255).to(torch.uint8).cpu()

old_line = "tensor = (tensor * 255).type(torch.uint8).cpu()"
new_line = "tensor = (tensor * 255).to(torch.float32).clamp(0, 255).to(torch.uint8).cpu()"

if old_line in content:
    content = content.replace(old_line, new_line)
    with open(utils_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 已修复: {utils_file}")
    print(f"   修改: {old_line}")
    print(f"   为:   {new_line}")
else:
    print(f"⚠️ 未找到需要修复的代码，可能已修复或文件版本不同")

# 同时修复 generate.py 中的视频保存调用
# 添加显式的类型转换
generate_file = "generate.py"

with open(generate_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 在 cache_video 调用前添加 .float() 转换
old_call = """cache_video(
                tensor=video[None],
                save_file=args.save_file,"""

new_call = """# 确保视频数据为 float32 类型以避免保存错误
            video_float = video[None].to(torch.float32)
            cache_video(
                tensor=video_float,
                save_file=args.save_file,"""

if old_call in content and "video_float" not in content:
    content = content.replace(old_call, new_call)
    with open(generate_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 已修复: {generate_file}")
else:
    print(f"⚠️ 未找到需要修复的代码或已修复: {generate_file}")

print("\n修复完成！现在可以重新运行测试命令。")
