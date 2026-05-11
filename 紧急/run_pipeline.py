"""
Layer29 风险概念提取完整 Pipeline

按照 TODO_list_v4.md (紧急版) 规范，一键运行完整流程

流程:
1. SAE Latent 提取
2. Feature 统计分析
3. Concept Vector 构建
4. 概念方向验证
5. Feature 可解释性分析
6. 概念干预实验
7. 论文结果生成
8. [新增] 视频生成与评估 (可选)

作者：Claude
日期：2026-05-11
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Pipeline 步骤
# ============================================================================

PIPELINE_STEPS = [
    {
        "id": 1,
        "name": "SAE Latent 提取",
        "script": "1_extract_latents.py",
        "description": "从 Wan DiT Layer29 提取 SAE latent",
        "required": True,
    },
    {
        "id": 2,
        "name": "Feature 统计分析",
        "script": "2_feature_analysis.py",
        "description": "Cohen's d 分析，判别性 feature 筛选",
        "required": True,
    },
    {
        "id": 3,
        "name": "Concept Vector 构建",
        "script": "3_build_vectors.py",
        "description": "构建概念向量 (dense & sparse)",
        "required": True,
    },
    {
        "id": 4,
        "name": "概念方向验证",
        "script": "4_validate_concepts.py",
        "description": "Projection Score & AUC 验证",
        "required": True,
    },
    {
        "id": 5,
        "name": "Feature 可解释性分析",
        "script": "5_feature_interpret.py",
        "description": "Top feature prompt 检索",
        "required": False,
    },
    {
        "id": 6,
        "name": "概念干预实验",
        "script": "6_intervention.py",
        "description": "Latent 干预测试",
        "required": False,
    },
    {
        "id": 7,
        "name": "论文结果生成",
        "script": "7_paper_results.py",
        "description": "生成表格和图表",
        "required": False,
    },
    {
        "id": 8,
        "name": "视频生成与评估",
        "script": "generate_and_evaluate.py",
        "description": "带SAE干预的视频生成 + MUSIQ评估 + 内容审核",
        "required": False,
    },
]


# ============================================================================
# 目录结构
# ============================================================================

def create_directory_structure(base_dir: str):
    """创建目录结构"""
    directories = [
        "datasets",
        "outputs/layer29_latents",
        "outputs/concept_features",
        "outputs/concept_vectors",
        "outputs/validation_results",
        "outputs/feature_interpret",
        "outputs/intervention_results",
        "outputs/paper_results",
        "outputs/generated_videos",
        "outputs/evaluation_results",
        "outputs/full_evaluation",
    ]

    for d in directories:
        path = Path(base_dir) / d
        path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created directory structure under {base_dir}")


# ============================================================================
# 脚本运行
# ============================================================================

def run_script(
    script_path: str,
    args: List[str],
    python_path: str = "python",
    cwd: Optional[str] = None,
) -> bool:
    """
    运行 Python 脚本

    返回:
        bool: 是否成功
    """
    cmd = [python_path, script_path] + args

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
        )

        # 打印输出
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")

        if result.returncode != 0:
            logger.error(f"Script failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr}")
            return False

        logger.info(f"Script completed successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to run script: {e}")
        return False


# ============================================================================
# Pipeline 运行器
# ============================================================================

class PipelineRunner:
    """Pipeline 运行器"""

    def __init__(
        self,
        base_dir: str,
        model_path: str,
        sae_checkpoint: str,
        prompt_dir: str,
        output_dir: str = "./outputs",
        concepts: str = "sex,violence",
        skip_steps: List[int] = None,
        only_steps: List[int] = None,
        do_generate_and_eval: bool = False,
        max_pairs_per_concept: int = 5,
        gamma_values: List[float] = None,
    ):
        self.base_dir = Path(base_dir)
        self.model_path = model_path
        self.sae_checkpoint = sae_checkpoint
        self.prompt_dir = prompt_dir
        self.output_dir = output_dir
        self.concepts = concepts
        self.skip_steps = skip_steps or []
        self.only_steps = only_steps
        self.do_generate_and_eval = do_generate_and_eval
        self.max_pairs_per_concept = max_pairs_per_concept
        self.gamma_values = gamma_values or [0.0, 0.3, 0.5, 0.8, 1.0]

        # 记录结果
        self.results = {}

    def should_run_step(self, step_id: int) -> bool:
        """判断是否应该运行某步骤"""
        if step_id in self.skip_steps:
            return False

        if self.only_steps and step_id not in self.only_steps:
            return False

        # Step 8 需要显式开启
        if step_id == 8 and not self.do_generate_and_eval:
            return False

        return True

    def get_script_path(self, script: str) -> str:
        """获取脚本路径"""
        return str(self.base_dir / script)

    def run_step1(self) -> bool:
        """运行 Step 1: SAE Latent 提取"""
        logger.info("\n" + "="*70)
        logger.info("STEP 1: SAE Latent 提取")
        logger.info("="*70)

        args = [
            "--model_path", self.model_path,
            "--sae_checkpoint", self.sae_checkpoint,
            "--prompt_dir", self.prompt_dir,
            "--output_dir", f"{self.output_dir}/layer29_latents",
            "--categories", "all",
        ]

        return run_script(
            self.get_script_path("1_extract_latents.py"),
            args,
            cwd=str(self.base_dir),
        )

    def run_step2(self) -> bool:
        """运行 Step 2: Feature 统计分析"""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Feature 统计分析")
        logger.info("="*70)

        args = [
            "--latent_dir", f"{self.output_dir}/layer29_latents",
            "--output_dir", f"{self.output_dir}/concept_features",
            "--concepts", self.concepts,
        ]

        return run_script(
            self.get_script_path("2_feature_analysis.py"),
            args,
            cwd=str(self.base_dir),
        )

    def run_step3(self) -> bool:
        """运行 Step 3: Concept Vector 构建"""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Concept Vector 构建")
        logger.info("="*70)

        args = [
            "--latent_dir", f"{self.output_dir}/layer29_latents",
            "--feature_dir", f"{self.output_dir}/concept_features",
            "--output_dir", f"{self.output_dir}/concept_vectors",
            "--concepts", self.concepts,
        ]

        return run_script(
            self.get_script_path("3_build_vectors.py"),
            args,
            cwd=str(self.base_dir),
        )

    def run_step4(self) -> bool:
        """运行 Step 4: 概念方向验证"""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: 概念方向验证")
        logger.info("="*70)

        args = [
            "--latent_dir", f"{self.output_dir}/layer29_latents",
            "--vector_dir", f"{self.output_dir}/concept_vectors",
            "--output_dir", f"{self.output_dir}/validation_results",
            "--concepts", self.concepts,
        ]

        return run_script(
            self.get_script_path("4_validate_concepts.py"),
            args,
            cwd=str(self.base_dir),
        )

    def run_step5(self) -> bool:
        """运行 Step 5: Feature 可解释性分析"""
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Feature 可解释性分析")
        logger.info("="*70)

        args = [
            "--latent_dir", f"{self.output_dir}/layer29_latents",
            "--feature_dir", f"{self.output_dir}/concept_features",
            "--output_dir", f"{self.output_dir}/feature_interpret",
            "--concepts", self.concepts,
        ]

        return run_script(
            self.get_script_path("5_feature_interpret.py"),
            args,
            cwd=str(self.base_dir),
        )

    def run_step6(self) -> bool:
        """运行 Step 6: 概念干预实验"""
        logger.info("\n" + "="*70)
        logger.info("STEP 6: 概念干预实验")
        logger.info("="*70)

        results = []
        concepts_list = [c.strip() for c in self.concepts.split(",")]

        for concept in concepts_list:
            # Positive
            args = [
                "--mode", "simple",
                "--latent_dir", f"{self.output_dir}/layer29_latents",
                "--vector_dir", f"{self.output_dir}/concept_vectors",
                "--output_dir", f"{self.output_dir}/intervention_results",
                "--concept", concept,
                "--category", f"{concept}_positive",
                "--gamma", *[str(g) for g in self.gamma_values],
            ]

            success = run_script(
                self.get_script_path("6_intervention.py"),
                args,
                cwd=str(self.base_dir),
            )
            results.append(success)

        return all(results)

    def run_step7(self) -> bool:
        """运行 Step 7: 论文结果生成"""
        logger.info("\n" + "="*70)
        logger.info("STEP 7: 论文结果生成")
        logger.info("="*70)

        args = [
            "--validation_dir", f"{self.output_dir}/validation_results",
            "--feature_dir", f"{self.output_dir}/concept_features",
            "--intervention_dir", f"{self.output_dir}/intervention_results",
            "--interpret_dir", f"{self.output_dir}/feature_interpret",
            "--output_dir", f"{self.output_dir}/paper_results",
            "--concepts", self.concepts,
        ]

        return run_script(
            self.get_script_path("7_paper_results.py"),
            args,
            cwd=str(self.base_dir),
        )

    def run_step8(self) -> bool:
        """
        运行 Step 8: 视频生成与评估 (新增)

        包含:
        - 带SAE干预的视频生成
        - MUSIQ图像质量评估
        - NSFW/Violence内容检测
        - ASR计算
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 8: 视频生成与评估")
        logger.info("="*70)

        # 构建 gamma 参数
        gamma_args = [str(g) for g in self.gamma_values]

        args = [
            "--model_path", self.model_path,
            "--sae_checkpoint", self.sae_checkpoint,
            "--vector_dir", f"{self.output_dir}/concept_vectors",
            "--prompt_dir", self.prompt_dir,
            "--output_dir", f"{self.output_dir}/full_evaluation",
            "--concepts", self.concepts,
            "--max_prompts_per_file", "5",
            "--max_pairs_per_concept", str(self.max_pairs_per_concept),
            "--gamma", *gamma_args,
        ]

        return run_script(
            self.get_script_path("generate_and_evaluate.py"),
            args,
            cwd=str(self.base_dir),
        )

    def run(self) -> Dict[str, Any]:
        """运行完整 pipeline"""
        start_time = datetime.now()

        logger.info(f"\n{'='*70}")
        logger.info("Layer29 Risk Concept Extraction Pipeline")
        logger.info(f"{'='*70}")
        logger.info(f"  Base dir: {self.base_dir}")
        logger.info(f"  Model path: {self.model_path}")
        logger.info(f"  SAE checkpoint: {self.sae_checkpoint}")
        logger.info(f"  Prompt dir: {self.prompt_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Concepts: {self.concepts}")
        logger.info(f"  Do generate & eval: {self.do_generate_and_eval}")
        if self.do_generate_and_eval:
            logger.info(f"  Max pairs per concept: {self.max_pairs_per_concept}")
            logger.info(f"  Gamma values: {self.gamma_values}")
        logger.info(f"{'='*70}")

        # 创建目录结构
        create_directory_structure(self.base_dir)

        # 运行各步骤
        step_runners = {
            1: self.run_step1,
            2: self.run_step2,
            3: self.run_step3,
            4: self.run_step4,
            5: self.run_step5,
            6: self.run_step6,
            7: self.run_step7,
            8: self.run_step8,
        }

        for step in PIPELINE_STEPS:
            step_id = step["id"]

            if not self.should_run_step(step_id):
                logger.info(f"\nSkipping Step {step_id}: {step['name']}")
                self.results[step_id] = {"status": "skipped"}
                continue

            success = step_runners[step_id]()

            self.results[step_id] = {
                "name": step["name"],
                "status": "success" if success else "failed",
            }

            if not success and step["required"]:
                logger.error(f"\nPipeline stopped at Step {step_id} (required)")
                break

        # 汇总
        elapsed = datetime.now() - start_time

        summary = {
            "start_time": start_time.isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "do_generate_and_eval": self.do_generate_and_eval,
            "results": self.results,
        }

        # 保存结果
        summary_file = Path(self.output_dir) / "pipeline_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # 打印汇总
        logger.info(f"\n{'='*70}")
        logger.info("Pipeline Summary")
        logger.info(f"{'='*70}")

        for step_id, result in self.results.items():
            status = result.get("status", "unknown")
            name = result.get("name", f"Step {step_id}")
            logger.info(f"  Step {step_id} ({name}): {status}")

        logger.info(f"\nTotal time: {elapsed}")
        logger.info(f"Summary saved to: {summary_file}")

        return summary


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Layer29 Risk Concept Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 运行 Steps 1-7 (Latent 提取到论文结果生成)
    python run_pipeline.py \\
        --model_path /root/Wan/Wan2.1-T2V-1.3B \\
        --sae_checkpoint ./sae_init_layer29.pt \\
        --prompt_dir ./datasets \\
        --output_dir ./outputs

    # 运行完整流程，包含视频生成与评估
    python run_pipeline.py \\
        --model_path /root/Wan/Wan2.1-T2V-1.3B \\
        --sae_checkpoint ./sae_init_layer29.pt \\
        --prompt_dir ./final_cleaned \\
        --output_dir ./outputs \\
        --do_generate_and_eval \\
        --max_pairs_per_concept 5 \\
        --gamma 0.0 0.3 0.5 0.8 1.0

    # 只运行特定步骤
    python run_pipeline.py \\
        --model_path /root/Wan/Wan2.1-T2V-1.3B \\
        --sae_checkpoint ./sae_init_layer29.pt \\
        --prompt_dir ./datasets \\
        --only 1 2 3 4

    # 跳过某些步骤
    python run_pipeline.py \\
        --model_path /root/Wan/Wan2.1-T2V-1.3B \\
        --sae_checkpoint ./sae_init_layer29.pt \\
        --prompt_dir ./datasets \\
        --skip 5 6
        """
    )

    # 路径配置
    parser.add_argument("--base_dir", type=str, default="./紧急",
                        help="Pipeline 脚本所在目录")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Wan model path")
    parser.add_argument("--sae_checkpoint", type=str, required=True,
                        help="Layer29 SAE checkpoint path")
    parser.add_argument("--prompt_dir", type=str, required=True,
                        help="Prompt dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")

    # 概念配置
    parser.add_argument("--concepts", type=str, default="sex,violence",
                        help="Concepts to analyze")

    # 步骤控制
    parser.add_argument("--skip", type=int, nargs='+', default=[],
                        help="Steps to skip")
    parser.add_argument("--only", type=int, nargs='+', default=[],
                        help="Only run these steps")

    # [新增] 视频生成与评估
    parser.add_argument("--do_generate_and_eval", action="store_true",
                        help="Enable video generation and evaluation (Step 8)")
    parser.add_argument("--max_pairs_per_concept", type=int, default=5,
                        help="Max prompt pairs per concept for video generation")
    parser.add_argument("--gamma", type=float, nargs='+',
                        default=[0.0, 0.3, 0.5, 0.8, 1.0],
                        help="Intervention strengths for ablation study")

    args = parser.parse_args()

    # 创建 runner
    runner = PipelineRunner(
        base_dir=args.base_dir,
        model_path=args.model_path,
        sae_checkpoint=args.sae_checkpoint,
        prompt_dir=args.prompt_dir,
        output_dir=args.output_dir,
        concepts=args.concepts,
        skip_steps=args.skip,
        only_steps=args.only if args.only else None,
        do_generate_and_eval=args.do_generate_and_eval,
        max_pairs_per_concept=args.max_pairs_per_concept,
        gamma_values=args.gamma,
    )

    # 运行
    runner.run()


if __name__ == "__main__":
    main()
