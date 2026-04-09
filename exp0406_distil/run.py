#!/usr/bin/env python3
"""Convenience runner for the distillation project."""

import subprocess
import sys
from pathlib import Path


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print_header(description)
    result = subprocess.run(cmd)
    return result.returncode == 0


def show_status() -> None:
    """Show project status."""
    print_header("项目状态 / Project Status")

    # Check data
    data_file = Path("data/generated/distillation_data.jsonl")
    if data_file.exists():
        count = sum(1 for _ in open(data_file, "r", encoding="utf-8"))
        print(f"✓ 蒸馏数据：{data_file} ({count} 条)")
    else:
        print("✗ 蒸馏数据：尚未生成")

    # Check model
    model_dir = Path("models/student/final")
    if model_dir.exists():
        print(f"✓ 训练好的模型：{model_dir}")
    else:
        # Check for checkpoints
        output_dir = Path("models/student")
        if output_dir.exists():
            checkpoints = sorted(output_dir.glob("checkpoint-*"))
            if checkpoints:
                print(f"~ 训练检查点：发现 {len(checkpoints)} 个 checkpoint (可续训)")
            else:
                print("✗ 训练好的模型：尚未训练")
        else:
            print("✗ 训练好的模型：尚未训练")

    # Check results
    results_file = Path("results/eval/results.json")
    if results_file.exists():
        print(f"✓ 评估结果：{results_file}")
    else:
        print("✗ 评估结果：尚未评估")

    print()


def main():
    if len(sys.argv) < 2:
        print("使用方法 / Usage:")
        print()
        print("  python run.py <command> [--resume]")
        print()
        print("可用命令 / Commands:")
        print("  generate   - 生成蒸馏数据 (调用教师模型)")
        print("  train      - 训练学生模型 (支持断点续训)")
        print("  eval       - 评估模型性能")
        print("  all        - 完整流程：生成 → 训练 → 评估")
        print("  status     - 查看当前状态")
        print("  debug      - 调试模式：生成 10 条数据测试")
        print()
        print("选项 / Options:")
        print("  --resume   - 断点续训：跳过已完成的步骤，从断点继续")
        print()
        print("示例 / Examples:")
        print("  python run.py generate")
        print("  python run.py all        # 完整流程")
        print("  python run.py all --resume  # 从断点继续")
        print("  python run.py debug")
        print()
        return

    command = sys.argv[1].lower()
    resume_mode = "--resume" in sys.argv

    if command == "status":
        show_status()
        return

    if command == "generate":
        success = run_command(
            [sys.executable, "scripts/generate_data.py"],
            "生成蒸馏数据 / Generating Distillation Data"
        )
        if success:
            print("\n✓ 数据生成完成!")
        else:
            print("\n✗ 数据生成失败，请检查错误信息")
        return

    if command == "train":
        success = run_command(
            [sys.executable, "scripts/train_student.py"],
            "训练学生模型 / Training Student Model"
        )
        if success:
            print("\n✓ 训练完成!")
        else:
            print("\n✗ 训练失败，请检查错误信息")
        return

    if command == "eval":
        success = run_command(
            [sys.executable, "scripts/evaluate.py"],
            "评估模型 / Evaluating Model"
        )
        if success:
            print("\n✓ 评估完成!")
        else:
            print("\n✗ 评估失败，请检查错误信息")
        return

    if command == "all":
        print("完整流程 / Full Pipeline")
        print("-" * 40)

        # Determine which steps to run based on existing outputs and --resume flag
        data_file = Path("data/generated/distillation_data.jsonl")
        model_dir = Path("models/student/final")
        results_file = Path("results/eval/results.json")

        # Check what's already done
        data_done = data_file.exists() and data_file.stat().st_size > 0
        model_done = model_dir.exists()
        eval_done = results_file.exists()

        # In resume mode, skip completed steps; otherwise always run everything
        run_generate = not resume_mode or not data_done
        run_train = not resume_mode or not model_done
        run_eval = not resume_mode or not eval_done

        # Step 1: Generate
        if run_generate:
            if not run_command(
                [sys.executable, "scripts/generate_data.py"],
                "步骤 1/3: 生成蒸馏数据"
            ):
                print("\n✗ 数据生成失败")
                return
            print("\n✓ 步骤 1 完成")
        else:
            print(f"⊘ 跳过步骤 1 (数据已存在：{data_file})")

        # Step 2: Train
        if run_train:
            if not run_command(
                [sys.executable, "scripts/train_student.py"],
                "步骤 2/3: 训练学生模型"
            ):
                print("\n✗ 训练失败，可运行 'python run.py all --resume' 继续")
                return
            print("\n✓ 步骤 2 完成")
        else:
            print(f"⊘ 跳过步骤 2 (模型已存在：{model_dir})")

        # Step 3: Evaluate
        if run_eval:
            if not run_command(
                [sys.executable, "scripts/evaluate.py"],
                "步骤 3/3: 评估模型"
            ):
                print("\n✗ 评估失败")
                return
            print("\n✓ 步骤 3 完成")
        else:
            print(f"⊘ 跳过步骤 3 (结果已存在：{results_file})")

        print_header("流程完成 / Pipeline Complete!")
        show_status()
        return

    if command == "debug":
        print("调试模式 / Debug Mode")
        print("-" * 40)
        success = run_command(
            [sys.executable, "scripts/generate_data.py", "--debug"],
            "生成 10 条测试数据"
        )
        if success:
            print("\n✓ 调试数据生成完成!")
        else:
            print("\n✗ 调试数据生成失败")
        return

    print(f"未知命令：{command}")
    print("运行 python run.py 查看使用方法")


if __name__ == "__main__":
    main()
