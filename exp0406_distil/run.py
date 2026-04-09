#!/usr/bin/env python3
"""Convenience runner for math data generation."""

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

    print()


def main():
    if len(sys.argv) < 2:
        print("使用方法 / Usage:")
        print()
        print("  python run.py <command>")
        print()
        print("可用命令 / Commands:")
        print("  generate         - 生成蒸馏数据 (单进程)")
        print("  generate-parallel - 生成蒸馏数据 (多进程并行)")
        print("  status           - 查看当前状态")
        print("  debug            - 调试模式：生成 10 条数据测试")
        print()
        print("示例 / Examples:")
        print("  python run.py generate")
        print("  python run.py generate-parallel --workers 32")
        print("  python run.py debug")
        print()
        return

    command = sys.argv[1].lower()

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

    if command == "generate-parallel":
        cmd = [sys.executable, "scripts/generate_parallel.py"]
        if len(sys.argv) > 2:
            cmd.extend(sys.argv[2:])
        success = run_command(cmd, "并行生成蒸馏数据 / Parallel Data Generation")
        if success:
            print("\n✓ 数据生成完成!")
        else:
            print("\n✗ 数据生成失败，请检查错误信息")
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
