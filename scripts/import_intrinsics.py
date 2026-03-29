"""
从 intrinsic.txt 自动更新 cameras.yaml 中的相机内参。

注意：yaml.dump 写回会丢失 YAML 注释。建议写入后用 git diff 确认变更。

用法示例:
    # 预览，不写入
    python scripts/import_intrinsics.py \\
        --camera hikvision_normal \\
        --from-file intrinsic_calib/intrinsic.txt \\
        --index 0 --dry-run

    # 写入并更新分辨率
    python scripts/import_intrinsics.py \\
        --camera hikvision_normal \\
        --from-file intrinsic_calib/intrinsic.txt \\
        --index 0 --width 2448 --height 2048
"""

import argparse
import sys
import os

import yaml

# 确保项目根目录在路径中
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from robovision.calibration.intrinsic import load_intrinsic_txt

_DEFAULT_CAMERAS_YAML = os.path.join(_ROOT_DIR, "config", "cameras.yaml")


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 intrinsic.txt 中的内参导入 cameras.yaml"
    )
    parser.add_argument(
        "--camera", required=True,
        help="cameras.yaml 中的相机名称（如 hikvision_normal）"
    )
    parser.add_argument(
        "--from-file", required=True, dest="from_file",
        help="intrinsic.txt 文件路径"
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="使用 intrinsic.txt 中第几组内参（从 0 开始，默认 0）"
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="同步更新 intrinsics.width（可选）"
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="同步更新 intrinsics.height（可选）"
    )
    parser.add_argument(
        "--cameras-yaml", default=_DEFAULT_CAMERAS_YAML,
        help=f"cameras.yaml 路径（默认 {_DEFAULT_CAMERAS_YAML}）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印变更，不写入文件"
    )
    return parser.parse_args()


def _fmt_matrix(rows):
    """格式化 3x3 相机矩阵为可读字符串。"""
    return (
        f"  fx={rows[0][0]:.4f}  fy={rows[1][1]:.4f}"
        f"  cx={rows[0][2]:.4f}  cy={rows[1][2]:.4f}"
    )


def main():
    args = parse_args()

    # --- 加载 intrinsic.txt ---
    if not os.path.isfile(args.from_file):
        print(f"[ERROR] 找不到内参文件: {args.from_file}", file=sys.stderr)
        sys.exit(1)

    results = load_intrinsic_txt(args.from_file)
    if not results:
        print(f"[ERROR] {args.from_file} 中未解析到任何内参", file=sys.stderr)
        sys.exit(1)

    if args.index >= len(results):
        print(
            f"[ERROR] --index {args.index} 超出范围，"
            f"文件中共 {len(results)} 组内参（索引 0~{len(results)-1}）",
            file=sys.stderr,
        )
        sys.exit(1)

    result = results[args.index]
    K_new = result.camera_matrix
    dist_new = result.dist_coeffs

    # --- 加载 cameras.yaml ---
    if not os.path.isfile(args.cameras_yaml):
        print(f"[ERROR] 找不到配置文件: {args.cameras_yaml}", file=sys.stderr)
        sys.exit(1)

    with open(args.cameras_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cameras = config.get("cameras", {})
    available = list(cameras.keys())

    if args.camera not in cameras:
        print(
            f"[ERROR] 相机 '{args.camera}' 不存在于 cameras.yaml\n"
            f"可用相机: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    cam_cfg = cameras[args.camera]
    intrinsics = cam_cfg.setdefault("intrinsics", {})

    # --- 打印对比 ---
    old_K = intrinsics.get("camera_matrix")
    old_dist = intrinsics.get("dist_coeffs")

    print(f"\n相机: {args.camera}")
    print(f"内参来源: {args.from_file} (index={args.index})")

    if old_K:
        print(f"\n[旧] camera_matrix: {_fmt_matrix(old_K)}")
    else:
        print("\n[旧] camera_matrix: (未设置)")

    new_K_rows = K_new.tolist()
    print(f"[新] camera_matrix: {_fmt_matrix(new_K_rows)}")

    if old_dist:
        print(f"\n[旧] dist_coeffs: {old_dist}")
    print(f"[新] dist_coeffs: {dist_new.tolist()}")

    if args.width is not None:
        old_w = intrinsics.get("width", "(未设置)")
        print(f"\n[旧] width:  {old_w}")
        print(f"[新] width:  {args.width}")

    if args.height is not None:
        old_h = intrinsics.get("height", "(未设置)")
        print(f"[旧] height: {old_h}")
        print(f"[新] height: {args.height}")

    if args.dry_run:
        print("\n[dry-run] 未写入文件。")
        return

    # --- 更新并写回 ---
    intrinsics["camera_matrix"] = new_K_rows
    intrinsics["dist_coeffs"] = dist_new.tolist()

    if args.width is not None:
        intrinsics["width"] = args.width
    if args.height is not None:
        intrinsics["height"] = args.height

    with open(args.cameras_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=None, sort_keys=False)

    print(f"\n[OK] 已写入 {args.cameras_yaml}")
    print("     注意：yaml.dump 会丢失原有注释，请用 git diff 确认变更。")


if __name__ == "__main__":
    main()
