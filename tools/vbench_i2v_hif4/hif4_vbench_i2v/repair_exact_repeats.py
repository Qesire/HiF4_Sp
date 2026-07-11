"""重建 VBench-I2V 输入目录，支持严格 exact 与历史 replicate-base 策略。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .build_eval_inputs import replace_from_template
from .constants import CAM_GROUP, SB_GROUP
from .utils import build_video_index, build_video_name_index, count_symlinks


def main() -> None:
    parser = argparse.ArgumentParser(description="重建 HiF4 case input")
    parser.add_argument("--template-case", required=True)
    parser.add_argument("--case-input", required=True)
    parser.add_argument("--generated-dir", default=None)
    parser.add_argument("--subject-dir", default=None)
    parser.add_argument("--background-dir", default=None)
    parser.add_argument("--camera-dir", default=None)
    parser.add_argument("--copy-mode", choices=["physical", "hardlink", "symlink", "reflink"], default="physical")
    parser.add_argument("--repeat-policy", choices=["exact", "replicate-base"], default="exact")
    parser.add_argument("--acknowledge-replicated-repeats", action="store_true")
    parser.add_argument("--forbid-symlink", action="store_true", default=True)
    parser.add_argument("--allow-symlink", action="store_false", dest="forbid_symlink")
    args = parser.parse_args()

    if args.repeat_policy == "replicate-base" and not args.acknowledge_replicated_repeats:
        raise SystemExit("replicate-base 必须同时传入 --acknowledge-replicated-repeats")

    template = Path(args.template_case)
    case_input = Path(args.case_input)
    if not template.is_dir():
        raise SystemExit(f"missing template: {template}")
    if not case_input.is_dir():
        raise SystemExit(f"missing case_input: {case_input}")

    sb_dirs: list[Path] = []
    camera_dirs: list[Path] = []
    if args.generated_dir:
        sb_dirs.append(Path(args.generated_dir))
        camera_dirs.append(Path(args.generated_dir))
    if args.subject_dir:
        sb_dirs.append(Path(args.subject_dir))
    if args.background_dir:
        sb_dirs.append(Path(args.background_dir))
    if args.camera_dir:
        camera_dirs.append(Path(args.camera_dir))
    if not sb_dirs or not camera_dirs:
        raise SystemExit("需要 --generated-dir 或 subject/background/camera 专用目录")

    if args.repeat_policy == "exact":
        sb_index = build_video_name_index(sb_dirs)
        camera_index = build_video_name_index(camera_dirs)
    else:
        sb_index = build_video_index(sb_dirs)
        camera_index = build_video_index(camera_dirs)
        print("WARNING_REPEAT_POLICY=replicate-base")

    sb_count = replace_from_template(
        template / SB_GROUP / "videos_quant_sb",
        case_input / SB_GROUP / "videos_quant_sb",
        sb_index,
        args.copy_mode,
        args.repeat_policy,
    )
    camera_count = replace_from_template(
        template / CAM_GROUP / "videos_quant_camera",
        case_input / CAM_GROUP / "videos_quant_camera",
        camera_index,
        args.copy_mode,
        args.repeat_policy,
    )

    links = count_symlinks(case_input)
    print(f"videos_quant_sb={sb_count}")
    print(f"videos_quant_camera={camera_count}")
    print(f"repeat_policy={args.repeat_policy}")
    print(f"symlink_count={links}")
    if args.forbid_symlink and links:
        raise SystemExit("检测到 symlink；默认禁止 symlink")
    print("REPAIR_EXACT_REPEATS_OK")


if __name__ == "__main__":
    main()
