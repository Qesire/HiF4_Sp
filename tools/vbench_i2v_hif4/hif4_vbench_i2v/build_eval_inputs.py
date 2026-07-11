"""把 HiF4/Wan 生成视频整理为 VBench-I2V case input。

默认采用严格 ``exact`` 策略：模板中的 ``base-3.mp4`` 必须由源目录中同名
``base-3.mp4`` 填充。为了复现已经完成的 scale60 实验，也提供显式的
``replicate-base`` 兼容策略：把一个基础生成结果物理复制到模板要求的 5 个文件名。
后者不是 5 次独立采样，必须同时传入确认参数并在报告中如实标注。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .constants import CAM_GROUP, SB_GROUP
from .utils import (
    build_video_index,
    build_video_name_index,
    copy_file,
    copy_tree_physical,
    count_symlinks,
    list_mp4,
    norm_video_base,
)


def replace_from_template(
    template_dir: Path,
    out_dir: Path,
    video_index: dict[str, Path],
    copy_mode: str,
    repeat_policy: str = "exact",
) -> int:
    """按模板文件名填充视频目录。

    ``exact`` 使用完整文件名匹配；``replicate-base`` 使用去除末尾 repeat id 后的
    prompt base 匹配，因此同一基础视频可能被物理复制到多个 repeat 文件名。
    """

    files = list_mp4(template_dir)
    if not files:
        raise RuntimeError(f"模板目录没有 mp4: {template_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    copied = 0
    for template_file in files:
        key = template_file.name if repeat_policy == "exact" else norm_video_base(template_file.name)
        source = video_index.get(key)
        if source is None:
            missing.append(template_file.name)
            continue
        copy_file(source, out_dir / template_file.name, copy_mode)
        copied += 1

    if missing:
        for name in missing[:50]:
            print(f"MISSING_SOURCE template_name={name} repeat_policy={repeat_policy}")
        if repeat_policy == "exact":
            raise RuntimeError(
                f"缺少 {len(missing)} 个模板文件对应的 exact repeat 源视频；"
                "请真实生成 prompt-0..prompt-4，或为历史复现实验显式选择 replicate-base。"
            )
        raise RuntimeError(f"缺少 {len(missing)} 个模板文件对应的基础视频")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 HiF4 的 VBench-I2V case input")
    parser.add_argument("--template-case", required=True)
    parser.add_argument("--out-case", required=True)
    parser.add_argument("--generated-dir", default=None, help="统一生成视频目录；也可分别提供 subject/background/camera")
    parser.add_argument("--subject-dir", default=None)
    parser.add_argument("--background-dir", default=None)
    parser.add_argument("--camera-dir", default=None)
    parser.add_argument("--copy-mode", choices=["physical", "hardlink", "symlink", "reflink"], default="physical")
    parser.add_argument(
        "--repeat-policy",
        choices=["exact", "replicate-base"],
        default="exact",
        help="exact=五个独立同名源文件；replicate-base=一个基础视频扩展为模板 repeat，仅用于复现历史兼容流程",
    )
    parser.add_argument(
        "--acknowledge-replicated-repeats",
        action="store_true",
        help="确认 replicate-base 不是五次独立采样，并接受在结果说明中显式标注",
    )
    parser.add_argument("--forbid-symlink", action="store_true", default=True, help="默认禁止输出中残留 symlink")
    parser.add_argument("--allow-symlink", action="store_false", dest="forbid_symlink", help="允许 symlink")
    args = parser.parse_args()

    if args.repeat_policy == "replicate-base" and not args.acknowledge_replicated_repeats:
        raise SystemExit("replicate-base 必须同时传入 --acknowledge-replicated-repeats")

    template = Path(args.template_case)
    out_case = Path(args.out_case)
    if not template.is_dir():
        raise SystemExit(f"missing template case: {template}")

    print(f"COPY_TEMPLATE {template} -> {out_case}")
    copy_tree_physical(template, out_case)

    source_dirs_sb: list[Path] = []
    source_dirs_camera: list[Path] = []
    if args.generated_dir:
        source_dirs_sb.append(Path(args.generated_dir))
        source_dirs_camera.append(Path(args.generated_dir))
    if args.subject_dir:
        source_dirs_sb.append(Path(args.subject_dir))
    if args.background_dir:
        source_dirs_sb.append(Path(args.background_dir))
    if args.camera_dir:
        source_dirs_camera.append(Path(args.camera_dir))
    if not source_dirs_sb or not source_dirs_camera:
        raise SystemExit("需要 --generated-dir 或 subject/background/camera 专用目录")

    if args.repeat_policy == "exact":
        sb_index = build_video_name_index(source_dirs_sb)
        camera_index = build_video_name_index(source_dirs_camera)
    else:
        sb_index = build_video_index(source_dirs_sb)
        camera_index = build_video_index(source_dirs_camera)
        print("WARNING_REPEAT_POLICY=replicate-base")
        print("WARNING_REPEAT_SEMANTICS=one_generated_video_is_physically_copied_to_multiple_repeat_filenames")

    sb_template = template / SB_GROUP / "videos_quant_sb"
    camera_template = template / CAM_GROUP / "videos_quant_camera"
    sb_out = out_case / SB_GROUP / "videos_quant_sb"
    camera_out = out_case / CAM_GROUP / "videos_quant_camera"

    sb_count = replace_from_template(sb_template, sb_out, sb_index, args.copy_mode, args.repeat_policy)
    camera_count = replace_from_template(camera_template, camera_out, camera_index, args.copy_mode, args.repeat_policy)

    links = count_symlinks(out_case)
    print(f"videos_quant_sb={sb_count}")
    print(f"videos_quant_camera={camera_count}")
    print(f"repeat_policy={args.repeat_policy}")
    print(f"symlink_count={links}")
    if args.forbid_symlink and links:
        raise SystemExit("检测到 symlink；默认禁止 symlink")
    print("BUILD_EVAL_INPUTS_OK")


if __name__ == "__main__":
    main()
