# VBench-I2V 生成输入契约

本工具包同时支持两种 repeat 策略。二者含义不同，结果说明中必须明确记录。

## 1. `exact`：严格官方采样语义（默认）

对 full_info 中每个 image-prompt pair 独立调用生成模型 5 次：

```text
<prompt>-0.mp4
<prompt>-1.mp4
<prompt>-2.mp4
<prompt>-3.mp4
<prompt>-4.mp4
```

输入构建命令：

```bash
python -m hif4_vbench_i2v.build_eval_inputs \
  --template-case /path/to/template \
  --generated-dir /path/to/exact_repeats \
  --out-case /path/to/out_case \
  --repeat-policy exact \
  --copy-mode physical
```

`base-3.mp4` 模板只能由同名源文件填充；缺少任何 repeat 都会失败。

## 2. `replicate-base`：历史 scale60 兼容复现

已完成的 Wan2.2 HiF4/MXFP4 scale60 实验每个 pair 只生成一个基础视频，然后物理
复制到 evaluator 所要求的 5 个 repeat 文件名。该策略用于保持既有多格式实验口径，
**不是五次独立采样**。

必须显式确认：

```bash
python -m hif4_vbench_i2v.build_eval_inputs \
  --template-case /path/to/template \
  --generated-dir /path/to/scale60_merged \
  --out-case /path/to/out_case \
  --repeat-policy replicate-base \
  --acknowledge-replicated-repeats \
  --copy-mode physical
```

输出会打印：

```text
WARNING_REPEAT_POLICY=replicate-base
WARNING_REPEAT_SEMANTICS=one_generated_video_is_physically_copied_to_multiple_repeat_filenames
```

验收时需要：

```bash
python -m hif4_vbench_i2v.validate_case_input \
  --case-input /path/to/out_case \
  --expected-sb 200 \
  --expected-camera 100 \
  --expected-repeats 5 \
  --allow-identical-repeat-files \
  --forbid-symlink
```

论文或报告中应写成“基础视频物理扩展以匹配 evaluator 输入布局”，不能写成
“每个 pair 独立采样 5 次”。

## 3. 推荐 seed manifest

新实验建议记录：

```text
filename  prompt  image_name  dimension  repeat_index  seed  variant  checkpoint
```

这样可以确认 repeat、seed、variant 和 checkpoint 的对应关系。

## 4. 通用验收

无论使用哪种策略，都应检查：

1. mp4 数量；
2. `0..4` 文件名布局；
3. 是否含 symlink；
4. repeat 策略和 seed manifest；
5. 生成参数是否在不同格式间一致。
