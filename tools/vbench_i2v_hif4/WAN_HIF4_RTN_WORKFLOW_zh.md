# Wan2.2-I2V HiF4 RTN-QDQ 接口与已验证工作流

本文补充本工具包原先缺失的**生成端**：如何直接复用 HiF4_Sp 官方
`quant_dequant_float/QType`，把 Wan2.2-I2V-A14B 的两个 DiT 接入 HiF4 W4A4
RTN fake-quant，并沿用 scale60、三 seed、VBench-I2V 十维评测和缺失重试链。

> 定位：这是生成质量研究所用的 **W4A4 RTN-QDQ 数值基线**。权重和激活在
> QDQ 后仍调用浮点 `F.linear`，不是 packed HiF4 GEMM，不应据此报告 4-bit
> 吞吐、显存或推理加速。

## 1. 已验证语义

```text
BF16 Wan2.2-I2V checkpoint
  ├─ low_noise_model: 选择 400 个 attention/FFN Linear
  └─ high_noise_model: 选择 400 个 attention/FFN Linear
       ↓
权重：构建时一次 HiF4 RTN-QDQ
激活：每次 forward 在线 HiF4 QDQ
计算：BF16/原 dtype F.linear
```

底层格式由本仓库官方接口决定：

```python
quant_dequant_float(x, QType("hifx4").dim(-1), force_fp32=True)
```

Wan 接入层位于：

```text
hif4_vbench_i2v/wan_adapter/hif4_backend.py
hif4_vbench_i2v/wan_adapter/hif4_linear.py
hif4_vbench_i2v/wan_adapter/generator_hooks.py
```

## 2. 三层接口

### 2.1 张量级

```python
hif4_qdq_tensor(x, dim=-1, kind="activation")
hif4_rtn_qdq_weight(weight, dim=-1)
```

负责动态加载 HiF4_Sp CUDA API、设备往返和调用统计。

### 2.2 Linear 级

```python
HiF4FakeQuantLinear.from_linear(linear)
replace_linear_with_hif4(model)
```

默认模块选择器包含 attention/FFN/projection，排除 time/timestep、embedding、
AdaLN/modulation、norm、patch、final/head、VAE/T5/text。Wan2.2-I2V-A14B 已验证
选择数为 `400 + 400 = 800`。

### 2.3 生成器级

已有 Wan 生成器只需：

```python
from hif4_vbench_i2v.wan_adapter.generator_hooks import (
    add_hif4_arguments,
    setup_hif4_if_needed,
    write_hif4_runtime_report,
)
```

在构造 parser 后：

```python
add_hif4_arguments(parser)
```

在 `wan.WanI2V(...)` 构造后、模型正式生成前：

```python
setup_hif4_if_needed(pipe, args, LOGGER)
```

在生成循环结束后：

```python
write_hif4_runtime_report(
    args,
    generated=generated,
    skipped=skipped,
    failed=failed,
    manifest=manifest_path,
    logger=LOGGER,
)
```

该接口会输出 setup/runtime JSON，记录选择清单、官方 API 来源、权重/激活
QDQ 调用次数以及生成成功数。

## 3. 环境分工

已验证工作流采用 split-env：

| 环境 | 用途 |
|---|---|
| `hif4` | HiF4 CUDA QDQ、Wan2.2 视频生成 |
| `vbench_i2v_official` | 官方 VBench-I2V 十维评测 |
| shell/base | 合并、构造输入、扫描和提交 |

推荐变量：

```bash
ROOT=/home/scc/pb23061276/projects/wan-quant
WAN_REPO=$ROOT/code/Wan2.2-I2V-A14B-W4A4
HIF4_REPO=/path/to/HiF4_Sp
CKPT_DIR=$ROOT/weights/Wan2.2-I2V-A14B-bf16
VBENCH_ROOT=$ROOT/data/VBench/vbench2_beta_i2v
FULL_INFO_JSON=$VBENCH_ROOT/vbench2_i2v_full_info.json
```

所有 Slurm 作业先清理失效 cwd：

```bash
unset PWD OLDPWD || true
cd /home/scc/pb23061276
pwd -P >/dev/null
```

## 4. 已验证链路

历史脚本编号：

```text
111 setup gate
→ 117 full W4A4 single-video smoke
→ 129 scale60 array generation
→ 129c merge 60 videos/seed
→ 147 build 200/100 case inputs
→ 143 official 10-dim eval
→ 145 scan missing
→ 148 retry missing until complete
```

本分支提供等价模板：

```text
examples/slurm_ustc_verified/111_hif4_setup_dryrun.sbatch
examples/slurm_ustc_verified/117_hif4_full_single_smoke.sbatch
examples/slurm_ustc_verified/129_hif4_rtn_w4a4_scale60_array_generate.sbatch
examples/slurm_ustc_verified/129c_merge_hif4_rtn_w4a4_scale60_array_outputs.sh
examples/slurm_ustc_verified/147_build_hif4_case_inputs_replicated.sh
```

评测、扫描、retry 继续复用本工具包已有的：

```text
examples/slurm_ustc_template/eval_one_case.sbatch
python -m hif4_vbench_i2v.scan_missing
python -m hif4_vbench_i2v.retry_missing
```

## 5. Setup 与 smoke 门禁

Setup：

```text
low_noise_model selected_count=400
high_noise_model selected_count=400
weight_qdq_calls=800
HIF4_SETUP_GATE_OK
```

完整 W4A4 smoke：

```text
weight_qdq_calls=800
activation_qdq_calls>0
MP4_COUNT=1
HIF4_FULL_SINGLE_SMOKE_OK
```

## 6. scale60 固定参数

每个 seed 对三个维度各生成 20 条基础视频：

```text
i2v_subject=20
i2v_background=20
camera_motion=20
total=60
```

固定生成参数：

```text
seeds: 42 / 43 / 44
size: 832*480
frame_num: 61
sample_steps: 4
sample_shift: 5.0
sample_guide_scale: 5.0
samples_per_prompt: 1
start_offset: 10
n_per_dim: 20
```

提交示例：

```bash
SCRIPT=tools/vbench_i2v_hif4/examples/slurm_ustc_verified/129_hif4_rtn_w4a4_scale60_array_generate.sbatch

J42=$(SEED=42 sbatch --parsable --chdir=/home/scc/pb23061276 "$SCRIPT")
J42=${J42%%;*}
J43=$(SEED=43 sbatch --parsable --dependency="afterany:$J42" --chdir=/home/scc/pb23061276 "$SCRIPT")
J43=${J43%%;*}
J44=$(SEED=44 sbatch --parsable --dependency="afterany:$J43" --chdir=/home/scc/pb23061276 "$SCRIPT")
```

## 7. 两种 repeat 策略必须区分

### `exact`：严格官方语义，默认

每个 image-prompt pair 独立生成 `-0` 到 `-4` 五个视频。工具默认只接受完整
同名源文件。

### `replicate-base`：已完成 scale60 实验的兼容复现

已验证的历史链路每个 pair 只生成 1 个视频，随后为了适配当前 evaluator 的目录和
命名要求，将 60 个基础视频**物理复制**成：

```text
subject/background: 200
camera: 100
symlink: 0
```

这不是五次独立采样，不能描述成官方 five-sample protocol。为了避免误用，命令必须
显式包含：

```bash
--repeat-policy replicate-base \
--acknowledge-replicated-repeats
```

验收时还需显式允许相同哈希：

```bash
--allow-identical-repeat-files
```

该模式只用于复现既有 BF16/HiF4/MXFP4 同口径比较；新实验建议优先使用 `exact`。

## 8. 评测与完成门禁

case 名：

```text
hif4_rtn_w4a4_seed42
hif4_rtn_w4a4_seed43
hif4_rtn_w4a4_seed44
```

输入门禁：

```text
videos_quant_sb=200
videos_quant_camera=100
videos_bf16_sb=200
videos_bf16_camera=100
symlink_count=0
```

十维扫描最终门禁：

```text
TOTAL_OK=30
TOTAL_EXPECTED=30
TOTAL_MISSING=0
```

## 9. 公平比较要求

BF16、HiF4、MXFP4、NVFP4 必须分别从同一 BF16 checkpoint 独立初始化；不要在
已量化权重上叠加另一种格式。格式比较还应固定：生成 prompt 子集、seed、尺寸、帧数、
sampler、steps、shift、CFG、模块选择器和 VBench 输入策略。
