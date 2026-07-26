# v4.9.2

## 路径兼容

- 恢复 v4.8.3 的固定项目布局和 Kit 部署方式：`$ROOT/artifacts/<kit>`。
- `REPO` 再次指向 `$ROOT/code/Wan2.2-I2V-A14B-W4A4`；LightX2V 使用独立变量 `LIGHTX2V_REPO`。
- `CKPT` 与 `BASE_MODEL_DIR` 默认恢复为 `$ROOT/weights/Wan2.2-I2V-A14B-bf16`。
- 仅新增 LightX2V 仓库和独立 high/low 蒸馏 DiT 权重路径。
- 共享目录恢复 `calibration/smooth_alpha0p5`、`prepared/smooth_shards` 和 `curvature/opens2v12_smooth_bf16_noqdq_fp32` 层级。
- gate 目录恢复为 `gate`；主状态文件恢复为 `last_2x2_jobs.env`，并兼容读取/写入 v4.9.1 的 `last_method_jobs.env`。

## 实验合同不变

- 不运行 BF16 视频 baseline。
- 共享 Smooth；Hessian 为 Smooth 后 BF16/no-QDQ；最终 runtime 为 HiF4 W4A4。
- 六个活动变体：2×2 主矩阵及 MagR-tree RTN/GPTQ。
- 所有 GPU 和重任务仅通过 sbatch；array 并发硬上限为 4。

## 来源清理

- 删除包内的外部参考文档副本。
- 删除不参与当前 DAG 的旧 Wan collector 归档目录。
- 新增第三方依赖与来源边界说明。
