#!/usr/bin/env bash
#SBATCH -J hif4-w4a4-smoke
#SBATCH -p Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:A100:1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o logs/hif4-w4a4-smoke-%j.out
#SBATCH -e logs/hif4-w4a4-smoke-%j.err

set -Eeuo pipefail
trap 'echo "[ERROR] line=$LINENO cmd=$BASH_COMMAND exit=$?" >&2' ERR
unset PWD OLDPWD || true
cd "${SAFE_CWD:-/home/scc/pb23061276}"
pwd -P >/dev/null

ROOT="${ROOT:-/home/scc/pb23061276/projects/wan-quant}"
WAN_REPO="${WAN_REPO:-$ROOT/code/Wan2.2-I2V-A14B-W4A4}"
HIF4_REPO="${HIF4_REPO:-$ROOT/code/HiF4_Sp}"
CKPT_DIR="${CKPT_DIR:-$ROOT/weights/Wan2.2-I2V-A14B-bf16}"
VBENCH_ROOT="${VBENCH_ROOT:-$ROOT/data/VBench/vbench2_beta_i2v}"
FULL_INFO_JSON="${FULL_INFO_JSON:-$VBENCH_ROOT/vbench2_i2v_full_info.json}"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/hif4_w4a4_single_smoke}"
GEN_DIR="$OUT_ROOT/generated"
SETUP_REPORT="$OUT_ROOT/hif4_setup_report.json"
RUNTIME_REPORT="$OUT_ROOT/hif4_runtime_report.json"
MANIFEST="$OUT_ROOT/manifest.jsonl"
rm -rf "$OUT_ROOT"
mkdir -p "$GEN_DIR" "$ROOT/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${HIF4_ENV:-hif4}"
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HIF4_REPO/tools/vbench_i2v_hif4:$WAN_REPO:${PYTHONPATH:-}"
export HIF4_MAIN_REPO="$HIF4_REPO"

cd "$WAN_REPO"
python tools/generate_vbench_i2v_svdquant.py \
  --mode hif4 \
  --hif4_main_repo "$HIF4_REPO" \
  --hif4_report_json "$SETUP_REPORT" \
  --hif4_runtime_report_json "$RUNTIME_REPORT" \
  --full_info_json "$FULL_INFO_JSON" \
  --vbench_root "$VBENCH_ROOT" \
  --ckpt_dir "$CKPT_DIR" \
  --output_dir "$GEN_DIR" \
  --manifest "$MANIFEST" \
  --dimensions i2v_subject \
  --start 0 \
  --limit 1 \
  --samples_per_prompt 1 \
  --seed 42 \
  --overwrite \
  --size "${VIDEO_SIZE:-832*480}" \
  --frame_num "${FRAME_NUM:-61}" \
  --sample_steps "${SAMPLE_STEPS:-4}" \
  --sample_shift "${SAMPLE_SHIFT:-5.0}" \
  --sample_guide_scale "${SAMPLE_GUIDE_SCALE:-5.0}" \
  --device_id 0 \
  --offload_model true \
  --t5_cpu \
  --convert_model_dtype \
  --freeze_condition_latent

MP4_COUNT=$(find "$GEN_DIR" -maxdepth 1 -type f -name '*.mp4' | wc -l)
echo "MP4_COUNT=$MP4_COUNT"
test "$MP4_COUNT" -eq 1

python - "$SETUP_REPORT" "$RUNTIME_REPORT" "${EXPECTED_PER_EXPERT:-400}" <<'PY'
import json, sys
setup_path, runtime_path, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
setup = json.load(open(setup_path, encoding="utf-8"))
runtime = json.load(open(runtime_path, encoding="utf-8"))
for name in ("low_noise_model", "high_noise_model"):
    assert int(setup["reports"][name]["selected_count"]) == expected
stats = runtime["stats_after_generation"]
print("stats_after_generation=", stats)
assert int(stats["weight_qdq_calls"]) == expected * 2
assert int(stats["activation_qdq_calls"]) > 0
assert int(runtime["generated"]) == 1
assert int(runtime["failed"]) == 0
print("HIF4_FULL_SINGLE_SMOKE_OK")
PY
