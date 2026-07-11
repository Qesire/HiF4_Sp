#!/usr/bin/env bash
#SBATCH -J hif4-setup
#SBATCH -p Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:A100:1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o logs/hif4-setup-%j.out
#SBATCH -e logs/hif4-setup-%j.err

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
OUT_ROOT="${OUT_ROOT:-$ROOT/experiments/hif4_setup_gate}"
REPORT="$OUT_ROOT/hif4_setup_report.json"
mkdir -p "$OUT_ROOT" "$ROOT/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${HIF4_ENV:-hif4}"
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$HIF4_REPO/tools/vbench_i2v_hif4:$WAN_REPO:${PYTHONPATH:-}"
export HIF4_MAIN_REPO="$HIF4_REPO"

for path in \
  "$WAN_REPO/tools/generate_vbench_i2v_svdquant.py" \
  "$HIF4_REPO/HiFloat4" \
  "$CKPT_DIR" \
  "$FULL_INFO_JSON"; do
  test -e "$path" || { echo "MISSING $path" >&2; exit 10; }
done

cd "$WAN_REPO"
python tools/generate_vbench_i2v_svdquant.py \
  --mode hif4 \
  --hif4_main_repo "$HIF4_REPO" \
  --hif4_setup_only \
  --hif4_report_json "$REPORT" \
  --full_info_json "$FULL_INFO_JSON" \
  --vbench_root "$VBENCH_ROOT" \
  --ckpt_dir "$CKPT_DIR" \
  --output_dir "$OUT_ROOT/output" \
  --dimensions i2v_subject \
  --start 0 \
  --limit 1 \
  --samples_per_prompt 1 \
  --device_id 0 \
  --offload_model true \
  --t5_cpu \
  --convert_model_dtype \
  --freeze_condition_latent

python - "$REPORT" "${EXPECTED_PER_EXPERT:-400}" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
data = json.load(open(path, encoding="utf-8"))
reports = data["reports"]
for name in ("low_noise_model", "high_noise_model"):
    got = int(reports[name]["selected_count"])
    print(name, "selected_count=", got)
    assert got == expected, (name, got, expected)
stats = data["stats_after_setup"]
print("stats_after_setup=", stats)
assert int(stats["weight_qdq_calls"]) == expected * 2, stats
print("HIF4_SETUP_GATE_OK")
PY
